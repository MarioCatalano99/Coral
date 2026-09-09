#!/usr/bin/env python3
"""Contextualize literature examples without treating them as independent validation."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from _analysis_io import (ANALYSIS_ROOT, CONFIG_PATH, DATA_ROOT, RESULTS_ROOT, load_config,
                          provenance, read_exposure, read_predictions, sha256_file, staged_outputs, verify_manifest, write_table)
from _rankings import rank_transcripts, gene_max_rankings

ANCHOR_COLUMNS = [
    "anchor_order",
    "rbp_gene",
    "Protein_id",
    "target_gene_name",
    "pmid",
    "doi",
    "citation_url",
    "evidence_context",
    "evidence_type",
    "evidence_summary",
]
REPRESENTATIVE_RULE = (
    "highest tumor_median_tpm among sequence-eligible GENCODE v23 transcripts; "
    "transcript_id ascending breaks ties"
)


def read_anchors(path: Path) -> pd.DataFrame:
    """Read and validate the pre-registered literature target table."""
    anchors = pd.read_csv(path, sep="\t", dtype={"pmid": str, "doi": str})
    if list(anchors.columns) != ANCHOR_COLUMNS:
        raise ValueError(f"unexpected literature-anchor schema: {list(anchors.columns)}")
    if anchors[ANCHOR_COLUMNS].isna().any().any():
        raise ValueError("literature-anchor configuration contains missing values")
    if anchors["anchor_order"].duplicated().any():
        raise ValueError("anchor_order values must be unique")
    if anchors[["Protein_id", "target_gene_name"]].duplicated().any():
        raise ValueError("literature anchors must define unique RBP-target pairs")
    return anchors.sort_values("anchor_order", kind="stable").reset_index(drop=True)


def select_representative_transcripts(
    anchors: pd.DataFrame, transcripts: pd.DataFrame
) -> pd.DataFrame:
    """Select one expression-representative transcript for each target gene."""
    targets = set(anchors["target_gene_name"])
    candidates = transcripts.loc[
        transcripts["gene_name"].isin(targets) & transcripts["sequence_eligible"]
    ].copy()
    gene_counts = candidates.groupby("gene_name")["gene_id"].nunique()
    ambiguous = gene_counts.loc[gene_counts != 1]
    missing = targets - set(candidates["gene_name"])
    if missing or not ambiguous.empty:
        raise ValueError(
            f"target mapping failure; missing={sorted(missing)}, "
            f"ambiguous={ambiguous.to_dict()}"
        )
    representatives = (
        candidates.sort_values(
            ["gene_name", "tumor_median_tpm", "transcript_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("gene_name", sort=False)
        .head(1)
        .rename(
            columns={
                "gene_name": "target_gene_name",
                "gene_id": "target_gene_id",
                "transcript_id": "representative_transcript_id",
                "primary_eligible": "representative_primary_eligible",
                "sensitivity_eligible": "representative_sensitivity_eligible",
            }
        )
    )
    columns = [
        "target_gene_name",
        "target_gene_id",
        "representative_transcript_id",
        "RNA_id",
        "sequence_length_nt",
        "tumor_median_tpm",
        "tumor_tpm_ge_1_count",
        "normal_tpm_ge_1_count",
        "representative_primary_eligible",
        "representative_sensitivity_eligible",
    ]
    return anchors.merge(representatives[columns], on="target_gene_name", validate="many_to_one")


def build_output(
    anchors: pd.DataFrame,
    transcripts: pd.DataFrame,
    proteins: pd.DataFrame,
    predictions: pd.DataFrame,
    exposure: pd.DataFrame,
    eligibility: pd.DataFrame,
    top_fraction: float = 0.01,
) -> pd.DataFrame:
    """Map anchors to representative isoforms and attach CORAL ranks and exposure."""
    protein_map = proteins[["gene_symbol", "Protein_id"]].drop_duplicates()
    observed = anchors[["rbp_gene", "Protein_id"]].drop_duplicates()
    expected = protein_map.rename(columns={"gene_symbol": "rbp_gene"})
    if not observed.merge(expected, how="outer", indicator=True).query(
        "_merge != 'both'"
    ).empty:
        raise ValueError("anchor RBP names and protein IDs do not match the protein manifest")

    mapped = select_representative_transcripts(anchors, transcripts)
    transcript_ranks = rank_transcripts(transcripts, eligibility, predictions, top_fraction=top_fraction)
    gene_ranks = gene_max_rankings(transcript_ranks)
    mapped = mapped.drop(columns=["representative_primary_eligible", "representative_sensitivity_eligible"])
    mapped = mapped.merge(eligibility[["transcript_id", "Protein_id", "primary_eligible", "sensitivity_eligible"]].rename(columns={"transcript_id": "representative_transcript_id", "primary_eligible": "representative_primary_eligible", "sensitivity_eligible": "representative_sensitivity_eligible"}), on=["representative_transcript_id", "Protein_id"], how="left", validate="one_to_one")
    target_scores = mapped.merge(
        predictions, on=["RNA_id", "Protein_id"], how="left", validate="many_to_one"
    )
    if target_scores["logit_margin"].isna().any():
        missing = target_scores.loc[
            target_scores["logit_margin"].isna(),
            ["rbp_gene", "target_gene_name", "representative_transcript_id"],
        ]
        raise ValueError(
            "representative literature isoforms lack production scores; score them as "
            f"literature-only overrides first: {missing.to_dict('records')}"
        )

    transcript_lookup = transcript_ranks[
        ["Protein_id", "transcript_id", "rank", "ranking_total", "percentile", "top_candidate"]
    ].rename(
        columns={
            "transcript_id": "representative_transcript_id",
            "rank": "primary_transcript_rank",
            "ranking_total": "primary_transcript_total",
            "percentile": "primary_transcript_percentile",
            "top_candidate": "top1pct_candidate",
        }
    )
    target_scores = target_scores.merge(
        transcript_lookup,
        on=["Protein_id", "representative_transcript_id"],
        how="left",
        validate="one_to_one",
    )
    target_scores["top1pct_candidate"] = target_scores.top1pct_candidate.fillna(False).astype(bool)
    # Out-of-universe anchors retain a virtual insertion rank, never candidate status.
    target_scores["virtual_insertion_rank"] = pd.Series(pd.NA, index=target_scores.index, dtype="Int64")
    for index, row in target_scores.loc[~target_scores.representative_primary_eligible].iterrows():
        background = transcript_ranks.loc[transcript_ranks.Protein_id == row.Protein_id]
        rank = 1 + int((background.logit_margin > row.logit_margin).sum())
        target_scores.loc[index, "virtual_insertion_rank"] = rank
        target_scores.loc[index, "primary_transcript_total"] = len(background)
    target_scores["rank_status"] = target_scores.representative_primary_eligible.map({True: "official_primary", False: "outside_primary_virtual_only"})

    gene_lookup = gene_ranks[
        [
            "Protein_id",
            "gene_id",
            "gene_score_transcript_id",
            "logit_margin",
            "rank",
            "ranking_total",
            "percentile",
        ]
    ].rename(
        columns={
            "gene_id": "target_gene_id",
            "logit_margin": "primary_gene_logit_margin",
            "rank": "primary_gene_rank",
            "ranking_total": "primary_gene_total",
            "percentile": "primary_gene_percentile",
        }
    )
    target_scores = target_scores.merge(
        gene_lookup, on=["Protein_id", "target_gene_id"], how="left", validate="one_to_one"
    )
    exposure_columns = [
        "RNA_id",
        "Protein_id",
        "exposure_class",
        "training_row_index",
        "training_label",
        "rna_identity",
        "protein_identity",
    ]
    target_scores = target_scores.merge(
        exposure[exposure_columns],
        how="left",
        on=["RNA_id", "Protein_id"],
        validate="many_to_one",
    )
    if target_scores.exposure_class.isna().any() or len(target_scores) != len(anchors):
        raise ValueError("literature anchor exposure join lost rows or has missing classifications")
    target_scores["representative_selection_rule"] = REPRESENTATIVE_RULE
    target_scores["representative_is_gene_score_isoform"] = (
        target_scores["representative_transcript_id"]
        == target_scores["gene_score_transcript_id"]
    )
    return target_scores.sort_values("anchor_order", kind="stable").reset_index(drop=True)


def main() -> None:
    """Publish literature anchors with complete exposure and explicit universe status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--anchors", type=Path, default=ANALYSIS_ROOT / "config/literature_anchors.tsv")
    args = parser.parse_args()
    config = load_config(args.config)
    verify_manifest(args.data_dir / "luad_inputs.json", args.data_dir)
    exposure_dir = args.data_dir / "luad_training_exposure"
    exposure, exposure_manifest = read_exposure(args.data_dir)
    predictions, manifests = read_predictions(args.data_dir / "luad_predictions", args.data_dir / "luad_rna_sequences.parquet", args.data_dir / "luad_proteins.tsv")
    output = build_output(read_anchors(args.anchors), pd.read_parquet(args.data_dir / "luad_transcripts.parquet"), pd.read_csv(args.data_dir / "luad_proteins.tsv", sep="\t"), predictions, exposure, pd.read_parquet(args.data_dir / "luad_eligibility.parquet"), config["top_fraction"])
    metadata = provenance([args.config, args.anchors, args.data_dir / "luad_inputs.json", exposure_dir / "run_manifest.json", *manifests], config)
    with staged_outputs(args.output_dir, args.output_dir / "run_manifests/literature_anchors.json", metadata) as stage:
        write_table(output, stage / "tables/literature_anchors.tsv")
        write_table(pd.read_csv(exposure_dir / "training_exposure_summary.tsv", sep="\t"), stage / "tables/training_exposure_summary.tsv")


if __name__ == "__main__":
    main()
