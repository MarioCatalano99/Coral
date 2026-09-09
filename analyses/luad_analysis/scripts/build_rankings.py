#!/usr/bin/env python3
"""Publish RBP-specific isoform rankings and CORAL cross-RBP comparisons."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from _analysis_io import (CONFIG_PATH, DATA_ROOT, RESULTS_ROOT, load_config, provenance,
                          read_exposure, read_predictions, staged_outputs, verify_manifest, write_table)
from _rankings import compare_rankings, gene_max_rankings, rank_transcripts


def main() -> None:
    """Validate complete inputs and publish rankings as one committed stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    prepared = verify_manifest(args.data_dir / "luad_inputs.json", args.data_dir)
    if prepared["config"] != config:
        raise ValueError("prepared inputs use a different analysis configuration")
    transcripts = pd.read_parquet(args.data_dir / "luad_transcripts.parquet")
    eligibility = pd.read_parquet(args.data_dir / "luad_eligibility.parquet")
    predictions, manifests = read_predictions(args.data_dir / "luad_predictions", args.data_dir / "luad_rna_sequences.parquet", args.data_dir / "luad_proteins.tsv")
    ranks = pd.concat([rank_transcripts(transcripts, eligibility, predictions, scope, config["top_fraction"]) for scope in ("primary", "sensitivity")], ignore_index=True)
    exposure, exposure_manifest = read_exposure(args.data_dir)
    ranks = ranks.merge(exposure.drop(columns="primary_eligible"), on=["RNA_id", "Protein_id"], how="left", validate="many_to_one")
    if ranks.exposure_class.isna().any():
        raise ValueError("eligible isoforms lack training-exposure classifications")
    correlations, overlap, contrasts = compare_rankings(ranks)
    exposure_summary = []
    for selection, subset in (("full_universe", ranks), ("top_candidate", ranks.loc[ranks.top_candidate])):
        counts_by_class = subset.groupby(["scope", "Protein_id", "exposure_class"]).size().rename("isoform_count").reset_index()
        counts_by_class["selection"] = selection
        counts_by_class["denominator"] = counts_by_class.groupby(["scope", "Protein_id"]).isoform_count.transform("sum")
        counts_by_class["fraction"] = counts_by_class.isoform_count / counts_by_class.denominator
        exposure_summary.append(counts_by_class)
    counts = ranks.groupby(["scope", "Protein_id"]).agg(isoforms=("transcript_id", "size"), sequences=("RNA_id", "nunique"), genes=("gene_id", "nunique"), top_count=("top_candidate", "sum"), tie_inclusive_count=("tie_inclusive_candidate", "sum")).reset_index()
    stability = []
    for protein_id, group in ranks.groupby("Protein_id"):
        a, b = [set(group.loc[(group.scope == scope) & group.top_candidate, "transcript_id"]) for scope in ("primary", "sensitivity")]
        stability.append({"Protein_id": protein_id, "primary_top": len(a), "sensitivity_top": len(b), "intersection": len(a & b), "jaccard": len(a & b) / len(a | b), "primary_top_retained_fraction": len(a & b) / len(a)})
    metadata = provenance([args.config, args.data_dir / "luad_inputs.json", *manifests, exposure_manifest], config)
    metadata["analysis_unit"] = "isoform; genes are descriptive maximum-isoform summaries only"
    metadata["comparisons"] = "CORAL across RBPs only; full common-isoform/sequence rankings; own-universe and common-universe average-rank percentiles"
    with staged_outputs(args.output_dir, args.output_dir / "run_manifests/rankings.json", metadata) as stage:
        write_table(ranks, stage / "data/transcript_rankings.parquet")
        write_table(pd.concat(exposure_summary, ignore_index=True), stage / "tables/ranking_training_exposure.tsv")
        write_table(gene_max_rankings(ranks.loc[ranks.scope == "primary"]), stage / "data/primary_gene_max_rankings.parquet")
        write_table(contrasts, stage / "data/within_rbp_percentile_contrasts.parquet")
        for name, table in (("ranking_universes", counts), ("full_ranking_correlations", correlations), ("target_set_overlap", overlap), ("ranking_sensitivity", pd.DataFrame(stability))):
            write_table(table, stage / f"tables/{name}.tsv")
    logging.info("Published %d isoform-RBP ranking rows", len(ranks))


if __name__ == "__main__":
    main()
