#!/usr/bin/env python3
"""Contextualize fixed RBP-specific candidate isoforms with sample expression."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from _analysis_io import (CONFIG_PATH, DATA_ROOT, RESULTS_ROOT, load_config, provenance,
                          staged_outputs, verify_manifest, write_table)
from _expression_context import (build_sample_edge_states, build_anchor_context,
    build_cohort_summary, build_edge_table, build_paired_outputs, build_recurrent_targets,
    build_rbp_subsets)

def validate_samples(expression: Path, samples: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Validate sample counts, order, types, and pairing."""
    required = {"sample_id", "sample_type", "patient_id", "pair_status", "expression_column_index"}
    if required - set(samples.columns) or samples["sample_id"].duplicated().any():
        raise ValueError("invalid sample manifest schema or duplicate IDs")
    samples = samples.sort_values("expression_column_index", kind="stable").reset_index(drop=True)
    if samples["expression_column_index"].tolist() != list(range(len(samples))):
        raise ValueError("sample expression indexes are not contiguous")
    expected = config["expected_samples"]
    counts = samples["sample_type"].value_counts().to_dict()
    if counts != {"primary_tumor": expected["tumors"], "solid_tissue_normal": expected["normals"]}:
        raise ValueError(f"unexpected sample-type counts: {counts}")
    if samples[["patient_id", "sample_type"]].duplicated().any():
        raise ValueError("multiple samples per patient and cohort")
    matched = samples.loc[samples["pair_status"] == "matched"]
    pair_counts = matched.groupby("patient_id")["sample_type"].nunique()
    if len(pair_counts) != expected["matched_pairs"] or not pair_counts.eq(2).all():
        raise ValueError("matched samples do not form 58 complete patient pairs")
    columns = pq.ParquetFile(expression).schema_arrow.names
    if columns != ["transcript_id", *samples["sample_id"].tolist()]:
        raise ValueError("expression sample columns do not match the sample manifest order")
    return samples


def main() -> None:
    """Publish sample states and descriptive/paired QC tables from validated rankings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    stages = [(args.data_dir / "luad_inputs.json", args.data_dir),
              (args.output_dir / "run_manifests/rankings.json", args.output_dir),
              (args.output_dir / "run_manifests/literature_anchors.json", args.output_dir)]
    for path, root in stages:
        if verify_manifest(path, root)["config"] != config:
            raise ValueError(f"stage configuration differs: {path}")
    expression_path = args.data_dir / "luad_expression.parquet"
    samples = validate_samples(expression_path, pd.read_csv(args.data_dir / "luad_samples.tsv", sep="\t"), config)
    proteins = pd.read_csv(args.data_dir / "luad_proteins.tsv", sep="\t")
    ranks = pd.read_parquet(args.output_dir / "data/transcript_rankings.parquet")
    primary = ranks.loc[ranks.scope == "primary"]
    top = primary.loc[primary.top_candidate].rename(columns={"rank": "primary_transcript_rank", "percentile": "primary_transcript_percentile"}).copy()
    anchors = pd.read_csv(args.output_dir / "tables/literature_anchors.tsv", sep="\t")
    required = sorted(set(top.transcript_id) | set(anchors.representative_transcript_id))
    expression = pd.read_parquet(expression_path, filters=[("transcript_id", "in", required)]).set_index("transcript_id")
    if set(expression.index) != set(required) or expression.index.duplicated().any() or not np.isfinite(expression.to_numpy()).all():
        raise ValueError("selected expression is incomplete, duplicated or non-finite")
    proxy = pd.read_csv(args.data_dir / "luad_rbp_expression.tsv", sep="\t")
    if proxy[["sample_id", "Protein_id"]].duplicated().any() or len(proxy) != len(samples) * len(proteins):
        raise ValueError("RBP expression proxy has incomplete coverage")
    rbp_expression = {protein_id: group.set_index("sample_id")[["rbp_gene_tpm", "rbp_available"]] for protein_id, group in proxy.groupby("Protein_id")}
    states, sample_context = build_sample_edge_states(top, expression, samples, rbp_expression, config["target_tpm_cutoff"])
    if len(states) != len(samples) * len(top):
        raise ValueError("sample-edge coverage is incomplete")
    paired_changes, paired_tests = build_paired_outputs(sample_context, proteins, samples, config)
    tables = {
        "edges": build_edge_table(top, states, proteins, samples, config["bh_alpha"]),
        "samples": sample_context,
        "cohorts": build_cohort_summary(sample_context, proteins),
        "paired_changes": paired_changes,
        "paired_tests": paired_tests,
        "recurrent_targets": build_recurrent_targets(top, states, proteins, samples, sample_context),
        "rbp_subsets": build_rbp_subsets(sample_context, proteins),
        "literature_anchors": build_anchor_context(anchors, expression, rbp_expression, samples, config["target_tpm_cutoff"]),
    }
    boundaries = []
    for protein_id, group in primary.groupby("Protein_id"):
        cutoff = group.loc[group["rank"] == group.top_count].iloc[0].logit_margin
        ties = group.loc[group.logit_margin == cutoff]
        boundaries.append({"Protein_id": protein_id, "cutoff_logit_margin": float(cutoff), "cutoff_tie_size": len(ties), "cutoff_tie_selected": int(ties.top_candidate.sum())})
    metadata = provenance([args.config, *[path for path, _ in stages]], config)
    metadata.update({"counts": {"candidate_edges": len(top), "samples": len(samples), "sample_edge_states": len(states)},
        "cutoff_boundaries": boundaries,
        "definitions": {"active": "candidate AND target isoform TPM >= target cutoff AND protein-coding RBP TPM sum >= RBP cutoff",
                        "paired_tests": "secondary oncofetal QC; shared gates and selection in this cohort preclude independent target validation",
                        "conditional_prevalence": "undefined if no samples pass the relevant RBP gate"}})
    with staged_outputs(args.output_dir, args.output_dir / "run_manifests/expression_context.json", metadata) as stage:
        write_table(states, stage / "data/sample_edge_states.parquet")
        for name, frame in tables.items():
            write_table(frame, stage / f"tables/expression_context_{name}.tsv")
    logging.info("Published %d candidate edges and %d sample-edge states", len(top), len(states))


if __name__ == "__main__":
    main()
