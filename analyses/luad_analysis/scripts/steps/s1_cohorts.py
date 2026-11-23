"""Step 1: which tumors express each RBP, and which transcripts are expressed alongside it?

- RBP-positive tumor: summed TPM of the RBP gene's protein_coding isoforms > 1.
- Universe of an RBP: transcripts with TPM >= 1 in >= 50% of its positive tumors and an
  ACGT-only sequence of <= 6,000 nt (CORAL's training length limit). All biotypes are kept.

Writes tables/cohorts.tsv, tables/universes.parquet and tables/cohort_summary.tsv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (LOG_OFFSET, LOGGER, OUT_TABLES, ceil_fraction, load_config, load_log_expression,  # noqa: E402
                    setup_logging, source, to_tpm, tumor_samples, write_table)

TRANSCRIPT_COLUMNS = ["transcript_id", "gene_id", "gene_name", "transcript_biotype", "RNA_id",
                      "sequence", "sequence_eligible", "sequence_length_nt"]


def main() -> None:
    setup_logging()
    config = load_config()
    tumors = tumor_samples()
    transcripts = pd.read_parquet(source("transcripts"), columns=TRANSCRIPT_COLUMNS)
    transcripts = transcripts.sort_values("transcript_id").reset_index(drop=True)
    transcripts["gc_fraction"] = transcripts.sequence.str.count("[GC]") / transcripts.sequence_length_nt
    log_expr = load_log_expression(tumors.sample_id.tolist(), transcripts.transcript_id.tolist())
    LOGGER.info("expression: %d transcripts x %d primary tumors", *log_expr.shape)

    # --- RBP-positive cohorts ---
    cohort_rows = []
    for rbp, spec in config["rbps"].items():
        coding = transcripts.gene_id.eq(spec["gene_id"]) & transcripts.transcript_biotype.eq("protein_coding")
        if not coding.any():
            raise ValueError(f"{rbp}: no protein_coding transcripts for {spec['gene_id']}")
        rbp_tpm = to_tpm(log_expr.to_numpy()[coding.to_numpy()]).sum(axis=0)
        cohort_rows.append(pd.DataFrame({
            "rbp": rbp, "sample_id": tumors.sample_id, "patient_id": tumors.patient_id,
            "rbp_coding_tpm": rbp_tpm, "positive": rbp_tpm > config["cohort"]["rbp_coding_tpm_above"]}))
        LOGGER.info("%s: %d coding isoforms, %d positive tumors", rbp, int(coding.sum()), int(cohort_rows[-1].positive.sum()))
    cohorts = pd.concat(cohort_rows, ignore_index=True)
    observed = cohorts.groupby("rbp", sort=False).positive.sum().to_dict()
    if observed != config["cohort"]["expected_positive"]:
        raise ValueError(f"positive cohorts {observed} differ from config {config['cohort']['expected_positive']}")
    write_table(cohorts, OUT_TABLES / "cohorts.tsv")

    # --- Universes: transcripts typically expressed in each RBP-positive cohort ---
    rule = config["universe"]
    threshold = np.log2(rule["target_tpm_at_least"] + LOG_OFFSET)
    length_ok = transcripts.sequence_length_nt.le(rule["max_rna_nt"]).to_numpy()
    sequence_ok = transcripts.sequence_eligible.to_numpy() & length_ok
    universes, summary = [], []
    for rbp in config["rbps"]:
        positive = cohorts.loc[cohorts.rbp.eq(rbp) & cohorts.positive, "sample_id"].tolist()
        values = log_expr[positive].to_numpy()
        expressed_count = (values >= threshold).sum(axis=1)
        minimum = ceil_fraction(rule["min_fraction_of_positive_tumors"], len(positive))
        expressed = expressed_count >= minimum
        keep = expressed & sequence_ok
        frame = transcripts.loc[keep, ["transcript_id", "RNA_id", "gene_id", "gene_name", "transcript_biotype",
                                       "sequence_length_nt", "gc_fraction"]].copy()
        frame.insert(0, "rbp", rbp)
        frame["fraction_expressed"] = expressed_count[keep] / len(positive)
        frame["mean_log2_tpm1"] = np.log2(to_tpm(values[keep]) + 1.0).mean(axis=1)
        universes.append(frame)
        summary.append({"rbp": rbp, "positive_tumors": len(positive), "min_tumors_expressed": minimum,
                        "expressed_transcripts": int(expressed.sum()),
                        "excluded_longer_than_max": int((expressed & ~length_ok).sum()),
                        "excluded_non_acgt": int((expressed & length_ok & ~sequence_ok).sum()),
                        "universe_transcripts": int(keep.sum()), "universe_genes": frame.gene_id.nunique()})
        LOGGER.info("%s universe: %d transcripts from %d genes", rbp, summary[-1]["universe_transcripts"], summary[-1]["universe_genes"])
    write_table(pd.concat(universes, ignore_index=True), OUT_TABLES / "universes.parquet")
    write_table(pd.DataFrame(summary), OUT_TABLES / "cohort_summary.tsv")


if __name__ == "__main__":
    main()
