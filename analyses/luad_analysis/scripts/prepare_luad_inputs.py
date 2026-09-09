#!/usr/bin/env python3
"""Prepare RBP-specific isoform universes from protein-coding RBP expression."""

from __future__ import annotations

import argparse
import gzip
import logging
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _analysis_io import CONFIG_PATH, DATA_ROOT, load_config, provenance, sha256_file, staged_outputs, write_table
from _luad_data import (
    GENCODE_GTF, GENCODE_FASTA, SOURCE_DIR, TUMOR_EXPRESSION, NORMAL_EXPRESSION,
    NORMAL_SOURCE_MANIFEST, XENA_METADATA, UNIPROT_RELEASE, build_sample_rows,
    expression_summaries, load_expression, parse_fasta, read_expression_axes,
    sha256_text, write_expression_parquet,
)
from download_luad_sources import load_protein_rows, validate_sources

LOGGER = logging.getLogger("coral.luad.prepare")


def load_gencode() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Load one annotation and one sequence for every GENCODE v23 transcript."""
    annotations: dict[str, dict[str, str]] = {}
    with gzip.open(GENCODE_GTF, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 9:
                raise ValueError(f"{GENCODE_GTF}:{line_number}: expected 9 fields")
            if fields[2] != "transcript":
                continue
            attributes = {}
            for item in fields[8].rstrip(";\n").split(";"):
                key, separator, value = item.strip().partition(" ")
                if separator:
                    attributes[key] = value.strip().strip('"')
            transcript_id = attributes.get("transcript_id")
            required = ("gene_id", "gene_name", "gene_type", "transcript_type")
            if not transcript_id or any(key not in attributes for key in required):
                raise ValueError(f"{GENCODE_GTF}:{line_number}: incomplete transcript attributes")
            if transcript_id in annotations:
                raise ValueError(f"duplicate GTF transcript: {transcript_id}")
            annotations[transcript_id] = {
                "gene_id": attributes["gene_id"],
                "gene_name": attributes["gene_name"],
                "gene_biotype": attributes["gene_type"],
                "transcript_biotype": attributes["transcript_type"],
            }
    sequences: dict[str, str] = {}
    for identifier, _, sequence in parse_fasta(GENCODE_FASTA):
        if identifier in sequences:
            raise ValueError(f"duplicate GENCODE FASTA transcript: {identifier}")
        sequences[identifier] = sequence
    return annotations, sequences


def build_eligibility(
    transcripts: pd.DataFrame, logged: np.ndarray, samples: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply prevalence independently within each coding-RBP-positive tumor set."""
    sample_ids = samples.sample_id.tolist()
    tumors = samples.sample_type.eq("primary_tumor").to_numpy()
    eligibility = []
    expression = []
    for protein_id, spec in config["rbp_genes"].items():
        gene = transcripts.gene_id.eq(spec["gene_id"])
        if gene.sum() != spec["transcript_count"] or not transcripts.loc[gene, "gene_name"].eq(spec["gene_symbol"]).all():
            raise ValueError(f"RBP annotation mismatch: {protein_id}")
        coding = gene & transcripts.transcript_biotype.eq("protein_coding")
        if not coding.any():
            raise ValueError(f"no protein-coding transcripts: {protein_id}")
        coding_tpm = np.maximum(np.exp2(logged[:, coding].astype(np.float64)) - 0.001, 0).sum(axis=1)
        available = coding_tpm >= config["rbp_tpm_cutoff"]
        positive = available & tumors
        denominator = int(positive.sum())
        minimum = math.ceil(config["prevalence_fraction"] * denominator)
        frame = transcripts[["transcript_id", "RNA_id"]].copy()
        frame["Protein_id"] = protein_id
        frame["rbp_positive_tumors"] = denominator
        frame["minimum_positive_tumors"] = minimum
        for scope, cutoff in (("primary", config["target_tpm_cutoff"]), ("sensitivity", config["sensitivity_tpm_cutoff"])):
            counts = (logged[positive] >= np.log2(cutoff + 0.001)).sum(axis=0)
            frame[f"{scope}_positive_tumor_count"] = counts
            frame[f"{scope}_eligible"] = transcripts.sequence_eligible & (counts >= minimum) & (denominator > 0)
        frame["exclusion_reason"] = np.where(~transcripts.sequence_eligible, "sequence_ineligible",
            np.where(denominator == 0, "no_rbp_positive_tumors", np.where(frame.primary_eligible, "", "below_rbp_positive_prevalence")))
        eligibility.append(frame)
        context = samples.copy()
        context["Protein_id"] = protein_id
        context["rbp_gene"] = spec["gene_symbol"]
        context["rbp_gene_tpm"] = coding_tpm
        context["rbp_available"] = available
        context["coding_transcript_ids"] = ";".join(transcripts.loc[coding, "transcript_id"])
        expression.append(context)
        LOGGER.info("%s: %d RBP-positive tumors, minimum %d; %d primary isoforms", spec["gene_symbol"], denominator, minimum, frame.primary_eligible.sum())
    return pd.concat(eligibility, ignore_index=True), pd.concat(expression, ignore_index=True)


def main() -> None:
    """Build and transactionally publish the current official input generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.batch_rows < 1:
        parser.error("batch rows must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    normal_source, _, uniprot = validate_sources()
    tids, tumors = read_expression_axes(TUMOR_EXPRESSION)
    normal_tids, normals = read_expression_axes(NORMAL_EXPRESSION)
    if tids != normal_tids or normals != normal_source["samples"]:
        raise ValueError("source expression axes disagree")
    expected = config["expected_samples"]
    if len(tumors) != expected["tumors"] or len(normals) != expected["normals"]:
        raise ValueError("unexpected cohort sizes")
    samples = pd.DataFrame(build_sample_rows(tumors, normals))
    annotations, sequences = load_gencode()
    if set(tids) - annotations.keys() or set(tids) - sequences.keys():
        raise ValueError("incomplete GENCODE mapping")
    transcripts = pd.DataFrame([{"transcript_id": tid, **annotations[tid], "sequence": sequences[tid]} for tid in tids])
    transcripts["transcript_id_unversioned"] = transcripts.transcript_id.str.split(".").str[0]
    transcripts["gene_id_unversioned"] = transcripts.gene_id.str.split(".").str[0]
    transcripts["sequence_length_nt"] = transcripts.sequence.str.len()
    transcripts["rna_sequence_sha256"] = transcripts.sequence.map(sha256_text)
    transcripts["RNA_id"] = "rna_" + transcripts.rna_sequence_sha256
    transcripts["sequence_eligible"] = transcripts.sequence.map(lambda seq: bool(seq) and set(seq) <= set("ACGT")) & transcripts.sequence_length_nt.le(config["max_rna_nt"])
    inputs = [args.config, TUMOR_EXPRESSION, NORMAL_EXPRESSION, NORMAL_SOURCE_MANIFEST, GENCODE_GTF, GENCODE_FASTA, XENA_METADATA, UNIPROT_RELEASE]
    inputs.extend(sorted(SOURCE_DIR.glob("uniprot_*.fasta")))
    metadata = provenance(inputs, config)
    metadata["definitions"] = {"rbp_proxy": config["rbp_proxy"], "eligibility": "TPM cutoff in ceil(prevalence_fraction * RBP-positive tumors), independently per RBP; no normal rescue",
                               "transcript_flags": "union across RBP-specific universes; use luad_eligibility.parquet for rankings",
                               "expression_transform": "max(2**logged - 0.001, 0)"}
    with staged_outputs(args.output_dir, args.output_dir / "luad_inputs.json", metadata) as stage:
        with tempfile.TemporaryDirectory(prefix="luad-prepare-") as temp:
            matrix = load_expression(tids, tumors, normals, Path(temp) / "expression.f32")
            for name, values in expression_summaries(matrix, len(tumors), args.batch_rows).items():
                transcripts[name] = values
            eligibility, context = build_eligibility(transcripts, matrix, samples, config)
            union = eligibility.groupby("transcript_id")[["primary_eligible", "sensitivity_eligible"]].any()
            for name in union:
                transcripts[name] = transcripts.transcript_id.map(union[name])
            # Descriptive anchors may require scores outside both official universes.
            from build_literature_anchors import read_anchors, select_representative_transcripts
            from _analysis_io import ANALYSIS_ROOT
            anchor_path = ANALYSIS_ROOT / "config/literature_anchors.tsv"
            representatives = select_representative_transcripts(read_anchors(anchor_path), transcripts)
            inference_ids = set(transcripts.loc[transcripts.sensitivity_eligible, "RNA_id"]) | set(representatives.RNA_id)
            rnas = transcripts.loc[transcripts.RNA_id.isin(inference_ids), ["RNA_id", "sequence", "rna_sequence_sha256", "sequence_length_nt", "primary_eligible", "sensitivity_eligible"]]
            flags = rnas.groupby("RNA_id")[["primary_eligible", "sensitivity_eligible"]].any()
            rnas = rnas.drop_duplicates("RNA_id").set_index("RNA_id")
            rnas[["primary_eligible", "sensitivity_eligible"]] = flags
            rnas = rnas.reset_index().rename(columns={"sequence": "RNA_seqs", "sequence_length_nt": "rna_nt_length"}).sort_values("RNA_id").reset_index(drop=True)
            rnas["literature_only"] = ~rnas.sensitivity_eligible
            write_expression_parquet(tids, samples.sample_id.tolist(), matrix, args.batch_rows, stage / "luad_expression.parquet")
            del matrix
        for name, frame in {"luad_samples.tsv": samples, "luad_transcripts.parquet": transcripts,
                            "luad_eligibility.parquet": eligibility, "luad_rbp_expression.tsv": context,
                            "luad_rna_sequences.parquet": rnas, "luad_proteins.tsv": pd.DataFrame(load_protein_rows(uniprot))}.items():
            write_table(frame, stage / name)
        metadata["inputs"][str(anchor_path.resolve())] = sha256_file(anchor_path)
        metadata["counts"] = {"transcripts": len(transcripts), "samples": len(samples), "inference_sequences": len(rnas),
                              "inference_pairs": len(rnas) * len(config["rbp_genes"]),
                              "primary_per_rbp": {k: int(v) for k, v in eligibility.groupby("Protein_id").primary_eligible.sum().items()},
                              "sensitivity_per_rbp": {k: int(v) for k, v in eligibility.groupby("Protein_id").sensitivity_eligible.sum().items()}}
    LOGGER.info("Prepared official LUAD inputs: %s", metadata["counts"])


if __name__ == "__main__":
    main()
