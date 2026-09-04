#!/usr/bin/env python3
"""Prepare the frozen LUAD expression universe and CORAL inference inputs.

This is the offline, deterministic transformation stage of the LUAD analysis.
It combines the supplied 513-tumor matrix with the 59 normals downloaded by
``download_luad_sources.py``, annotates every transcript from GENCODE v23, and
applies the pre-registered sequence and expression-prevalence rules.

Inputs:
    data/TCGA/LUAD_isoforms.tsv
    data/TCGA/luad_sources/LUAD_normal_isoforms.tsv.gz
    data/TCGA/luad_sources/gencode.v23.{annotation.gtf,transcripts.fa}.gz
    data/TCGA/luad_sources/uniprot_*.fasta and uniprot_release.json

Outputs under ``data/TCGA/``:
    luad_expression.parquet
        Transcript rows by 513 tumor and 59 normal columns, retaining the Xena
        ``log2(TPM + 0.001)`` unit as float32.
    luad_samples.tsv
        Sample type, participant ID, pairing status, and expression-column order.
    luad_transcripts.parquet
        All 198,619 transcript IDs with annotation, sequence, expression
        summaries, eligibility flags, exclusion reason, and ``RNA_id`` mapping.
    luad_rna_sequences.parquet
        The 85,851 unique sensitivity-universe sequences, sorted by ``RNA_id``.
        Its canonical ``RNA_id,RNA_seqs`` columns can be passed directly to the
        Step-4 all-vs-all API; ``primary_eligible`` selects the 63,881 primary
        sequences after the single superset inference run.
    luad_proteins.tsv
        The three proteins in fixed IGF2BP1, IGF2BP3, LIN28B order. Canonical
        ``Protein_id,Prot_seqs`` columns are followed by provenance fields.
    luad_inputs.json
        Source/output hashes, dimensions, thresholds, counts, script hashes,
        software versions, and the deterministic preparation run ID.

Large matrices are processed in bounded blocks and outputs are atomically
renamed. The command fails before completion on provenance, schema, sequence,
sample, numerical, or resolved-count mismatches. It also reopens and validates
all generated tables, so no separate verification command is required.

Run from the repository root in the ``newrpidnabert`` environment:

    python analyses/luad_analysis/scripts/prepare_luad_inputs.py
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa

from _luad_data import (
    EXPECTED,
    EXPRESSION_PARQUET,
    GENCODE_BASE_URL,
    GENCODE_FASTA,
    GENCODE_GTF,
    MAX_RNA_NT,
    NORMAL_EXPRESSION,
    NORMAL_PREVALENCE_MINIMUM,
    NORMAL_SOURCE_MANIFEST,
    PRIMARY_TPM_CUTOFF,
    PROTEIN_MANIFEST,
    RNA_MANIFEST,
    RUN_MANIFEST,
    SAMPLE_MANIFEST,
    SENSITIVITY_TPM_CUTOFF,
    SOURCE_DIR,
    TRANSCRIPT_MANIFEST,
    TUMOR_EXPRESSION,
    TUMOR_PREVALENCE_MINIMUM,
    UNIPROT_RELEASE,
    XENA_DATASET,
    XENA_METADATA,
    XENA_METADATA_URL,
    XENA_PROBEMAP_URL,
    assert_count,
    build_sample_rows,
    expression_summaries,
    load_expression,
    md5_file,
    parse_fasta,
    read_expression_axes,
    sha256_file,
    sha256_text,
    validate_outputs,
    write_expression_parquet,
    write_json_atomic,
    write_parquet_atomic,
    write_tsv,
)
from download_luad_sources import load_protein_rows, validate_sources


LOGGER = logging.getLogger("luad.prepare")


def parse_args() -> argparse.Namespace:
    """Parse bounded-table preparation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-rows", type=int, default=4_096,
                        help="Transcript rows per in-memory block (default: 4096).")
    return parser.parse_args()


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


def build_sequence_tables(
    transcript_ids: list[str],
    annotations: dict[str, dict[str, str]],
    sequence_by_id: dict[str, str],
    summaries: dict[str, np.ndarray],
) -> tuple[pa.Table, pa.Table, dict[str, Any]]:
    """Build transcript-level metadata and the deduplicated inference universe."""
    missing_annotation = [value for value in transcript_ids if value not in annotations]
    missing_sequence = [value for value in transcript_ids if value not in sequence_by_id]
    if missing_annotation or missing_sequence:
        raise ValueError(
            f"GENCODE mapping incomplete: {len(missing_annotation)} annotations and "
            f"{len(missing_sequence)} sequences missing"
        )
    annotation_rows = [annotations[value] for value in transcript_ids]
    sequences = [sequence_by_id[value] for value in transcript_ids]
    lengths = np.fromiter((len(value) for value in sequences), dtype=np.int32)
    alphabet_valid = np.fromiter(
        (bool(value) and set(value) <= set("ACGT") for value in sequences), dtype=np.bool_
    )
    within_limit = lengths <= MAX_RNA_NT
    sequence_eligible = within_limit & alphabet_valid
    primary_expression = (
        summaries["tumor_tpm_ge_1_count"] >= TUMOR_PREVALENCE_MINIMUM
    ) | (summaries["normal_tpm_ge_1_count"] >= NORMAL_PREVALENCE_MINIMUM)
    sensitivity_expression = (
        summaries["tumor_tpm_ge_0_5_count"] >= TUMOR_PREVALENCE_MINIMUM
    ) | (summaries["normal_tpm_ge_0_5_count"] >= NORMAL_PREVALENCE_MINIMUM)
    primary = sequence_eligible & primary_expression
    sensitivity = sequence_eligible & sensitivity_expression

    assert_count("transcripts at most 6000 nt", int(within_limit.sum()), "at_most_6000_nt")
    assert_count("transcripts over 6000 nt", int((~within_limit).sum()), "over_6000_nt")
    assert_count(
        "invalid-alphabet transcripts within limit",
        int((within_limit & ~alphabet_valid).sum()),
        "invalid_alphabet_within_limit",
    )
    assert_count("primary eligible transcripts", int(primary.sum()), "primary_transcripts")
    assert_count("sensitivity eligible transcripts", int(sensitivity.sum()), "sensitivity_transcripts")

    gene_ids = [row["gene_id"] for row in annotation_rows]
    sequence_hashes = [sha256_text(value) for value in sequences]
    rna_ids = [f"rna_{digest}" for digest in sequence_hashes]
    primary_indexes, sensitivity_indexes = np.flatnonzero(primary), np.flatnonzero(sensitivity)
    assert_count("primary genes", len({gene_ids[i] for i in primary_indexes}), "primary_genes")
    assert_count(
        "sensitivity genes", len({gene_ids[i] for i in sensitivity_indexes}), "sensitivity_genes"
    )

    unique_rnas: dict[str, dict[str, Any]] = {}
    for index in sensitivity_indexes:
        rna_id = rna_ids[index]
        row = unique_rnas.setdefault(
            rna_id,
            {
                "RNA_id": rna_id,
                "RNA_seqs": sequences[index],
                "rna_sequence_sha256": sequence_hashes[index],
                "rna_nt_length": int(lengths[index]),
                "primary_eligible": False,
            },
        )
        row["primary_eligible"] = bool(row["primary_eligible"] or primary[index])
    ordered_rnas = [unique_rnas[key] for key in sorted(unique_rnas)]
    assert_count("sensitivity unique sequences", len(ordered_rnas), "sensitivity_unique_sequences")
    assert_count(
        "primary unique sequences",
        sum(bool(row["primary_eligible"]) for row in ordered_rnas),
        "primary_unique_sequences",
    )

    exclusions = np.full(len(transcript_ids), "", dtype=object)
    exclusions[lengths > MAX_RNA_NT] = "length_gt_6000"
    exclusions[within_limit & ~alphabet_valid] = "invalid_alphabet"
    exclusions[sequence_eligible & ~primary_expression] = "below_primary_expression_prevalence"
    transcript_columns: dict[str, Any] = {
        "transcript_id": transcript_ids,
        "transcript_id_unversioned": [value.split(".", 1)[0] for value in transcript_ids],
        "RNA_id": rna_ids,
        "gene_id": gene_ids,
        "gene_id_unversioned": [value.split(".", 1)[0] for value in gene_ids],
        "gene_name": [row["gene_name"] for row in annotation_rows],
        "gene_biotype": [row["gene_biotype"] for row in annotation_rows],
        "transcript_biotype": [row["transcript_biotype"] for row in annotation_rows],
        "sequence": sequences,
        "sequence_length_nt": lengths,
        "rna_sequence_sha256": sequence_hashes,
        **summaries,
        "sequence_eligible": sequence_eligible,
        "primary_expression_eligible": primary_expression,
        "primary_eligible": primary,
        "sensitivity_expression_eligible": sensitivity_expression,
        "sensitivity_eligible": sensitivity,
        "exclusion_reason": exclusions.tolist(),
    }
    transcript_table = pa.table(transcript_columns).replace_schema_metadata(
        {
            b"primary_rule": b"ACGT, 1..6000 nt, TPM>=1 in >=52 tumors or >=6 normals",
            b"sensitivity_rule": b"ACGT, 1..6000 nt, TPM>=0.5 in >=52 tumors or >=6 normals",
        }
    )
    rna_table = pa.Table.from_pylist(ordered_rnas)
    lengths_summary = {
        "min": int(lengths.min()),
        "p25": float(np.quantile(lengths, 0.25)),
        "median": float(np.quantile(lengths, 0.5)),
        "p75": float(np.quantile(lengths, 0.75)),
        "max": int(lengths.max()),
    }
    counts = {
        "length_distribution_nt": lengths_summary,
        "primary": {"genes": EXPECTED["primary_genes"], "transcripts": int(primary.sum()),
                    "unique_sequences": EXPECTED["primary_unique_sequences"]},
        "sensitivity": {"genes": EXPECTED["sensitivity_genes"],
                        "transcripts": int(sensitivity.sum()),
                        "unique_sequences": len(ordered_rnas)},
    }
    return transcript_table, rna_table, counts


def software_versions() -> dict[str, str]:
    """Record the interpreter and core packages used by this preparation."""
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn"):
        versions[package] = importlib.metadata.version(package)
    versions["conda_environment"] = Path(os.environ.get("CONDA_PREFIX", "unknown")).name
    return versions


def main() -> None:
    """Prepare and validate every LUAD table and its provenance manifest."""
    args = parse_args()
    if args.batch_rows < 1:
        raise ValueError("--batch-rows must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    normal_source, xena, uniprot = validate_sources()
    transcript_ids, tumors = read_expression_axes(TUMOR_EXPRESSION)
    normal_transcripts, normals = read_expression_axes(NORMAL_EXPRESSION)
    if normal_transcripts != transcript_ids or normals != normal_source["samples"]:
        raise ValueError("tumor and normal expression axes do not match frozen provenance")
    assert_count("tumor samples", len(tumors), "tumors")
    assert_count("normal samples", len(normals), "normals")
    assert_count("transcripts", len(transcript_ids), "transcripts")
    samples = build_sample_rows(tumors, normals)
    annotations, sequences = load_gencode()
    LOGGER.info("Loaded GENCODE annotation and transcript sequences")

    with tempfile.TemporaryDirectory(prefix="luad_prepare_", dir=SOURCE_DIR) as temp_dir:
        matrix = load_expression(transcript_ids, tumors, normals, Path(temp_dir) / "expression.f32")
        LOGGER.info("Loaded %d tumor and %d normal expression rows", len(tumors), len(normals))
        summaries = expression_summaries(matrix, len(tumors), args.batch_rows)
        transcript_table, rna_table, counts = build_sequence_tables(
            transcript_ids, annotations, sequences, summaries
        )
        write_expression_parquet(transcript_ids, [*tumors, *normals], matrix, args.batch_rows)
        del matrix
    write_parquet_atomic(transcript_table, TRANSCRIPT_MANIFEST)
    write_parquet_atomic(rna_table, RNA_MANIFEST)
    write_tsv(
        SAMPLE_MANIFEST,
        ["sample_id", "sample_type", "patient_id", "pair_status", "expression_column_index"],
        samples,
    )
    proteins = load_protein_rows(uniprot)
    write_tsv(PROTEIN_MANIFEST, list(proteins[0]), proteins)
    validate_outputs([*tumors, *normals])

    scripts = {
        name: sha256_file(Path(__file__).resolve().parent / name)
        for name in ("download_luad_sources.py", "prepare_luad_inputs.py", "_luad_data.py")
    }
    source_hashes = {
        "tumor_expression": sha256_file(TUMOR_EXPRESSION),
        "normal_expression": sha256_file(NORMAL_EXPRESSION),
        "normal_source_manifest": sha256_file(NORMAL_SOURCE_MANIFEST),
        "xena_metadata": sha256_file(XENA_METADATA),
        "gencode_gtf": sha256_file(GENCODE_GTF),
        "gencode_fasta": sha256_file(GENCODE_FASTA),
        "uniprot_release_manifest": sha256_file(UNIPROT_RELEASE),
    }
    outputs = [EXPRESSION_PARQUET, SAMPLE_MANIFEST, TRANSCRIPT_MANIFEST,
               RNA_MANIFEST, PROTEIN_MANIFEST]
    output_hashes = {path.name: sha256_file(path) for path in outputs}
    configuration = {
        "max_rna_nt": MAX_RNA_NT,
        "normal_prevalence_minimum": NORMAL_PREVALENCE_MINIMUM,
        "primary_tpm_cutoff": PRIMARY_TPM_CUTOFF,
        "sensitivity_tpm_cutoff": SENSITIVITY_TPM_CUTOFF,
        "strict_rna_alphabet": "ACGT",
        "tumor_prevalence_minimum": TUMOR_PREVALENCE_MINIMUM,
    }
    run_id = sha256_text(
        json.dumps(
            {"configuration": configuration, "scripts": scripts, "sources": source_hashes},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    write_json_atomic(
        RUN_MANIFEST,
        {
            "schema_version": 1,
            "run_id": run_id,
            "sources": {
                "gencode": {
                    "assembly": "GRCh38.p3",
                    "release": 23,
                    "gtf_url": f"{GENCODE_BASE_URL}/gencode.v23.annotation.gtf.gz",
                    "gtf_md5": md5_file(GENCODE_GTF),
                    "fasta_url": f"{GENCODE_BASE_URL}/gencode.v23.transcripts.fa.gz",
                    "fasta_md5": md5_file(GENCODE_FASTA),
                },
                "uniprot": uniprot,
                "xena": {
                    "dataset": XENA_DATASET,
                    "metadata_url": XENA_METADATA_URL,
                    "probe_map_url": XENA_PROBEMAP_URL,
                    "release_date": xena["version"],
                    "unit": xena["unit"],
                },
            },
            "hashes": {"inputs": source_hashes, "outputs": output_hashes},
            "dimensions": {
                "expression_rows": len(transcript_ids),
                "expression_sample_columns": len(tumors) + len(normals),
                "matched_pairs": EXPECTED["matched_pairs"],
                "normal_samples": len(normals),
                "proteins": len(proteins),
                "rna_inference_sequences": rna_table.num_rows,
                "transcript_manifest_rows": transcript_table.num_rows,
                "tumor_samples": len(tumors),
            },
            "expression_transform": {
                "logged_to_tpm": "max(2**logged_value - 0.001, 0)",
                "threshold_comparison": "logged_value >= log2(TPM_cutoff + 0.001)",
            },
            "thresholds": configuration,
            "counts": counts,
            "software_versions": software_versions(),
            "scripts": scripts,
        },
    )
    LOGGER.info("LUAD inputs complete and validated; run ID %s", run_id)
if __name__ == "__main__":
    main()
