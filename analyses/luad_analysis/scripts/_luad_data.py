"""Private data and I/O primitives for LUAD input preparation.

The public commands are ``download_luad_sources.py`` and
``prepare_luad_inputs.py``. This module keeps their shared immutable paths,
resolved counts, parsers, hashing functions, and bounded expression-matrix
operations in one place. It has no command-line interface and does not perform
network access by itself.

The large expression matrix is represented temporarily as an on-disk float32
array. This keeps memory use bounded while converting the supplied sample-row
TSV into the transcript-row Parquet required by downstream analysis.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(os.environ.get("CORAL_REPO_ROOT", Path(__file__).resolve().parents[3]))
TCGA_DIR = REPO_ROOT / "data" / "TCGA"
SOURCE_DIR = TCGA_DIR / "luad_sources"

TUMOR_EXPRESSION = TCGA_DIR / "LUAD_isoforms.tsv"
NORMAL_EXPRESSION = SOURCE_DIR / "LUAD_normal_isoforms.tsv.gz"
NORMAL_SOURCE_MANIFEST = SOURCE_DIR / "LUAD_normal_isoforms.source.json"
XENA_METADATA = SOURCE_DIR / "TcgaTargetGtex_rsem_isoform_tpm.json"
GENCODE_GTF = SOURCE_DIR / "gencode.v23.annotation.gtf.gz"
GENCODE_FASTA = SOURCE_DIR / "gencode.v23.transcripts.fa.gz"
UNIPROT_RELEASE = SOURCE_DIR / "uniprot_release.json"

EXPRESSION_PARQUET = TCGA_DIR / "luad_expression.parquet"
SAMPLE_MANIFEST = TCGA_DIR / "luad_samples.tsv"
TRANSCRIPT_MANIFEST = TCGA_DIR / "luad_transcripts.parquet"
RNA_MANIFEST = TCGA_DIR / "luad_rna_sequences.parquet"
PROTEIN_MANIFEST = TCGA_DIR / "luad_proteins.tsv"
RUN_MANIFEST = TCGA_DIR / "luad_inputs.json"

XENA_HUB = "https://toil.xenahubs.net"
XENA_DATASET = "TcgaTargetGtex_rsem_isoform_tpm"
XENA_DATA_URL = f"{XENA_HUB}/data/"
XENA_METADATA_URL = f"{XENA_HUB}/download/{XENA_DATASET}.json"
XENA_PROBEMAP_URL = (
    f"{XENA_HUB}/download/probeMap/gencode.v23.annotation.transcript.probemap"
)
GENCODE_BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_23"
)
GENCODE_SOURCES = {
    GENCODE_GTF: (
        f"{GENCODE_BASE_URL}/gencode.v23.annotation.gtf.gz",
        "4c06cb6f2eb5d4e1e161f895be32c33b",
    ),
    GENCODE_FASTA: (
        f"{GENCODE_BASE_URL}/gencode.v23.transcripts.fa.gz",
        "98ba7a199a2bc9b0a707d9567edb4aec",
    ),
}
PROTEINS = {
    "IGF2BP1": ("Q9NZI8", "Q9NZI8-1", 577),
    "IGF2BP3": ("O00425", "O00425-1", 579),
    "LIN28B": ("Q6ZN17", "Q6ZN17-1", 250),
}
PINNED_PROTEIN_SHA256 = {
    "Q9NZI8": "e9dfe3add14d17a938d9a42fa9356ddc680181c6d3029b216f9a4577c92b8cb4",
    "O00425": "f4773c1f8040890d506a167b70759e2a084bbfa1c1606db2ba105c4878e4c798",
    "Q6ZN17": "5ec90b937f38e6859645039f0d3418f863a952bd7e8a2c160ba5a63bc24b36ad",
}
UNIPROT_URL_TEMPLATE = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
UNPAIRED_NORMAL = "TCGA-44-6144-11"

PRIMARY_TPM_CUTOFF = 1.0
SENSITIVITY_TPM_CUTOFF = 0.5

EXPECTED = {
    "tumor_expression_sha256": (
        "b1b53f9857bd26e2a7ff7cb5be64625ff388909b1a5d74cb86699d4b80c561e2"
    ),
    "tumors": 513,
    "normals": 59,
    "matched_pairs": 58,
    "transcripts": 198_619,
    "at_most_6000_nt": 193_459,
    "invalid_alphabet_within_limit": 3,
    "over_6000_nt": 5_160,
}
PINNED_SOURCE_SHA256 = {
    NORMAL_EXPRESSION: "6ca2248abdc92c9ba6702e2a0ba63b02534634fcf44dd001bd37c4e28fc71b78",
    XENA_METADATA: "4166b92917233c0b954b563a275d12dc956fe590f7d6735ef85d2ec671b96c60",
    GENCODE_GTF: "adfda26c291cfb68ae13b54b285f1c0023abc2dfc703b8871f43b9655ab64a56",
    GENCODE_FASTA: "f19557d65d3aaba05a25ae4e78f260bcbff76e37843e767ac0222174da9916d2",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file's MD5 digest for comparison with GENCODE checksums."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON through a temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def open_text(path: Path) -> TextIO:
    """Open plain or gzip-compressed UTF-8 text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_expression_axes(path: Path) -> tuple[list[str], list[str]]:
    """Read unique transcript and sample identifiers without parsing values."""
    with open_text(path) as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        if not header or header[0] != "sample":
            raise ValueError(f"{path}: first column must be 'sample'")
        transcript_ids = header[1:]
        sample_ids = []
        for line_number, line in enumerate(handle, start=2):
            sample_id, separator, _ = line.partition("\t")
            if not separator:
                raise ValueError(f"{path}:{line_number}: malformed expression row")
            sample_ids.append(sample_id)
    if len(transcript_ids) != len(set(transcript_ids)):
        raise ValueError(f"{path}: duplicate transcript identifiers")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path}: duplicate sample identifiers")
    return transcript_ids, sample_ids


def parse_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield FASTA identifier, full header, and uppercase sequence."""
    header: str | None = None
    parts: list[str] = []
    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header.split()[0].split("|")[0], header, "".join(parts)
                header, parts = line[1:], []
            elif header is None:
                raise ValueError(f"{path}: sequence before first FASTA header")
            else:
                parts.append(line.upper())
    if header is not None:
        yield header.split()[0].split("|")[0], header, "".join(parts)


def populate_expression(
    path: Path,
    transcript_ids: list[str],
    sample_ids: list[str],
    matrix: np.memmap,
    offset: int,
) -> None:
    """Parse one sample-row expression source into assigned memmap rows."""
    with open_text(path) as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        if header[0] != "sample" or header[1:] != transcript_ids:
            raise ValueError(f"{path}: transcript axis mismatch")
        observed = []
        for index, line in enumerate(handle):
            if index >= len(sample_ids):
                raise ValueError(f"{path}: too many sample rows")
            sample_id, separator, text = line.partition("\t")
            values = np.fromstring(text, dtype=np.float32, sep="\t") if separator else np.array([])
            if values.size != len(transcript_ids) or not np.isfinite(values).all():
                raise ValueError(f"{path}: invalid values for {sample_id}")
            matrix[offset + index] = values
            observed.append(sample_id)
    if observed != sample_ids:
        raise ValueError(f"{path}: sample axis mismatch")


def load_expression(
    transcript_ids: list[str], tumor_samples: list[str], normal_samples: list[str], path: Path
) -> np.memmap:
    """Load tumor and normal logged expression into a disk-backed float32 array."""
    matrix = np.memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(tumor_samples) + len(normal_samples), len(transcript_ids)),
    )
    populate_expression(TUMOR_EXPRESSION, transcript_ids, tumor_samples, matrix, 0)
    populate_expression(
        NORMAL_EXPRESSION, transcript_ids, normal_samples, matrix, len(tumor_samples)
    )
    matrix.flush()
    return matrix


def expression_summaries(
    matrix: np.memmap, tumor_count: int, batch_rows: int
) -> dict[str, np.ndarray]:
    """Compute prevalence counts and descriptive TPM summaries in bounded blocks."""
    count = matrix.shape[1]
    specs = {
        "tumor_tpm_ge_1_count": np.int16,
        "normal_tpm_ge_1_count": np.int8,
        "tumor_tpm_ge_0_5_count": np.int16,
        "normal_tpm_ge_0_5_count": np.int8,
        "tumor_mean_tpm": np.float32,
        "tumor_median_tpm": np.float32,
        "normal_mean_tpm": np.float32,
        "normal_median_tpm": np.float32,
    }
    result = {name: np.empty(count, dtype=dtype) for name, dtype in specs.items()}
    primary_logged = np.log2(PRIMARY_TPM_CUTOFF + 0.001)
    sensitivity_logged = np.log2(SENSITIVITY_TPM_CUTOFF + 0.001)
    for start in range(0, count, batch_rows):
        stop = min(start + batch_rows, count)
        tumor = np.asarray(matrix[:tumor_count, start:stop])
        normal = np.asarray(matrix[tumor_count:, start:stop])
        result["tumor_tpm_ge_1_count"][start:stop] = np.count_nonzero(tumor >= primary_logged, 0)
        result["normal_tpm_ge_1_count"][start:stop] = np.count_nonzero(normal >= primary_logged, 0)
        result["tumor_tpm_ge_0_5_count"][start:stop] = np.count_nonzero(
            tumor >= sensitivity_logged, 0
        )
        result["normal_tpm_ge_0_5_count"][start:stop] = np.count_nonzero(
            normal >= sensitivity_logged, 0
        )
        tumor_tpm = np.maximum(np.exp2(tumor.astype(np.float64)) - 0.001, 0.0)
        normal_tpm = np.maximum(np.exp2(normal.astype(np.float64)) - 0.001, 0.0)
        for cohort, values in (("tumor", tumor_tpm), ("normal", normal_tpm)):
            result[f"{cohort}_mean_tpm"][start:stop] = np.mean(values, axis=0)
            result[f"{cohort}_median_tpm"][start:stop] = np.median(values, axis=0)
    return result


def write_expression_parquet(
    transcript_ids: list[str], sample_ids: list[str], matrix: np.memmap, batch_rows: int,
    destination: Path = EXPRESSION_PARQUET,
) -> None:
    """Atomically write the transcript-row logged-expression Parquet."""
    schema = pa.schema(
        [pa.field("transcript_id", pa.string())]
        + [pa.field(sample_id, pa.float32()) for sample_id in sample_ids],
        metadata={
            b"orientation": b"transcript_rows_sample_columns",
            b"source_unit": b"log2(TPM + 0.001)",
            b"tpm_transform": b"max(2**logged_value - 0.001, 0)",
        },
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    with pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=["transcript_id"]) as writer:
        for start in range(0, len(transcript_ids), batch_rows):
            stop = min(start + batch_rows, len(transcript_ids))
            block = np.asarray(matrix[:, start:stop]).T
            arrays = [pa.array(transcript_ids[start:stop], type=pa.string())]
            arrays.extend(pa.array(block[:, index]) for index in range(len(sample_ids)))
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
    os.replace(temporary, destination)


def write_parquet_atomic(table: pa.Table, path: Path, row_group_size: int = 4_096) -> None:
    """Write a compressed Arrow table through an atomic rename."""
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=row_group_size)
    os.replace(temporary, path)


def assert_count(label: str, observed: int, expected_key: str) -> None:
    """Fail when an observed count differs from the resolved analysis contract."""
    expected = int(EXPECTED[expected_key])
    if observed != expected:
        raise ValueError(f"{label}: expected {expected}, observed {observed}")


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a deterministic TSV through a temporary file and atomic rename."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a small TSV manifest as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def participant_id(sample_id: str) -> str:
    """Return the participant portion of a four-field TCGA sample barcode."""
    fields = sample_id.split("-")
    if len(fields) != 4:
        raise ValueError(f"unexpected TCGA sample identifier: {sample_id}")
    return "-".join(fields[:3])


def build_sample_rows(tumors: list[str], normals: list[str]) -> list[dict[str, Any]]:
    """Build sample records and validate the exact paired-patient set."""
    if any(sample.split("-")[3][:2] != "01" for sample in tumors):
        raise ValueError("tumor matrix contains a non-primary sample")
    if any(sample.split("-")[3][:2] != "11" for sample in normals):
        raise ValueError("normal matrix contains a non-solid-tissue-normal sample")
    for cohort in (tumors, normals):
        patients = [participant_id(sample) for sample in cohort]
        if len(patients) != len(set(patients)):
            raise ValueError("more than one sample per patient and cohort")
    matched = {participant_id(sample) for sample in tumors} & {
        participant_id(sample) for sample in normals
    }
    assert_count("matched tumor/normal pairs", len(matched), "matched_pairs")
    rows = []
    for index, sample in enumerate([*tumors, *normals]):
        participant = participant_id(sample)
        rows.append(
            {
                "sample_id": sample,
                "sample_type": "primary_tumor" if index < len(tumors) else "solid_tissue_normal",
                "patient_id": participant,
                "pair_status": "matched" if participant in matched else "unpaired",
                "expression_column_index": index,
            }
        )
    return rows
