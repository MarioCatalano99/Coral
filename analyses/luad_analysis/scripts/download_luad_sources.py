#!/usr/bin/env python3
"""Download and freeze the external sources used to prepare LUAD inputs.

This is the network-facing stage of the LUAD analysis. It validates the
supplied ``data/TCGA/LUAD_isoforms.tsv`` matrix, downloads GENCODE v23
annotation and transcript sequences, retrieves the three reviewed canonical
UniProt proteins, and queries the UCSC Xena dataset for the 59 normal samples
specified by the analysis design.

Inputs:
    data/TCGA/LUAD_isoforms.tsv
        Supplied 513-primary-tumor expression matrix. Its SHA-256, sample
        count, and transcript count must match the frozen analysis contract.
    UCSC Xena ``TcgaTargetGtex_rsem_isoform_tpm``
        Source of 58 patient-matched solid-tissue normals plus the fixed
        unpaired normal TCGA-44-6144-11.
    GENCODE human release 23 and UniProt REST endpoints.

Outputs under ``data/TCGA/luad_sources/``:
    LUAD_normal_isoforms.tsv.gz and its ``.source.json`` provenance sidecar;
    GENCODE v23 GTF and transcript FASTA; Xena dataset metadata; three UniProt
    FASTA files; and ``uniprot_release.json``.

Downloads are cached, checksummed, and atomically renamed. Normal expression
is requested in bounded transcript blocks and rejected if any requested sample,
transcript, or value is missing. Reruns reuse validated cached sources unless
``--refresh-normals`` is passed.

Run from the repository root in the ``newrpidnabert`` environment:

    python analyses/luad_analysis/scripts/download_luad_sources.py
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from _luad_data import (
    EXPECTED,
    GENCODE_FASTA,
    GENCODE_GTF,
    GENCODE_SOURCES,
    NORMAL_EXPRESSION,
    NORMAL_SOURCE_MANIFEST,
    PINNED_SOURCE_SHA256,
    PROTEINS,
    SOURCE_DIR,
    TUMOR_EXPRESSION,
    UNIPROT_RELEASE,
    UNIPROT_URL_TEMPLATE,
    UNPAIRED_NORMAL,
    XENA_DATASET,
    XENA_DATA_URL,
    XENA_METADATA,
    XENA_METADATA_URL,
    md5_file,
    parse_fasta,
    read_expression_axes,
    sha256_file,
    sha256_text,
    write_json_atomic,
)


LOGGER = logging.getLogger("luad.sources")
MAX_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    """Parse source-download options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xena-chunk-size",
        type=int,
        default=5_000,
        help="Transcripts requested per Xena API call (default: 5000).",
    )
    parser.add_argument(
        "--refresh-normals",
        action="store_true",
        help="Refetch normals even when the cached matrix passes validation.",
    )
    return parser.parse_args()


def fetch_bytes(url: str, data: bytes | None = None) -> tuple[bytes, dict[str, str]]:
    """Fetch bytes with bounded retries and normalized response headers."""
    headers = {"User-Agent": "CORAL-LUAD-input-preparation/1.0"}
    if data is not None:
        headers["Content-Type"] = "text/plain"
    request = Request(url, data=data, headers=headers)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=180) as response:
                normalized = {key.lower(): value for key, value in response.headers.items()}
                return response.read(), normalized
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"failed to fetch {url} after {attempt} attempts") from error
            delay = 2 ** (attempt - 1)
            LOGGER.warning("Fetch failed for %s; retrying in %d seconds", url, delay)
            time.sleep(delay)
    raise AssertionError("retry loop ended unexpectedly")


def download_file(
    url: str,
    destination: Path,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> None:
    """Download one immutable source atomically, or validate its cached copy."""
    if destination.exists():
        if expected_md5 and md5_file(destination) != expected_md5:
            raise ValueError(f"cached source has wrong MD5: {destination}")
        if expected_sha256 and sha256_file(destination) != expected_sha256:
            raise ValueError(f"cached source has wrong SHA-256: {destination}")
        LOGGER.info("Using cached source: %s", destination)
        return
    payload, _ = fetch_bytes(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    if expected_md5 and md5_file(temporary) != expected_md5:
        temporary.unlink()
        raise ValueError(f"downloaded source has wrong MD5: {url}")
    if expected_sha256 and sha256_file(temporary) != expected_sha256:
        temporary.unlink()
        raise ValueError(f"downloaded source has wrong SHA-256: {url}")
    os.replace(temporary, destination)


def xena_fetch(samples: list[str], probes: list[str]) -> list[list[float | str]]:
    """Fetch a probe-by-sample block from the Xena dense-matrix API."""
    query = (
        f'(fetch [{{:table "{XENA_DATASET}" '
        f':columns {json.dumps(probes, separators=(",", ":"))} '
        f':samples {json.dumps(samples, separators=(",", ":"))}}}])'
    )
    payload, _ = fetch_bytes(XENA_DATA_URL, query.encode("utf-8"))
    result = json.loads(payload)
    if not isinstance(result, list) or len(result) != len(probes):
        raise ValueError("Xena response has the wrong probe dimension")
    return result


def patient_id(sample_id: str) -> str:
    """Extract the TCGA participant barcode from a sample barcode."""
    fields = sample_id.split("-")
    if len(fields) != 4 or fields[0] != "TCGA":
        raise ValueError(f"unexpected TCGA sample identifier: {sample_id}")
    return "-".join(fields[:3])


def discover_normals(tumor_samples: list[str], sentinel_probe: str) -> list[str]:
    """Find the 58 paired normals and add the specified unpaired normal."""
    if any(sample.split("-")[3][:2] != "01" for sample in tumor_samples):
        raise ValueError("tumor input contains a recurrent or non-primary sample")
    candidates = sorted({f"{patient_id(sample)}-11" for sample in tumor_samples})
    rows = xena_fetch(candidates, [sentinel_probe])
    if len(rows) != 1 or len(rows[0]) != len(candidates):
        raise ValueError("Xena normal-discovery response has the wrong dimensions")
    paired = [
        sample
        for sample, value in zip(candidates, rows[0], strict=True)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if len(paired) != EXPECTED["matched_pairs"] or UNPAIRED_NORMAL in paired:
        raise ValueError(f"expected 58 paired normals, found {len(paired)}")
    normals = sorted([*paired, UNPAIRED_NORMAL])
    if len(normals) != EXPECTED["normals"] or len(normals) != len(set(normals)):
        raise ValueError("normal cohort is not exactly 59 unique samples")
    return normals


def normal_cache_is_valid(transcript_ids: list[str], samples: list[str]) -> bool:
    """Return whether cached normal expression matches its provenance sidecar."""
    if not NORMAL_EXPRESSION.exists() or not NORMAL_SOURCE_MANIFEST.exists():
        return False
    try:
        metadata = json.loads(NORMAL_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(
        metadata.get("samples") == samples
        and metadata.get("transcript_count") == len(transcript_ids)
        and metadata.get("transcript_ids_sha256")
        == sha256_text("\n".join(transcript_ids) + "\n")
        and metadata.get("sha256") == sha256_file(NORMAL_EXPRESSION)
        == PINNED_SOURCE_SHA256[NORMAL_EXPRESSION]
    )


def write_normal_expression(
    transcript_ids: list[str], samples: list[str], values: np.memmap
) -> str:
    """Serialize normal expression as deterministic sample-row gzip text."""
    temporary = NORMAL_EXPRESSION.with_name(f".{NORMAL_EXPRESSION.name}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                handle.write("sample\t" + "\t".join(transcript_ids) + "\n")
                for index, sample in enumerate(samples):
                    serialized = (format(float(value), ".9g") for value in values[:, index])
                    handle.write(sample + "\t" + "\t".join(serialized) + "\n")
    os.replace(temporary, NORMAL_EXPRESSION)
    return sha256_file(NORMAL_EXPRESSION)


def fetch_normals(
    transcript_ids: list[str], tumor_samples: list[str], chunk_size: int, refresh: bool
) -> None:
    """Fetch and freeze all normal expression values in bounded blocks."""
    if chunk_size < 1:
        raise ValueError("--xena-chunk-size must be positive")
    samples = discover_normals(tumor_samples, transcript_ids[0])
    if not refresh and normal_cache_is_valid(transcript_ids, samples):
        LOGGER.info("Using cached normal expression: %s", NORMAL_EXPRESSION)
        return
    with tempfile.TemporaryDirectory(prefix="luad_normals_", dir=SOURCE_DIR) as temp_dir:
        values = np.memmap(
            Path(temp_dir) / "values.float32",
            mode="w+",
            dtype=np.float32,
            shape=(len(transcript_ids), len(samples)),
        )
        chunks = math.ceil(len(transcript_ids) / chunk_size)
        for chunk, start in enumerate(range(0, len(transcript_ids), chunk_size), start=1):
            stop = min(start + chunk_size, len(transcript_ids))
            block = np.asarray(xena_fetch(samples, transcript_ids[start:stop]), dtype=np.float32)
            if block.shape != (stop - start, len(samples)) or not np.isfinite(block).all():
                raise ValueError(f"Xena chunk {chunk} is incomplete or non-finite")
            values[start:stop] = block
            LOGGER.info("Fetched normal-expression chunk %d/%d", chunk, chunks)
        values.flush()
        digest = write_normal_expression(transcript_ids, samples, values)
    if digest != PINNED_SOURCE_SHA256[NORMAL_EXPRESSION]:
        raise ValueError(f"normal expression SHA-256 changed: {digest}")
    write_json_atomic(
        NORMAL_SOURCE_MANIFEST,
        {
            "dataset": XENA_DATASET,
            "endpoint": XENA_DATA_URL,
            "normal_count": len(samples),
            "samples": samples,
            "sha256": digest,
            "transcript_count": len(transcript_ids),
            "transcript_ids_sha256": sha256_text("\n".join(transcript_ids) + "\n"),
            "value_format": "float32 serialized with Python .9g",
        },
    )


def fetch_uniprot() -> None:
    """Fetch or validate the three reviewed canonical protein FASTA files."""
    if UNIPROT_RELEASE.exists():
        cached = json.loads(UNIPROT_RELEASE.read_text(encoding="utf-8"))
        entries = cached.get("entries", [])
        if len(entries) == len(PROTEINS) and all(
            (SOURCE_DIR / f"uniprot_{entry['accession']}.fasta").exists()
            and sha256_file(SOURCE_DIR / f"uniprot_{entry['accession']}.fasta")
            == entry.get("sha256")
            for entry in entries
        ):
            LOGGER.info("Using cached UniProt release %s", cached.get("release"))
            return
    releases: set[tuple[str, str]] = set()
    entries: list[dict[str, Any]] = []
    for symbol, (accession, isoform, expected_length) in PROTEINS.items():
        url = UNIPROT_URL_TEMPLATE.format(accession=accession)
        payload, headers = fetch_bytes(url)
        lines = payload.decode("utf-8").splitlines()
        sequence = "".join(lines[1:]).upper()
        if not lines or not lines[0].startswith(f">sp|{accession}|"):
            raise ValueError(f"UniProt entry is not reviewed: {accession}")
        if len(sequence) != expected_length:
            raise ValueError(f"wrong canonical length for {accession}")
        releases.add((headers["x-uniprot-release"], headers["x-uniprot-release-date"]))
        path = SOURCE_DIR / f"uniprot_{accession}.fasta"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        entries.append(
            {
                "accession": accession,
                "gene_symbol": symbol,
                "isoform": isoform,
                "length_aa": expected_length,
                "sha256": sha256_file(path),
                "source_url": url,
            }
        )
    if len(releases) != 1:
        raise ValueError(f"protein responses span multiple UniProt releases: {releases}")
    release, release_date = releases.pop()
    write_json_atomic(
        UNIPROT_RELEASE,
        {"entries": entries, "release": release, "release_date": release_date},
    )


def validate_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate frozen sources and return their parsed provenance metadata."""
    required = [TUMOR_EXPRESSION, NORMAL_EXPRESSION, NORMAL_SOURCE_MANIFEST,
                XENA_METADATA, GENCODE_GTF, GENCODE_FASTA, UNIPROT_RELEASE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing LUAD sources: {missing}")
    if sha256_file(TUMOR_EXPRESSION) != EXPECTED["tumor_expression_sha256"]:
        raise ValueError("supplied tumor-expression SHA-256 mismatch")
    for path, expected_digest in PINNED_SOURCE_SHA256.items():
        if sha256_file(path) != expected_digest:
            raise ValueError(f"frozen source SHA-256 mismatch: {path}")
    normal = json.loads(NORMAL_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    xena = json.loads(XENA_METADATA.read_text(encoding="utf-8"))
    uniprot = json.loads(UNIPROT_RELEASE.read_text(encoding="utf-8"))
    if normal.get("sha256") != PINNED_SOURCE_SHA256[NORMAL_EXPRESSION]:
        raise ValueError("normal-expression sidecar mismatch")
    expected_xena = {"version": "2016-09-02", "unit": "log2(tpm+0.001)",
                     ":probeMap": "/probeMap/gencode.v23.annotation.transcript.probemap"}
    if any(xena.get(key) != value for key, value in expected_xena.items()):
        raise ValueError("Xena metadata differs from the resolved source")
    entries = uniprot.get("entries", [])
    if len(entries) != len(PROTEINS):
        raise ValueError("UniProt manifest must contain exactly three entries")
    for entry in entries:
        path = SOURCE_DIR / f"uniprot_{entry['accession']}.fasta"
        if not path.exists() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"UniProt source mismatch: {path}")
    return normal, xena, uniprot


def load_protein_rows(uniprot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fixed-order proteins in the canonical Step-4 input schema."""
    rows = []
    for symbol, (accession, isoform, expected_length) in PROTEINS.items():
        records = list(parse_fasta(SOURCE_DIR / f"uniprot_{accession}.fasta"))
        if len(records) != 1 or records[0][0] != "sp" or f"|{accession}|" not in records[0][1]:
            raise ValueError(f"invalid reviewed UniProt FASTA for {accession}")
        sequence = records[0][2]
        if len(sequence) != expected_length or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"invalid canonical protein sequence for {accession}")
        rows.append({"Protein_id": isoform, "Prot_seqs": sequence, "gene_symbol": symbol,
                     "uniprot_accession": accession, "uniprot_isoform": isoform,
                     "protein_aa_length": len(sequence),
                     "protein_sequence_sha256": sha256_text(sequence), "reviewed": "true",
                     "uniprot_release": uniprot["release"]})
    return rows


def main() -> None:
    """Download or validate all source data required for LUAD preparation."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if not TUMOR_EXPRESSION.exists():
        raise FileNotFoundError(f"missing supplied matrix: {TUMOR_EXPRESSION}")
    if sha256_file(TUMOR_EXPRESSION) != EXPECTED["tumor_expression_sha256"]:
        raise ValueError("supplied LUAD tumor matrix SHA-256 mismatch")
    transcript_ids, tumor_samples = read_expression_axes(TUMOR_EXPRESSION)
    if len(transcript_ids) != EXPECTED["transcripts"]:
        raise ValueError(f"expected 198619 transcripts, found {len(transcript_ids)}")
    if len(tumor_samples) != EXPECTED["tumors"]:
        raise ValueError(f"expected 513 tumors, found {len(tumor_samples)}")

    download_file(
        XENA_METADATA_URL,
        XENA_METADATA,
        expected_sha256=PINNED_SOURCE_SHA256[XENA_METADATA],
    )
    for destination, (url, expected_md5) in GENCODE_SOURCES.items():
        download_file(url, destination, expected_md5=expected_md5)
    fetch_uniprot()
    fetch_normals(transcript_ids, tumor_samples, args.xena_chunk_size, args.refresh_normals)
    LOGGER.info("LUAD sources are complete under %s", SOURCE_DIR)


if __name__ == "__main__":
    main()
