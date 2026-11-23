"""Validate complete, immutable all-vs-all prediction runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


COLUMNS = ["RNA_id", "Protein_id", "logit_margin", "interaction_probability", "prediction"]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file with bounded memory."""
    if chunk_size < 1:
        raise ValueError("hash chunk size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(path: Path) -> str:
    """Hash checkpoint filenames and contents deterministically."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file() and "__pycache__" not in item.parts and not item.name.startswith("."))
    if not files:
        raise ValueError(f"checkpoint directory is empty: {path}")
    for file_path in files:
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def validate_scores(frame: pd.DataFrame) -> None:
    """Reject missing keys, duplicate pairs and invalid model outputs."""
    if list(frame.columns) != COLUMNS or frame.isna().any().any():
        raise ValueError("invalid or incomplete prediction schema")
    if frame[COLUMNS[:2]].duplicated().any():
        raise ValueError("duplicate prediction pairs")
    for name in COLUMNS[:2]:
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"invalid prediction identifier: {name}")
    if not np.isfinite(frame[COLUMNS[2:]].to_numpy()).all():
        raise ValueError("non-finite prediction values")
    if not frame.interaction_probability.between(0, 1).all():
        raise ValueError("probabilities must be between zero and one")
    if not frame.prediction.isin([0, 1]).all() or not (frame.prediction == (frame.logit_margin > 0)).all():
        raise ValueError("labels disagree with logit margins")


def read_predictions(
    directory: Path, rnas_path: Path, proteins_path: Path, *, precision: str = "float32"
) -> tuple[pd.DataFrame, list[Path]]:
    """Load only a complete run bound to the supplied sequence manifests."""
    manifests = sorted(directory.glob("rank-*-manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no completed prediction manifests: {directory}")
    payloads = [json.loads(path.read_text()) for path in manifests]
    world_size = len(payloads)
    if {m["rank"] for m in payloads} != set(range(world_size)):
        raise ValueError("prediction rank coverage is incomplete")
    shared = ["world_size", "checkpoint_sha256", "code_sha256", "encoder_sha256",
              "rna_encoder", "protein_encoder", "batch_size", "precision", "seed",
              "max_rna_nt", "max_protein_aa", "rna_manifest_sha256", "protein_manifest_sha256"]
    if any(any(m.get(k) != payloads[0].get(k) for k in shared) for m in payloads):
        raise ValueError("prediction ranks describe incompatible runs")
    first = payloads[0]
    if first["world_size"] != world_size or first["precision"] != precision:
        raise ValueError("prediction world size or precision does not match")
    if first["rna_manifest_sha256"] != sha256_file(rnas_path) or first["protein_manifest_sha256"] != sha256_file(proteins_path):
        raise ValueError("prediction inputs do not match supplied manifests")
    rnas = pd.read_parquet(rnas_path)["RNA_id"]
    proteins = pd.read_csv(proteins_path, sep="\t")["Protein_id"]
    if rnas.duplicated().any() or proteins.duplicated().any():
        raise ValueError("duplicate input identifiers")
    parts: list[Path] = []
    frames: list[pd.DataFrame] = []
    for manifest in payloads:
        if manifest.get("status") != "complete":
            raise ValueError("prediction rank is not complete")
        rank_frames = []
        for item in manifest["parts"]:
            path = directory / item["path"]
            if path.parent != directory or path in parts or sha256_file(path) != item["sha256"]:
                raise ValueError(f"invalid prediction part: {path}")
            metadata = pq.read_metadata(path)
            attrs = metadata.metadata or {}
            if metadata.num_rows != item["rows"] or attrs.get(b"run_hash", b"").decode() != manifest["run_hash"] or attrs.get(b"rank", b"").decode() != str(manifest["rank"]):
                raise ValueError(f"prediction metadata mismatch: {path}")
            frame = pd.read_parquet(path)
            validate_scores(frame)
            parts.append(path)
            rank_frames.append(frame)
        rank_frame = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame(columns=COLUMNS)
        expected_rnas = set(rnas.iloc[manifest["rank"]::world_size])
        if len(rank_frame) != manifest["expected_rows"] or set(rank_frame.RNA_id) != expected_rnas:
            raise ValueError("rank RNA coverage does not match its deterministic shard")
        frames.append(rank_frame)
    if set(directory.glob("rank-*-part-*.parquet")) != set(parts):
        raise ValueError("unlisted or missing prediction parts")
    result = pd.concat(frames, ignore_index=True)
    validate_scores(result)
    if len(result) != len(rnas) * len(proteins) or set(result.Protein_id) != set(proteins):
        raise ValueError("incomplete prediction Cartesian product")
    if not result.groupby("RNA_id").size().eq(len(proteins)).all():
        raise ValueError("incomplete per-RNA protein coverage")
    if precision == "float32":
        probability = 1 / (1 + np.exp(-result.logit_margin.to_numpy(float)))
        if not np.allclose(probability, result.interaction_probability, atol=2e-6, rtol=2e-5):
            raise ValueError("probabilities disagree with float32 margins")
    return result, manifests
