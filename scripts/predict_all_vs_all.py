"""Stream all-vs-all CORAL predictions from RNA and protein manifests."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coral import paths
from coral.data import (
    PROTEIN_ALPHABET,
    RNA_ALPHABET,
    validate_sequence_frame,
)
from coral.inference import iter_all_vs_all_predictions
from coral.model import CoralConfig, CoralPretraining
from coral.prediction_io import sha256_file, validate_scores


LOGGER = logging.getLogger("coral.all_vs_all")


def parse_batch_size(value: str) -> int | str:
    """Parse an RNA batch size or the literal 'auto'."""
    if value == "auto":
        return value
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "batch size must be a positive integer or 'auto'"
        ) from error
    if result < 1:
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'")
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnas", type=Path, required=True)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=paths.checkpoint(9))
    parser.add_argument("--rna-encoder", default=str(paths.DNABERT_DIR))
    parser.add_argument("--protein-encoder", default=paths.PROTEIN_ENCODER)
    parser.add_argument("--batch-size", type=parse_batch_size, default="auto")
    parser.add_argument("--max-rna-nt", type=int, default=6_000)
    parser.add_argument("--max-protein-aa", type=int, default=1_250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    """Read a Parquet, TSV, or CSV manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported manifest format: {path}")


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


def execution_context() -> tuple[int, int, int, torch.device]:
    """Resolve torchrun rank metadata and the local inference device."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world_size:
        raise ValueError("invalid RANK/WORLD_SIZE environment")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, torch.device("cuda", local_rank)
    if world_size != 1:
        raise RuntimeError("multi-process inference requires CUDA")
    return rank, world_size, local_rank, torch.device("cpu")


def build_run_config(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Build the immutable configuration used to validate resumed parts."""
    code_paths = [
        ROOT / "src/coral/inference.py",
        ROOT / "src/coral/data.py",
        ROOT / "src/coral/model.py",
        ROOT / "src/coral/prediction_io.py",
        Path(__file__).resolve(),
    ]
    config: dict[str, Any] = {
        "rna_manifest": str(args.rnas.resolve()),
        "rna_manifest_sha256": sha256_file(args.rnas),
        "protein_manifest": str(args.proteins.resolve()),
        "protein_manifest_sha256": sha256_file(args.proteins),
        "checkpoint": str(args.weights.resolve()),
        "checkpoint_sha256": checkpoint_hash(args.weights),
        "rna_encoder": args.rna_encoder,
        "protein_encoder": args.protein_encoder,
        "batch_size": args.batch_size,
        "precision": "float32",
        "encoder_sha256": {"rna": checkpoint_hash(Path(args.rna_encoder)), "protein": checkpoint_hash(Path(args.protein_encoder))},
        "max_rna_nt": args.max_rna_nt,
        "max_protein_aa": args.max_protein_aa,
        "seed": args.seed,
        "rank": rank,
        "world_size": world_size,
        "code_sha256": {path.name: sha256_file(path) for path in code_paths},
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["run_hash"] = hashlib.sha256(encoded).hexdigest()
    return config


def part_paths(output_dir: Path, rank: int) -> list[Path]:
    """Return this rank's prediction parts in numeric order."""
    return sorted(output_dir.glob(f"rank-{rank:05d}-part-*.parquet"))


def next_part_index(output_dir: Path, rank: int) -> int:
    """Return the next unused numeric part index for a rank."""
    paths_for_rank = part_paths(output_dir, rank)
    if not paths_for_rank:
        return 0
    indexes = [int(path.stem.rsplit("-", 1)[1]) for path in paths_for_rank]
    return max(indexes) + 1


def read_completed_rnas(
    output_dir: Path,
    rank: int,
    run_hash: str,
    protein_ids: set[str],
) -> set[str]:
    """Validate existing parts and return fully completed RNA IDs."""
    completed: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for path in part_paths(output_dir, rank):
        receipt = path.with_suffix(".sha256")
        if not receipt.exists() or receipt.read_text().strip() != sha256_file(path):
            raise ValueError(f"prediction part hash mismatch or missing receipt: {path}")
        metadata = pq.read_metadata(path).metadata or {}
        if metadata.get(b"run_hash", b"").decode() != run_hash:
            raise ValueError(f"run hash mismatch in existing part: {path}")
        frame = pd.read_parquet(path)
        validate_scores(frame)
        for rna_id, group in frame.groupby("RNA_id", sort=False):
            observed = set(group["Protein_id"])
            if observed != protein_ids or len(group) != len(protein_ids):
                raise ValueError(f"incomplete protein coverage for {rna_id} in {path}")
            if rna_id in completed:
                raise ValueError(f"RNA appears in more than one prediction part: {rna_id}")
            completed.add(rna_id)
            for protein_id in observed:
                pair = (rna_id, protein_id)
                if pair in seen_pairs:
                    raise ValueError(f"duplicate prediction pair in existing parts: {pair}")
                seen_pairs.add(pair)
    return completed


def write_part_atomic(
    frame: pd.DataFrame,
    path: Path,
    run_hash: str,
    rank: int,
) -> None:
    """Write one Parquet part through an atomic rename."""
    validate_scores(frame)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({b"run_hash": run_hash.encode(), b"rank": str(rank).encode()})
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table.replace_schema_metadata(metadata), temporary, compression="zstd")
    os.replace(temporary, path)
    receipt = path.with_suffix(".sha256")
    temporary_receipt = receipt.with_suffix(".sha256.tmp")
    temporary_receipt.write_text(sha256_file(path) + "\n")
    os.replace(temporary_receipt, receipt)


def write_rank_manifest(
    output_dir: Path,
    rank: int,
    config: dict[str, Any],
    expected_rows: int,
) -> None:
    """Atomically record output hashes and coverage for one rank."""
    parts = part_paths(output_dir, rank)
    row_count = sum(pq.read_metadata(path).num_rows for path in parts)
    if row_count != expected_rows:
        raise RuntimeError(f"rank {rank} wrote {row_count} rows; expected {expected_rows}")
    manifest = {
        **config,
        "status": "complete",
        "expected_rows": expected_rows,
        "parts": [
            {
                "path": path.name,
                "rows": pq.read_metadata(path).num_rows,
                "sha256": sha256_file(path),
            }
            for path in parts
        ],
    }
    path = output_dir / f"rank-{rank:05d}-manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_model_and_tokenizers(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CoralPretraining, PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    """Load tokenizers, base encoders, and merged LoRA weights."""
    rna_tokenizer = AutoTokenizer.from_pretrained(args.rna_encoder, trust_remote_code=True)
    protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_encoder)
    config = CoralConfig(
        rna_encoder_checkpoint=args.rna_encoder,
        protein_encoder_checkpoint=args.protein_encoder,
    )
    LOGGER.info("Loading base encoders on rank-local device %s", device)
    base_model = CoralPretraining(config=config)
    peft_model = PeftModel.from_pretrained(base_model, args.weights)
    model = peft_model.merge_and_unload().float().to(device).eval()
    return model, rna_tokenizer, protein_tokenizer


def main() -> None:
    """Run or resume one deterministic torchrun shard."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    # Resolve cached Hub assets once; filenames and weight bytes enter the run hash.
    if not Path(args.protein_encoder).is_dir():
        from huggingface_hub import snapshot_download
        args.protein_encoder = snapshot_download(args.protein_encoder, local_files_only=True)
    args.rna_encoder = str(Path(args.rna_encoder).resolve())
    args.protein_encoder = str(Path(args.protein_encoder).resolve())
    started = time.monotonic()

    rank, world_size, _, device = execution_context()
    rnas = validate_sequence_frame(
        read_table(args.rnas),
        id_column="RNA_id",
        sequence_column="RNA_seqs",
        alphabet=RNA_ALPHABET,
        max_length=args.max_rna_nt,
    )
    proteins = validate_sequence_frame(
        read_table(args.proteins),
        id_column="Protein_id",
        sequence_column="Prot_seqs",
        alphabet=PROTEIN_ALPHABET,
        max_length=args.max_protein_aa,
    )
    rank_rnas = rnas.iloc[rank::world_size].reset_index(drop=True)
    protein_ids = set(proteins["Protein_id"])
    expected_rows = len(rank_rnas) * len(proteins)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_run_config(args, rank, world_size)
    completed = read_completed_rnas(
        args.output_dir, rank, config["run_hash"], protein_ids
    )
    if not completed.issubset(set(rank_rnas.RNA_id)):
        raise ValueError("resumed RNA IDs do not belong to this rank")
    remaining = rank_rnas.loc[~rank_rnas["RNA_id"].isin(completed)].reset_index(drop=True)
    LOGGER.info(
        "Rank %d/%d has %d completed and %d remaining RNAs",
        rank,
        world_size,
        len(completed),
        len(remaining),
    )

    diagnostics = {}
    if not remaining.empty:
        model, rna_tokenizer, protein_tokenizer = load_model_and_tokenizers(args, device)
        next_part = next_part_index(args.output_dir, rank)
        iterator = iter_all_vs_all_predictions(
            model,
            remaining,
            proteins,
            rna_tokenizer,
            protein_tokenizer,
            batch_size=args.batch_size,
            diagnostics=diagnostics,
            preflight=True,
            max_rna_nt=args.max_rna_nt,
            max_protein_aa=args.max_protein_aa,
        )
        progress_enabled = sys.stderr.isatty()
        progress = tqdm(
            total=len(rank_rnas),
            initial=len(completed),
            desc=f"Rank {rank}",
            unit="RNA",
            position=rank,
            leave=True,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not progress_enabled,
        )
        log_context = logging_redirect_tqdm() if progress_enabled else nullcontext()
        with progress, log_context:
            for offset, frame in enumerate(iterator):
                completed_in_part = int(frame["RNA_id"].nunique())
                if len(frame) != completed_in_part * len(proteins):
                    raise RuntimeError("prediction part has incomplete protein coverage")
                path = (
                    args.output_dir
                    / f"rank-{rank:05d}-part-{next_part + offset:06d}.parquet"
                )
                write_part_atomic(frame, path, config["run_hash"], rank)
                progress.update(completed_in_part)
                if offset % 100 == 0:
                    LOGGER.info("Rank %d wrote %s (%d rows)", rank, path.name, len(frame))

    final_completed = read_completed_rnas(
        args.output_dir, rank, config["run_hash"], protein_ids
    )
    if final_completed != set(rank_rnas["RNA_id"]):
        raise RuntimeError(f"rank {rank} output has incomplete or unexpected RNA coverage")
    config["runtime"] = {**diagnostics, "elapsed_seconds": time.monotonic() - started,
        "python": platform.python_version(), "torch": torch.__version__,
        "software": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "peft", "numpy", "pandas", "pyarrow")},
        "cuda": torch.version.cuda}
    write_rank_manifest(args.output_dir, rank, config, expected_rows)
    LOGGER.info("Rank %d complete: %d prediction rows", rank, expected_rows)


if __name__ == "__main__":
    main()
