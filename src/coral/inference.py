"""Inference utilities for CORAL.

Exposes three entry points:

- `predict_interactions`: score a pre-specified list of RNA-protein pairs.
- `iter_all_vs_all_predictions`: stream every RNA x protein combination.
- `predict_all_vs_all_interactions`: materialize a bounded all-vs-all job.

The functions expect a model that has been loaded via the recipe in
`scripts/predict.py` (PeftModel.from_pretrained + merge_and_unload).

Preflight checks stay beside the scoring helpers and reuse prepared tokens.
This intentionally exceeds the 400-line target to keep inference setup local.
"""

import gc
import logging
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .data import (
    PROTEIN_ALPHABET,
    RNA_ALPHABET,
    CoralDataCollator,
    DataCollatorForInference,
    InferenceRNADataset,
    tokenize_to_lookup,
    validate_sequence_frame,
)
from .model import CoralPretraining
from .prediction_io import COLUMNS as OUTPUT_COLUMNS


LOGGER = logging.getLogger(__name__)
RNA_TOKEN_BUCKETS = (128, 256, 512, 768, 1_024)
MAX_AUTO_BATCH_SIZE = 256


def _model_device(model: CoralPretraining) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _autocast_context(device: torch.device) -> AbstractContextManager:
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _token_bucket(token_length: int) -> int:
    for bucket_index, upper_bound in enumerate(RNA_TOKEN_BUCKETS):
        if token_length <= upper_bound:
            return bucket_index
    return len(RNA_TOKEN_BUCKETS)


def _bucketed_indices(token_ids: list[list[int]]) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(len(RNA_TOKEN_BUCKETS) + 1)]
    for index, values in enumerate(token_ids):
        buckets[_token_bucket(len(values))].append(index)
    for bucket in buckets:
        bucket.sort(key=lambda index: (len(token_ids[index]), index))
    return buckets


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[dict, dict]:
    non_blocking = device.type == "cuda"
    rna_batch = {
        "input_ids": batch["rna_input_ids"].to(device, non_blocking=non_blocking),
        "attention_mask": batch["rna_attention_mask"].to(device, non_blocking=non_blocking),
    }
    protein_batch = {
        "input_ids": batch["protein_input_ids"].to(device, non_blocking=non_blocking),
        "attention_mask": batch["protein_attention_mask"].to(
            device, non_blocking=non_blocking
        ),
    }
    return rna_batch, protein_batch


def _score_batch(
    model: CoralPretraining,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rna_batch, protein_batch = _move_batch(batch, device)
    with _autocast_context(device):
        _, probabilities, logit_margin = model(
            rna_batch,
            protein_batch,
            return_logit_margin=True,
        )
    if probabilities.dtype != torch.float32 or logit_margin.dtype != torch.float32:
        raise TypeError("all-vs-all inference requires float32 outputs")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(logit_margin).all():
        raise FloatingPointError("CORAL produced a non-finite score")
    probability_sums = probabilities.double().sum(dim=1)
    if not torch.allclose(probability_sums, torch.ones_like(probability_sums), atol=2e-6):
        raise FloatingPointError("CORAL interaction probabilities do not sum to one")
    predictions = (logit_margin > 0).to(torch.long)
    return predictions, probabilities, logit_margin


def _check_float32(
    model: CoralPretraining,
    rnas: pd.DataFrame,
    proteins: pd.DataFrame,
    collator: DataCollatorForInference,
) -> dict[str, Any]:
    """Check tokenization, repeatability and padding using the inference lookups."""
    device = _model_device(model)
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise TypeError("model contains parameters outside float32")
    unpadded = DataCollatorForInference(
        collator.rna_token_ids, collator.protein_token_ids,
        collator.rna_tokenizer, collator.protein_tokenizer, pad_to_multiple_of=None,
    )
    checks = []
    start = time.monotonic()
    with torch.inference_mode():
        for bucket in _bucketed_indices(collator.rna_token_ids):
            if not bucket:
                continue
            index = bucket[-1]
            features = [{"rna_index": index}]
            batch = collator(features)
            direct_rna = collator.rna_tokenizer(
                [rnas.iloc[index].RNA_seqs] * len(proteins), padding=True,
                truncation=False, pad_to_multiple_of=8, return_tensors="pt",
            )
            direct_protein = collator.protein_tokenizer(
                proteins.Prot_seqs.tolist(), padding=True,
                truncation=False, pad_to_multiple_of=8, return_tensors="pt",
            )
            for prefix, direct in (("rna", direct_rna), ("protein", direct_protein)):
                for key in ("input_ids", "attention_mask"):
                    if not torch.equal(batch[f"{prefix}_{key}"], direct[key]):
                        raise ValueError("lookup and direct tokenization disagree")
            first = _score_batch(model, batch, device)[2]
            repeat = _score_batch(model, batch, device)[2]
            other = _score_batch(model, unpadded(features), device)[2]
            if not torch.equal(first, repeat):
                raise ValueError("repeated float32 scores differ")
            error = float((first - other).abs().max())
            if not torch.allclose(first, other, atol=5e-4, rtol=1e-5):
                raise ValueError(f"padding-dependent float32 margin: {error}")
            checks.append({"rna_tokens": len(collator.rna_token_ids[index]),
                           "max_padding_margin_error": error})
    return {
        "precision": "float32", "tf32": False, "checks": checks,
        "elapsed_seconds": time.monotonic() - start, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def _autotune_batch_size(
    model: CoralPretraining,
    collator: DataCollatorForInference,
    bucket: list[int],
    device: torch.device,
) -> int:
    if device.type != "cuda":
        LOGGER.info("Using RNA batch size 1 because CUDA autotuning is unavailable")
        return 1

    longest_rna = bucket[-1]
    memory_limit = int(torch.cuda.get_device_properties(device).total_memory * 0.85)
    safe_batch_size = 0
    candidate = 1
    maximum = min(MAX_AUTO_BATCH_SIZE, len(bucket))
    while candidate <= maximum:
        batch = collator([{"rna_index": longest_rna}] * candidate)
        try:
            torch.cuda.reset_peak_memory_stats(device)
            _score_batch(model, batch, device)
            peak_memory = torch.cuda.max_memory_allocated(device)
        except torch.OutOfMemoryError:
            LOGGER.info("RNA batch size %d exhausted CUDA memory", candidate)
            torch.cuda.empty_cache()
            break
        if peak_memory > memory_limit:
            break
        safe_batch_size = candidate
        candidate *= 2

    if safe_batch_size == 0:
        raise RuntimeError("RNA batch size 1 exceeds the CUDA memory budget")
    LOGGER.info(
        "Selected RNA batch size %d for token length %d",
        safe_batch_size,
        len(collator.rna_token_ids[longest_rna]),
    )
    return safe_batch_size


def iter_all_vs_all_predictions(
    model: CoralPretraining,
    rnas: pd.DataFrame,
    proteins: pd.DataFrame,
    rna_tokenizer: PreTrainedTokenizerBase,
    protein_tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int | Literal["auto"] = "auto",
    precision: Literal["float32"] = "float32",
    max_rna_nt: int = 6_000,
    max_protein_aa: int = 1_250,
    shard_index: int = 0,
    num_shards: int = 1,
    diagnostics: dict | None = None,
    preflight: bool = False,
) -> Iterator[pd.DataFrame]:
    """Yield bounded prediction batches, optionally checking numerics before scoring.

    Preflight reuses this shard's token lookups and records its results in
    diagnostics when supplied. A failed check raises before any batch is yielded.
    """
    if precision != "float32":
        raise ValueError("only float32 inference is supported")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if batch_size != "auto" and (not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer or 'auto'")

    valid_rnas = validate_sequence_frame(
        rnas,
        id_column="RNA_id",
        sequence_column="RNA_seqs",
        alphabet=RNA_ALPHABET,
        max_length=max_rna_nt,
    )
    valid_proteins = validate_sequence_frame(
        proteins,
        id_column="Protein_id",
        sequence_column="Prot_seqs",
        alphabet=PROTEIN_ALPHABET,
        max_length=max_protein_aa,
    )
    valid_rnas = valid_rnas.iloc[shard_index::num_shards].reset_index(drop=True)
    if valid_rnas.empty:
        return

    LOGGER.info("Tokenizing %d unique RNA sequences", len(valid_rnas))
    rna_token_ids = tokenize_to_lookup(valid_rnas["RNA_seqs"].tolist(), rna_tokenizer)
    LOGGER.info("Tokenizing %d unique protein sequences", len(valid_proteins))
    protein_token_ids = tokenize_to_lookup(
        valid_proteins["Prot_seqs"].tolist(), protein_tokenizer
    )
    collator = DataCollatorForInference(
        rna_token_ids,
        protein_token_ids,
        rna_tokenizer,
        protein_tokenizer,
        pad_to_multiple_of=8,
    )

    device = _model_device(model)
    model.float().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if preflight:
        checks = _check_float32(model, valid_rnas, valid_proteins, collator)
        if diagnostics is not None:
            diagnostics["preflight"] = checks
        LOGGER.info("Float32 preflight: %s", checks)
    with torch.inference_mode():
        for bucket in _bucketed_indices(rna_token_ids):
            if not bucket:
                continue
            current_batch_size = (
                _autotune_batch_size(model, collator, bucket, device)
                if batch_size == "auto"
                else batch_size
            )
            if diagnostics is not None:
                diagnostics.setdefault("buckets", []).append({
                    "max_tokens": len(rna_token_ids[bucket[-1]]),
                    "rna_count": len(bucket), "rna_batch_size": current_batch_size,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                })
            dataloader = DataLoader(
                InferenceRNADataset(bucket),
                batch_size=current_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            for batch in dataloader:
                predictions, probabilities, logit_margin = _score_batch(model, batch, device)
                rna_indices = batch["rna_indices"].tolist()
                protein_indices = batch["protein_indices"].tolist()
                yield pd.DataFrame(
                    {
                        "RNA_id": [valid_rnas.iloc[index]["RNA_id"] for index in rna_indices],
                        "Protein_id": [
                            valid_proteins.iloc[index]["Protein_id"]
                            for index in protein_indices
                        ],
                        "logit_margin": logit_margin.float().cpu().numpy(),
                        "interaction_probability": probabilities[:, 1].float().cpu().numpy(),
                        "prediction": predictions.to(torch.int8).cpu().numpy(),
                    },
                    columns=OUTPUT_COLUMNS,
                )


def predict_all_vs_all_interactions(
    model: CoralPretraining,
    rnas: pd.DataFrame,
    proteins: pd.DataFrame,
    rna_tokenizer: PreTrainedTokenizerBase,
    protein_tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int | Literal["auto"] = "auto",
    precision: Literal["float32"] = "float32",
    max_rna_nt: int = 6_000,
    max_protein_aa: int = 1_250,
    shard_index: int = 0,
    num_shards: int = 1,
) -> pd.DataFrame:
    """Materialize all-vs-all predictions for a bounded input."""
    parts = list(
        iter_all_vs_all_predictions(
            model,
            rnas,
            proteins,
            rna_tokenizer,
            protein_tokenizer,
            batch_size=batch_size,
            precision=precision,
            max_rna_nt=max_rna_nt,
            max_protein_aa=max_protein_aa,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    )
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def tokenize_for_inference(
    examples,
    rna_tokenizer,
    protein_tokenizer,
    max_rna_length: Optional[int] = None,
    max_protein_length: Optional[int] = None,
):
    """Per-pair tokenization used inside `predict_interactions`."""
    rna_tokenized = rna_tokenizer(
        examples["RNA_seqs"],
        truncation=bool(max_rna_length),
        max_length=max_rna_length,
        padding=False,
        return_special_tokens_mask=True,
    )
    protein_tokenized = protein_tokenizer(
        examples["Prot_seqs"],
        truncation=bool(max_protein_length),
        max_length=max_protein_length,
        padding=False,
        return_special_tokens_mask=True,
    )

    return {
        "rna_input_ids": rna_tokenized["input_ids"],
        "rna_attention_mask": rna_tokenized["attention_mask"],
        "rna_special_tokens_mask": rna_tokenized["special_tokens_mask"],
        "protein_input_ids": protein_tokenized["input_ids"],
        "protein_attention_mask": protein_tokenized["attention_mask"],
        "protein_special_tokens_mask": protein_tokenized["special_tokens_mask"],
    }


def decode_sequences(input_ids: torch.Tensor, tokenizer):
    """Decode a batch of input_ids back to strings, skipping special tokens."""
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)


def process_attention_scores(
    cross_attentions,
    rna_batch,
    protein_batch,
    rna_tokenizer,
    protein_tokenizer,
    batch_size,
):
    """Extract the heads [0, 7] average of cross-attention and trim to real tokens."""
    rna_to_protein_att, protein_to_rna_att = cross_attentions

    rna_to_protein_att = rna_to_protein_att[:, -1, [1, 2]].mean(dim=1)
    protein_to_rna_att = protein_to_rna_att[:, -1, [1, 2]].mean(dim=1)

    rna_tokens = decode_sequences(rna_batch["input_ids"], rna_tokenizer)
    protein_tokens = decode_sequences(protein_batch["input_ids"], protein_tokenizer)

    processed_rna_to_protein_att = []
    processed_protein_to_rna_att = []

    for i in range(batch_size):
        protein_len = len(protein_tokens[i].split(" "))
        rna_len = len(rna_tokens[i].split(" "))

        rna_att = rna_to_protein_att[i][1 : rna_len + 1, 1 : protein_len + 1].cpu().numpy()
        protein_att = protein_to_rna_att[i][1 : protein_len + 1, 1 : rna_len + 1].cpu().numpy()

        processed_rna_to_protein_att.append(rna_att)
        processed_protein_to_rna_att.append(protein_att)

    return {
        "rna_to_protein_attentions": processed_rna_to_protein_att,
        "protein_to_rna_attentions": processed_protein_to_rna_att,
    }


def predict_interactions(
    model,
    interactions_df: pd.DataFrame,
    rna_tokenizer: AutoTokenizer,
    protein_tokenizer: AutoTokenizer,
    batch_size: int = 4,
    save_attentions: bool = False,
    device: str = "cuda",
    padding: Union[str, None] = None,
    max_rna_length: Optional[int] = None,
    max_protein_length: Optional[int] = None,
    num_workers: int = 4,
    return_embeddings: bool = False,
    pad_to_multiple_of: Optional[int] = None,
) -> pd.DataFrame:
    """Score a given list of RNA-protein pairs.

    Args:
        interactions_df: Must contain RNA_id, Protein_id, RNA_seqs, Prot_seqs.
            If a `labels` column is present, metrics are reported.
        save_attentions: If True, also return a dict of per-pair cross-attention arrays.
        max_rna_length, max_protein_length: Truncation limits (paper uses 6000 / 1250).

    Returns:
        Results DataFrame, optionally paired with an attention dict.
    """
    required_cols = ["RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs"]
    missing_cols = [col for col in required_cols if col not in interactions_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if return_embeddings:
        raise NotImplementedError("predict_interactions does not expose model embeddings")

    has_labels = "labels" in interactions_df.columns

    dataset = HFDataset.from_pandas(interactions_df)

    tokenized_dataset = dataset.map(
        tokenize_for_inference,
        batched=True,
        num_proc=num_workers,
        remove_columns=["RNA_seqs", "Prot_seqs"],
        fn_kwargs={
            "rna_tokenizer": rna_tokenizer,
            "protein_tokenizer": protein_tokenizer,
            "max_rna_length": max_rna_length,
            "max_protein_length": max_protein_length,
        },
    )

    data_collator = CoralDataCollator(
        rna_tokenizer=rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        rna_max_length=max_rna_length if padding else None,
        protein_max_length=max_protein_length if padding else None,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )

    LOGGER.info(
        "Processing %d RNA-protein pairs in %d batches", len(dataset), len(dataloader)
    )

    model.eval()
    model.to(device)

    results = []
    all_rna_to_protein_attentions = {} if save_attentions else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting interactions")):
            rna_batch = {
                "input_ids": batch["rna_input_ids"].to(device),
                "attention_mask": batch["rna_attention_mask"].to(device),
            }
            protein_batch = {
                "input_ids": batch["protein_input_ids"].to(device),
                "attention_mask": batch["protein_attention_mask"].to(device),
            }

            batch_size_actual = batch["rna_input_ids"].size(0)

            if save_attentions:
                predictions, match_probabilities, crossattentions, _ = model(
                    rna_batch, protein_batch, output_attentions=True
                )
                processed_crossattentions = process_attention_scores(
                    crossattentions,
                    rna_batch,
                    protein_batch,
                    rna_tokenizer,
                    protein_tokenizer,
                    batch_size_actual,
                )
                batch_rna_to_protein_attentions = processed_crossattentions["rna_to_protein_attentions"]
            else:
                predictions, match_probabilities = model(rna_batch, protein_batch)

            start_idx = batch_idx * batch_size

            for i in range(batch_size_actual):
                idx = start_idx + i
                if idx < len(interactions_df):
                    result = {
                        "RNA_id": interactions_df.iloc[idx]["RNA_id"],
                        "Protein_id": interactions_df.iloc[idx]["Protein_id"],
                        "PDB_id": interactions_df.iloc[idx]["Protein_id"],
                        "RNA_seqs": interactions_df.iloc[idx]["RNA_seqs"],
                        "Prot_seqs": interactions_df.iloc[idx]["Prot_seqs"],
                        "prediction": predictions[i].item(),
                        "interaction_probability": match_probabilities[i, 1].item(),
                        "non_interaction_probability": match_probabilities[i, 0].item(),
                    }

                    if has_labels:
                        result["true_label"] = interactions_df.iloc[idx]["labels"]

                    if save_attentions:
                        key = f'{result["RNA_id"]}|{result["Protein_id"]}'
                        all_rna_to_protein_attentions[key] = batch_rna_to_protein_attentions[i]

                    results.append(result)

            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    results_df = pd.DataFrame(results)

    if has_labels:
        y_true = results_df["true_label"].values
        y_pred = results_df["prediction"].values
        y_prob = results_df["interaction_probability"].values

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        }

        for metric, value in metrics.items():
            LOGGER.info("%s: %.4f", metric, value)

    if save_attentions:
        return results_df, all_rna_to_protein_attentions
    return results_df


def prepare_interactions_dataframe(
    positive_interactions: pd.DataFrame,
    rna_seqs: pd.DataFrame,
    protein_seqs: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Join a list of positive (RNA_id, Protein_id) pairs with their sequences."""
    df = positive_interactions.copy()
    df["labels"] = 1

    df = df.merge(rna_seqs[["RNA_id", "RNA_seqs"]], on="RNA_id", how="left")
    df = df.merge(protein_seqs[["Protein_id", "Prot_seqs"]], on="Protein_id", how="left")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df[["RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs", "labels"]]
