"""Inference utilities for CORAL.

Exposes two entry points:

- `predict_interactions`: score a pre-specified list of RNA-protein pairs.
- `predict_allVsall_interactions`: score every RNA x every protein combination.

Both functions expect a model that has been loaded via the recipe in
`scripts/predict.py` (PeftModel.from_pretrained + merge_and_unload).
"""

import gc
from typing import Optional, Union

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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .data import CoralDataCollator


class RNAProteinDataset(Dataset):
    """All-vs-all RNA x protein pair dataset with eager tokenization."""

    def __init__(self, rna_df, protein_df, rna_tokenizer, protein_tokenizer):
        self.rna_df = rna_df.reset_index(drop=True)
        self.protein_df = protein_df.reset_index(drop=True)
        self.rna_tokenizer = rna_tokenizer
        self.protein_tokenizer = protein_tokenizer

        print("Pre-tokenizing RNA sequences...")
        self.tokenized_rna = self.rna_tokenizer(
            self.rna_df["sequence"].tolist(),
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )

        print("Pre-tokenizing protein sequences...")
        self.tokenized_proteins = self.protein_tokenizer(
            self.protein_df["sequence"].tolist(),
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )

        self.n_rna = len(self.rna_df)
        self.n_protein = len(self.protein_df)
        self.total_pairs = self.n_rna * self.n_protein

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        rna_idx = idx // self.n_protein
        protein_idx = idx % self.n_protein

        return {
            "rna_tokens": self.tokenized_rna["input_ids"][rna_idx],
            "protein_tokens": self.tokenized_proteins["input_ids"][protein_idx],
            "rna_id": self.rna_df.iloc[rna_idx]["seq id"],
            "protein_id": self.protein_df.iloc[protein_idx]["seq id"],
            "rna_attention_mask": self.tokenized_rna["attention_mask"][rna_idx],
            "protein_attention_mask": self.tokenized_proteins["attention_mask"][protein_idx],
            "pair_idx": idx,
        }


def predict_allVsall_interactions(
    model,
    rna_df,
    protein_df,
    rna_tokenizer,
    protein_tokenizer,
    batch_size: int = 4,
    device: str = "cuda",
):
    """Score every (RNA, protein) combination from two sequence frames."""
    dataset = RNAProteinDataset(rna_df, protein_df, rna_tokenizer, protein_tokenizer)
    print("Dataset size: ", len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    model.to(device)

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting interactions"):
            rna_batch = {
                "input_ids": batch["rna_tokens"].to(device),
                "attention_mask": batch["rna_attention_mask"].to(device),
            }
            protein_batch = {
                "input_ids": batch["protein_tokens"].to(device),
                "attention_mask": batch["protein_attention_mask"].to(device),
            }

            predictions, match_probabilities = model(rna_batch, protein_batch)

            for i, prob in enumerate(match_probabilities.cpu().numpy()):
                results.append(
                    {
                        "rna_id": batch["rna_id"][i],
                        "protein_id": batch["protein_id"][i],
                        "interaction_probability": prob[1].item(),
                        "prediction": predictions[i].item(),
                    }
                )

            if len(results) % (batch_size * 100) == 0:
                torch.cuda.empty_cache()
                gc.collect()

    return pd.DataFrame(results)


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

    print(f"Processing {len(dataset)} RNA-protein pairs in {len(dataloader)} batches...")

    model.eval()
    model.to(device)

    results = []
    all_embeddings = [] if return_embeddings else None
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

    if return_embeddings:
        results_df["embeddings"] = all_embeddings[: len(results_df)]

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

        print("\nPrediction Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

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
