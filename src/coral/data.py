"""Data utilities for CORAL: tokenization, collators, and small helpers.

This module consolidates concerns that used to live in separate files:

- `tokenize_function`, `seed_worker`, `specificity_score`, `read_fasta`: general helpers.
- `CoralDataCollator`: padding + optional MLM masking for training and inference.
- `DataCollatorForInference`: lookup-based padding for large all-vs-all inference.
- `DataCollatorForRBDPrediction`: padding + a 1D per-residue RNA-binding-domain
  mask for the RBD interpretability analysis.

All three collators emit the same four core keys — `{rna,protein}_input_ids` and
`{rna,protein}_attention_mask` — plus whatever labels their analysis needs, so a
caller can swap one for another without touching the training loop.
"""

import random
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from .rbd import intervals_to_residue_mask


RNA_ALPHABET = frozenset("ACGT")
PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


# --- General helpers ---------------------------------------------------------

def seed_worker(worker_id: int) -> None:
    """Per-worker seeder so DataLoader shuffling is reproducible."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def tokenize_function(examples, rna_tokenizer, protein_tokenizer):
    """Tokenize a batch of RNA + protein sequences with no truncation or padding."""
    rna_tokenized = rna_tokenizer(
        examples["RNA_seqs"],
        truncation=False,
        padding=False,
        return_special_tokens_mask=True,
    )

    protein_tokenized = protein_tokenizer(
        examples["Prot_seqs"],
        truncation=False,
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


def read_fasta(file: str) -> pd.DataFrame:
    """Parse a FASTA file into a two-column (seq id, sequence) DataFrame."""
    with open(file) as f:
        fasta_seqs = f.read().split(">")[1:]
        ids = [seq.split("\n")[0].split("|")[0] for seq in fasta_seqs]
        seqs = ["".join(seq.split("\n")[1:]) for seq in fasta_seqs]
        return pd.DataFrame(list(zip(ids, seqs)), columns=["seq id", "sequence"])


def specificity_score(y_true, y_pred) -> float:
    """True-negative rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)


# --- Large-scale inference --------------------------------------------------

def validate_sequence_frame(
    frame: pd.DataFrame,
    *,
    id_column: str,
    sequence_column: str,
    alphabet: frozenset[str],
    max_length: int,
) -> pd.DataFrame:
    """Validate and return a position-indexed inference manifest."""
    missing = [column for column in (id_column, sequence_column) if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if max_length < 1:
        raise ValueError("maximum sequence length must be positive")

    result = frame.loc[:, [id_column, sequence_column]].reset_index(drop=True).copy()
    if result.empty:
        raise ValueError(f"{id_column}/{sequence_column} manifest must not be empty")
    for row_index, (identifier, sequence) in enumerate(
        result.itertuples(index=False, name=None)
    ):
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{id_column} at row {row_index} must be a nonempty string")
        if identifier != identifier.strip():
            raise ValueError(f"{id_column} at row {row_index} has surrounding whitespace")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"{sequence_column} at row {row_index} must be a nonempty string")
        if len(sequence) > max_length:
            raise ValueError(
                f"{sequence_column} at row {row_index} has length {len(sequence)}, "
                f"above the limit {max_length}"
            )
        invalid = set(sequence) - alphabet
        if invalid:
            symbols = "".join(sorted(invalid))
            raise ValueError(
                f"{sequence_column} at row {row_index} contains invalid symbols: {symbols}"
            )

    for column in (id_column, sequence_column):
        duplicate = result[column].duplicated(keep=False)
        if duplicate.any():
            value = result.loc[duplicate, column].iloc[0]
            raise ValueError(f"{column} values must be unique; duplicate: {value}")
    return result


def tokenize_to_lookup(
    sequences: Sequence[str],
    tokenizer,
    *,
    chunk_size: int = 1_024,
) -> list[list[int]]:
    """Tokenize unique sequences once without padding or truncation."""
    if chunk_size < 1:
        raise ValueError("tokenization chunk size must be positive")

    token_ids: list[list[int]] = []
    for start in range(0, len(sequences), chunk_size):
        encoded = tokenizer(
            list(sequences[start : start + chunk_size]),
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"]
        if len(input_ids) != min(chunk_size, len(sequences) - start):
            raise RuntimeError("tokenizer returned an unexpected number of sequences")
        token_ids.extend([list(map(int, values)) for values in input_ids])

    if len(token_ids) != len(sequences) or any(not values for values in token_ids):
        raise RuntimeError("tokenization produced an incomplete or empty lookup table")
    return token_ids


class InferenceRNADataset(torch.utils.data.Dataset):
    """Return RNA lookup indices in a deterministic caller-provided order."""

    def __init__(self, rna_order: Sequence[int]):
        self.rna_order = list(rna_order)

    def __len__(self) -> int:
        return len(self.rna_order)

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"rna_index": self.rna_order[index]}


class DataCollatorForInference:
    """Expand RNA indices across proteins and dynamically pad token lookups."""

    def __init__(
        self,
        rna_token_ids: Sequence[Sequence[int]],
        protein_token_ids: Sequence[Sequence[int]],
        rna_tokenizer,
        protein_tokenizer,
        *,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        if not rna_token_ids:
            raise ValueError("RNA token lookup must not be empty")
        if not protein_token_ids:
            raise ValueError("protein token lookup must not be empty")
        self.rna_token_ids = rna_token_ids
        self.protein_token_ids = protein_token_ids
        self.rna_tokenizer = rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad(self, token_ids: Sequence[Sequence[int]], tokenizer) -> dict[str, torch.Tensor]:
        features = [{"input_ids": values} for values in token_ids]
        return tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rna_indices = [int(feature["rna_index"]) for feature in features]
        protein_count = len(self.protein_token_ids)
        expanded_rna_indices = [
            rna_index for rna_index in rna_indices for _ in range(protein_count)
        ]
        protein_indices = list(range(protein_count)) * len(rna_indices)

        batch_rna = self._pad(
            [self.rna_token_ids[index] for index in expanded_rna_indices],
            self.rna_tokenizer,
        )
        batch_protein = self._pad(
            [self.protein_token_ids[index] for index in protein_indices],
            self.protein_tokenizer,
        )
        return {
            "rna_input_ids": batch_rna["input_ids"],
            "rna_attention_mask": batch_rna["attention_mask"],
            "protein_input_ids": batch_protein["input_ids"],
            "protein_attention_mask": batch_protein["attention_mask"],
            "rna_indices": torch.tensor(expanded_rna_indices, dtype=torch.long),
            "protein_indices": torch.tensor(protein_indices, dtype=torch.long),
        }


# --- Standard collator (training + inference) --------------------------------

class CoralDataCollator:
    """Pads RNA/protein batches and optionally applies BERT-style MLM masking."""

    def __init__(
        self,
        rna_tokenizer,
        protein_tokenizer,
        mlm_probability: Optional[float] = None,
        padding: bool = True,
        rna_max_length: Optional[int] = None,
        protein_max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.rna_tokenizer = rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.mlm_probability = mlm_probability
        self.padding = padding
        self.rna_max_length = rna_max_length
        self.protein_max_length = protein_max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad(self, features, tokenizer, max_length):
        return tokenizer.pad(
            features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

    def torch_mask_tokens(self, inputs, special_tokens_mask, tokenizer):
        """Standard BERT MLM: 80% [MASK], 10% random token, 10% unchanged."""
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [
                    tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in labels.tolist()
                ],
                dtype=torch.bool,
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # loss computed only on masked tokens

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def __call__(self, features):
        features_rna = [
            {
                "input_ids": f["rna_input_ids"],
                "attention_mask": f["rna_attention_mask"],
                "special_tokens_mask": f["rna_special_tokens_mask"],
            }
            for f in features
        ]

        features_proteins = [
            {
                "input_ids": f["protein_input_ids"],
                "attention_mask": f["protein_attention_mask"],
                "special_tokens_mask": f["protein_special_tokens_mask"],
            }
            for f in features
        ]

        batch_rna = self._pad(features_rna, self.rna_tokenizer, self.rna_max_length)
        batch_protein = self._pad(features_proteins, self.protein_tokenizer, self.protein_max_length)

        rna_ids = batch_rna["input_ids"]
        protein_ids = batch_protein["input_ids"]

        batch = {
            "rna_input_ids": rna_ids,
            "protein_input_ids": protein_ids,
            "rna_attention_mask": batch_rna["attention_mask"],
            "protein_attention_mask": batch_protein["attention_mask"],
            # Passed through for callers that need to locate CLS/SEP/PAD, e.g.
            # attention analyses cropping to pure sequence positions.
            "rna_special_tokens_mask": batch_rna["special_tokens_mask"],
            "protein_special_tokens_mask": batch_protein["special_tokens_mask"],
        }

        if self.mlm_probability:
            masked_rna_ids, rna_labels = self.torch_mask_tokens(
                rna_ids, batch_rna["special_tokens_mask"], self.rna_tokenizer
            )
            masked_protein_ids, protein_labels = self.torch_mask_tokens(
                protein_ids, batch_protein["special_tokens_mask"], self.protein_tokenizer
            )
            batch["rna_input_ids"] = masked_rna_ids
            batch["protein_input_ids"] = masked_protein_ids
            batch["rna_labels_for_MLM"] = rna_labels
            batch["protein_labels_for_MLM"] = protein_labels

        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        return batch


# --- RBD-prediction collator (attention interpretability) --------------------

class DataCollatorForRBDPrediction:
    """Builds per-residue RNA-binding-domain masks for the protein side.

    The labels are a 1D per-residue mask (RBD vs non-RBD). ESM-2 emits one
    token per residue, so no nucleotide-to-BPE mapping is needed and the mask
    comes straight from the parsed RBD intervals.

    `rbd_labels` is shape (batch, protein_len): `-100` at CLS/EOS/padding,
    `0`/`1` for non-RBD/RBD residues.
    """

    def __init__(
        self,
        rna_tokenizer,
        protein_tokenizer,
        all_rbd_intervals: Dict[str, List[Tuple[int, int]]],
        padding: bool = True,
        rna_max_length: Optional[int] = None,
        protein_max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.rna_tokenizer = rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.all_rbd_intervals = all_rbd_intervals
        self.padding = padding
        self.rna_max_length = rna_max_length
        self.protein_max_length = protein_max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad(self, features, tokenizer, max_length):
        return tokenizer.pad(
            features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

    def get_rbd_labels(self, batch_proteins, protein_keys):
        special = batch_proteins["special_tokens_mask"]
        batch_size, seq_len = special.shape
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        for i, key in enumerate(protein_keys):
            residue_positions = (special[i] == 0).nonzero(as_tuple=False).flatten()
            n_residues = residue_positions.numel()
            if n_residues == 0:
                continue
            intervals = self.all_rbd_intervals.get(key, [])
            residue_mask = intervals_to_residue_mask(intervals, n_residues)
            labels[i, residue_positions] = torch.from_numpy(
                residue_mask.astype(np.int64)
            )

        return labels

    def __call__(self, features):
        features_rna = [
            {
                "input_ids": f["rna_input_ids"],
                "attention_mask": f["rna_attention_mask"],
                "special_tokens_mask": f["rna_special_tokens_mask"],
            }
            for f in features
        ]

        features_proteins = [
            {
                "input_ids": f["protein_input_ids"],
                "attention_mask": f["protein_attention_mask"],
                "special_tokens_mask": f["protein_special_tokens_mask"],
            }
            for f in features
        ]

        protein_keys = [f["Protein_id"] for f in features]
        pair_keys = [f'{f["RNA_id"]}|{f["Protein_id"]}' for f in features]

        batch_rna = self._pad(features_rna, self.rna_tokenizer, self.rna_max_length)
        batch_proteins = self._pad(features_proteins, self.protein_tokenizer,
                                   self.protein_max_length)

        rbd_labels = self.get_rbd_labels(batch_proteins, protein_keys)

        return {
            "rna_input_ids": batch_rna["input_ids"],
            "protein_input_ids": batch_proteins["input_ids"],
            "rna_attention_mask": batch_rna["attention_mask"],
            "protein_attention_mask": batch_proteins["attention_mask"],
            "rbd_labels": rbd_labels,
            "protein_keys": protein_keys,
            "pair_keys": pair_keys,
        }
