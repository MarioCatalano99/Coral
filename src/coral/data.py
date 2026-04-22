"""Data utilities for CORAL: tokenization, collators, and small helpers.

This module consolidates three concerns that used to live in separate files:

- `tokenize_function`, `seed_worker`, `specificity_score`, `read_fasta`: general helpers.
- `CoralDataCollator`: padding + optional MLM masking for training and inference.
- `DataCollatorForContactPrediction`: padding + structural-contact label construction
  for the attention-interpretability analysis.
"""

import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


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


# --- Contact-prediction collator (attention interpretability) ----------------

class DataCollatorForContactPrediction:
    """Builds per-pair contact-label tensors for the cross-attention analysis."""

    def __init__(
        self,
        rna_tokenizer,
        protein_tokenizer,
        all_contacts: dict,
        padding: bool = True,
        rna_max_length: Optional[int] = None,
        protein_max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.rna_tokenizer = rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.all_contacts = all_contacts
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

    def get_position_to_token(self, rna_tokens):
        position_to_token = {}
        current_position = 0
        for token_idx, token in enumerate(rna_tokens):
            for i in range(len(token)):
                position_to_token[current_position + i] = token_idx
            current_position += len(token)
        return position_to_token

    def map_rna_tokens_positions(self, rna_tokens, contacts):
        token_contacts = []
        for tokens, current_contacts in zip(rna_tokens, contacts):
            new_contacts = []
            position_to_token = self.get_position_to_token(tokens)
            for nucleotide_pos, amino_pos in current_contacts:
                if nucleotide_pos in position_to_token:
                    token_idx = position_to_token[nucleotide_pos]
                    new_contacts.append([token_idx + 1, amino_pos + 1])
                else:
                    print(f"Warning: Nucleotide position {nucleotide_pos} not found in tokens")
                    new_contacts.append([None, amino_pos])
            token_contacts.append(new_contacts)
        return token_contacts

    def get_contact_labels(self, batch_rna, batch_proteins, contact_keys):
        batch_rna_input_ids = batch_rna["input_ids"]
        batch_rna_special_tokens_mask = batch_rna["special_tokens_mask"]
        batch_protein_input_ids = batch_proteins["input_ids"]
        batch_protein_special_tokens_mask = batch_proteins["special_tokens_mask"]

        rna_tokens = [
            self.rna_tokenizer.convert_ids_to_tokens(rna_input_ids[~rna_special_tokens_mask.bool()])
            for rna_input_ids, rna_special_tokens_mask in zip(
                batch_rna_input_ids, batch_rna_special_tokens_mask
            )
        ]

        sequence_contacts = [self.all_contacts[key] for key in contact_keys]
        token_contacts = self.map_rna_tokens_positions(rna_tokens, sequence_contacts)
        all_contact_labels = []

        for i in range(len(batch_rna_input_ids)):
            rna_input_ids = batch_rna_input_ids[i]
            protein_input_ids = batch_protein_input_ids[i]
            current_contacts = token_contacts[i]
            rna_interface_positions, protein_interface_positions = zip(*current_contacts)

            rna_special_tokens_mask = batch_rna_special_tokens_mask[i]
            protein_special_tokens_mask = batch_protein_special_tokens_mask[i]

            rna_expanded = (1 - rna_special_tokens_mask).unsqueeze(-1)
            protein_expanded = (1 - protein_special_tokens_mask).unsqueeze(0)

            att_mask = torch.logical_not(rna_expanded * protein_expanded).bool()

            contact_labels = torch.zeros(len(rna_input_ids), len(protein_input_ids))
            contact_labels[rna_interface_positions, protein_interface_positions] = 1
            contact_labels[att_mask] = -100
            all_contact_labels.append(contact_labels)

        return torch.stack(all_contact_labels, dim=0)

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

        contact_keys = [f'{f["RNA_id"]}_{f["Protein_id"][-1]}' for f in features]

        batch_rna = self._pad(features_rna, self.rna_tokenizer, self.rna_max_length)
        batch_proteins = self._pad(features_proteins, self.protein_tokenizer, self.protein_max_length)

        contact_labels = self.get_contact_labels(batch_rna, batch_proteins, contact_keys)

        return {
            "rna_input_ids": batch_rna["input_ids"],
            "protein_input_ids": batch_proteins["input_ids"],
            "rna_attention_mask": batch_rna["attention_mask"],
            "protein_attention_mask": batch_proteins["attention_mask"],
            "contact_labels": contact_labels,
            "contact_keys": contact_keys,
        }
