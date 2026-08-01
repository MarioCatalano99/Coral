"""Shared inference scaffolding for contact_analysis scripts.

Used by `screen_attention_heads.py` and `attention_analysis.py`. Kept private
(underscore prefix) and small — only the parts both scripts need. Mirrors
`RBD_analysis/scripts/_pipeline.py` but wires in the contact-prediction collator
and a couple of head-mode helpers.

Contact analysis has no pair/protein strategy axis (every unit is one PDB
complex), so the only output-routing axis is the head-averaging mode
(all 12 heads vs the screened subset). The two helpers at the bottom handle
that.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from coral.data import DataCollatorForContactPrediction
from coral.model import (
    CoralPretraining,
    CoralConfig,
)


PROTEIN_ENCODER = "facebook/esm2_t30_150M_UR50D"
HEAD_MODES = ("all", "best")
N_HEADS = 12


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) for deterministic inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tokenize(examples, rna_tokenizer, protein_tokenizer,
             max_rna_length=None, max_protein_length=None):
    rna = rna_tokenizer(examples["RNA_seqs"],
                        truncation=bool(max_rna_length),
                        max_length=max_rna_length,
                        padding=False,
                        return_special_tokens_mask=True)
    prot = protein_tokenizer(examples["Prot_seqs"],
                             truncation=bool(max_protein_length),
                             max_length=max_protein_length,
                             padding=False,
                             return_special_tokens_mask=True)
    return {
        "rna_input_ids": rna["input_ids"],
        "rna_attention_mask": rna["attention_mask"],
        "rna_special_tokens_mask": rna["special_tokens_mask"],
        "protein_input_ids": prot["input_ids"],
        "protein_attention_mask": prot["attention_mask"],
        "protein_special_tokens_mask": prot["special_tokens_mask"],
    }


def prepare_dataloader(interactions_df: pd.DataFrame,
                       rna_tokenizer: AutoTokenizer,
                       protein_tokenizer: AutoTokenizer,
                       all_contacts: dict,
                       batch_size: int = 4,
                       max_rna_length: Optional[int] = None,
                       max_protein_length: Optional[int] = None,
                       num_workers: int = 4,
                       device: str = "cuda") -> DataLoader:
    df = interactions_df.copy()
    df["RNA_seqs"] = df["RNA_seqs"].str.replace("U", "T")  # DNABERT expects DNA alphabet
    dataset = HFDataset.from_pandas(df)

    tokenized = dataset.map(
        tokenize, batched=True, num_proc=num_workers,
        remove_columns=["RNA_seqs", "Prot_seqs"],
        fn_kwargs=dict(rna_tokenizer=rna_tokenizer,
                       protein_tokenizer=protein_tokenizer,
                       max_rna_length=max_rna_length,
                       max_protein_length=max_protein_length),
    )

    collator = DataCollatorForContactPrediction(
        rna_tokenizer=rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        all_contacts=all_contacts,
        rna_max_length=max_rna_length,
        protein_max_length=max_protein_length,
    )

    return DataLoader(
        tokenized, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
        pin_memory=device == "cuda",
    )


def load_pretrained_model(checkpoint_dir: str,
                          rna_encoder_checkpoint: str,
                          protein_encoder_checkpoint: str,
                          device: torch.device):
    config = CoralConfig(
        rna_encoder_checkpoint=rna_encoder_checkpoint,
        protein_encoder_checkpoint=protein_encoder_checkpoint,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2,
    )
    print("Loading base encoders...")
    base_model = CoralPretraining(config=config)
    print(f"Loading trained weights from {checkpoint_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    print("Merging LoRA weights...")
    merged = peft_model.merge_and_unload()
    merged.to(device).eval()
    return merged


def filter_interactions(predictions_csv: Path) -> pd.DataFrame:
    """Keep predicted-positive pairs; return the four columns inference needs."""
    df = pd.read_csv(predictions_csv)
    needed = {"RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs", "prediction"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{predictions_csv}: missing columns {missing}")
    df = df.loc[df["prediction"] == 1,
                ["RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs"]]
    return df.reset_index(drop=True)


# --- Head-mode output routing ------------------------------------------------ #
def head_mode_from_selection(selected_heads_path) -> str:
    """'all' when no selected-heads file was provided, 'best' otherwise."""
    return "all" if selected_heads_path is None else "best"


def head_subdir(results_dir: Path, head_mode: str) -> Path:
    """Head-mode output directory: results/<model>/heads_<mode>/ (created)."""
    if head_mode not in HEAD_MODES:
        raise ValueError(f"Unknown head_mode {head_mode!r}; expected one of {HEAD_MODES}.")
    sub = results_dir / f"heads_{head_mode}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


# --- Cross-attention fusion -------------------------------------------------- #
def geometric_mean_fuse(r2p: np.ndarray, p2r: np.ndarray) -> np.ndarray:
    """Geometric-mean fusion of RNA→protein and protein→RNA attention.

    r2p: (..., rna_len, prot_len); p2r: (..., prot_len, rna_len) as returned by
    AttentionAnalyzer for one complex. Returns (..., rna_len, prot_len).
    swapaxes(-1, -2) aligns p2r to r2p's orientation and works for both the 2D
    head-averaged case and the 3D per-head case. Softmax outputs are >= 0, so the
    sqrt is always real.
    """
    return np.sqrt(r2p * np.swapaxes(p2r, -1, -2))
