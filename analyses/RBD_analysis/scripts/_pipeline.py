"""Shared inference scaffolding for RBD_analysis scripts.

Used by `screen_attention_heads.py` and `attention_analysis.py`. Kept
private (underscore prefix) and small — only the parts both scripts need.

Inference is always per-pair; the unit of analysis is the *protein*, so
`aggregate_to_protein` averages a protein's per-pair vectors into one consensus
profile. The head-mode helpers at the bottom route output to heads_{all,best}/.
"""

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from coral.data import DataCollatorForRBDPrediction
from coral.model import (
    CoralPretraining,
    CoralConfig,
)


PROTEIN_ENCODER = "facebook/esm2_t30_150M_UR50D"
HEAD_MODES = ("all", "best")
N_HEADS = 12
VECTORS_FILENAME = "rbd_protein_vectors.npz"


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
                       all_rbd_intervals: dict,
                       batch_size: int = 4,
                       max_rna_length: Optional[int] = None,
                       max_protein_length: Optional[int] = None,
                       num_workers: int = 4,
                       device: str = "cuda") -> DataLoader:
    df = interactions_df.copy()
    df["RNA_seqs"] = df["RNA_seqs"].str.replace("U", "T")
    dataset = HFDataset.from_pandas(df)

    tokenized = dataset.map(
        tokenize, batched=True, num_proc=num_workers,
        remove_columns=["RNA_seqs", "Prot_seqs"],
        fn_kwargs=dict(rna_tokenizer=rna_tokenizer,
                       protein_tokenizer=protein_tokenizer,
                       max_rna_length=max_rna_length,
                       max_protein_length=max_protein_length),
    )

    collator = DataCollatorForRBDPrediction(
        rna_tokenizer=rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        all_rbd_intervals=all_rbd_intervals,
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


def filter_pairs(predictions_csv: Path,
                 rbd_intervals: dict,
                 protein_summary: pd.DataFrame) -> pd.DataFrame:
    """Keep true-positive pairs whose protein passes Step-2 filters and has ≥1 RBD."""
    df = pd.read_csv(predictions_csv)
    needed = {"RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs", "prediction"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{predictions_csv}: missing columns {missing}")

    keep = set(protein_summary
               .loc[protein_summary["passes_filters"] == 1, "protein_id"])
    keep &= set(rbd_intervals.keys())
    df = df.loc[(df["prediction"] == 1) & df["Protein_id"].isin(keep)]
    return df.loc[:, ["RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs"]].reset_index(drop=True)


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


# --- Pair -> protein aggregation --------------------------------------------- #
def aggregate_to_protein(
    attentions: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    protein_ids: Sequence[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[int]]:
    """Average per-pair attention vectors element-wise within each protein.

    Works for both the 1D per-pair vector (L_protein,) used by the test and the
    2D per-head array (n_heads, L_protein) used by head screening — the mean is
    taken over the pair axis either way. The mask is a function of `protein_id`
    only, so all pairs sharing a protein must produce the same mask — this is
    asserted to fail loud if tokenization/truncation drift is ever introduced.

    Returns
    -------
    protein_attentions : list of np.ndarray, one consensus profile per protein
    protein_masks      : list of np.ndarray, the per-protein RBD mask
    ordered_protein_ids: list of protein_id strings, in insertion order
    n_partners         : list of ints, number of TP partners per protein
    """
    if not (len(attentions) == len(masks) == len(protein_ids)):
        raise ValueError("attentions, masks, and protein_ids must have the same length.")

    # --- Group indices by protein_id, preserving first-seen order ---
    groups: dict[str, list[int]] = {}
    for idx, pid in enumerate(protein_ids):
        groups.setdefault(pid, []).append(idx)

    protein_attentions: List[np.ndarray] = []
    protein_masks: List[np.ndarray] = []
    ordered_pids: List[str] = []
    n_partners: List[int] = []

    for pid, idxs in groups.items():
        ref_mask = masks[idxs[0]]
        ref_attn_len = attentions[idxs[0]].shape[-1]

        # --- Sanity: every partner must agree on protein length and RBD mask ---
        vecs = []
        for j in idxs:
            attn = attentions[j]
            if attn.shape[-1] != ref_attn_len:
                raise ValueError(
                    f"Protein {pid}: attention vector length mismatch across partners "
                    f"({attn.shape[-1]} vs {ref_attn_len}). Tokenization/truncation drift?"
                )
            if not np.array_equal(masks[j], ref_mask):
                raise ValueError(
                    f"Protein {pid}: RBD mask mismatch across partners. "
                    f"Tokenization/truncation drift?"
                )
            vecs.append(attn)

        protein_attentions.append(np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32))
        protein_masks.append(ref_mask.astype(np.int8))
        ordered_pids.append(pid)
        n_partners.append(len(idxs))

    return protein_attentions, protein_masks, ordered_pids, n_partners


def drop_low_partner_proteins(attentions, masks, protein_ids, n_partners,
                              min_partners: int):
    """Filter out proteins backed by fewer than `min_partners` TP partners."""
    if min_partners <= 1:
        return attentions, masks, protein_ids, n_partners
    keep = [i for i, k in enumerate(n_partners) if k >= min_partners]
    dropped = len(protein_ids) - len(keep)
    if dropped:
        print(f"Dropping {dropped} proteins with < {min_partners} TP partners.")
    return ([attentions[i] for i in keep], [masks[i] for i in keep],
            [protein_ids[i] for i in keep], [n_partners[i] for i in keep])
