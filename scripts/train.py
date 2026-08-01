"""Train CORAL from scratch on a single train/val fold.

Reference implementation of the protocol described in the paper Methods.
Retraining is hardware-dependent (expect ~24-40 GB VRAM on the longest
sequences); the released weights at `checkpoints/coral-epoch-9/` are the
recommended starting point for inference and reproduction of Table 2.

Known seed --> fold-size combinations used in the paper:
  # componentwise, 60/80: [256, 294, 2161, 21624, 74750]
  # componentwise, 30/50: [460, 4708, 5608, 43483, 55152]
  # pairwise,      60/80: [628, 4021, 11471, 14300, 53128]
  # pairwise,      30/50: [558, 5169, 5765, 8990, 24160]
  # redundant:          [1596, 3587, 9070, 29690, 77770]
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

# --- Path setup so the script runs from the repo root without install ------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coral import paths
from coral.data import CoralDataCollator, seed_worker, specificity_score, tokenize_function
from coral.model import CoralConfig, CoralPretraining


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True,
                   help="Dataset split name, e.g. componentwise_splits_60_80, pairwise_splits_30_50, redundant_split")
    p.add_argument("--fold", type=int, required=True,
                   help="Fold random-seed identifier (subdirectory name under the split)")
    p.add_argument("--data-root", default=str(paths.DATASETS_DIR),
                   help="Root directory containing split subdirectories")
    p.add_argument("--output-dir", default=str(paths.REPO_ROOT / "results"),
                   help="Output directory for metrics; created if absent")
    p.add_argument("--rna-encoder", default=str(paths.DNABERT_DIR),
                   help="DNABERT-2 checkpoint path (local clone is required)")
    p.add_argument("--protein-encoder", default=paths.PROTEIN_ENCODER,
                   help="ESM-2 checkpoint (HF hub id or local path)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup-pct", type=float, default=0.05,
                   help="Fraction of total training steps used for linear warmup")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank; 16 is what the released checkpoint was trained with")
    p.add_argument("--mlm-scaling", type=float, default=1.0)
    p.add_argument("--num-threads", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)

    result_path = Path(args.output_dir) / args.split / str(args.fold)
    result_path.mkdir(parents=True, exist_ok=True)

    fold_dir = Path(args.data_root) / args.split / f"fold_{args.fold}"
    train_df = pd.read_csv(fold_dir / "train.csv")[["RNA_seqs", "Prot_seqs", "labels"]]
    val_df = pd.read_csv(fold_dir / "val.csv")[["RNA_seqs", "Prot_seqs", "labels"]]

    raw_datasets = DatasetDict(
        {"train": Dataset.from_pandas(train_df), "validation": Dataset.from_pandas(val_df)}
    )

    rna_tokenizer = AutoTokenizer.from_pretrained(args.rna_encoder, trust_remote_code=True)
    protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_encoder, trust_remote_code=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_threads,
        remove_columns=["RNA_seqs", "Prot_seqs"],
        fn_kwargs={"rna_tokenizer": rna_tokenizer, "protein_tokenizer": protein_tokenizer},
    )

    data_collator = CoralDataCollator(
        rna_tokenizer=rna_tokenizer, protein_tokenizer=protein_tokenizer, mlm_probability=0.15
    )

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print(f"number of train batches: {len(train_dataloader)}")
    print(f"number of val batches: {len(val_dataloader)}")

    model_config = CoralConfig(
        rna_encoder_checkpoint=args.rna_encoder,
        protein_encoder_checkpoint=args.protein_encoder,
    )
    model = CoralPretraining(config=model_config).to(device)

    # --- LoRA target selection: encoders are adapted, head/cross-encoder are saved ---
    target_modules, modules_to_save = [], []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            if name.startswith("bert.protein_encoder") or name.startswith("bert.rna_encoder"):
                target_modules.append(name)
            elif any(name.startswith(p) for p in ("bert.cross_encoder", "cls", "bert.projection",
                                                  "bert.pooler", "reverse_projection")):
                modules_to_save.append(name)

    lora_config = LoraConfig(r=args.lora_rank, target_modules=target_modules, modules_to_save=modules_to_save)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    optimizer = AdamW(peft_model.parameters(), lr=args.lr)

    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = int(args.warmup_pct * num_training_steps)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(args.epochs):
        # --- Training loop ---
        peft_model.train()
        train_loss_sum = 0.0
        train_preds, train_labels = [], []
        for batch in train_dataloader:
            matched_labels = batch["labels"].to(device)
            rna_batch = {
                "input_ids": batch["rna_input_ids"].to(device),
                "attention_mask": batch["rna_attention_mask"].to(device),
                "labels_for_MLM": batch["rna_labels_for_MLM"].to(device),
            }
            protein_batch = {
                "input_ids": batch["protein_input_ids"].to(device),
                "attention_mask": batch["protein_attention_mask"].to(device),
                "labels_for_MLM": batch["protein_labels_for_MLM"].to(device),
            }

            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                loss, _, predictions, _ = peft_model(
                    rna_batch, protein_batch, matched_labels,
                    task_mask_lm=True, mlm_scaling_factor=args.mlm_scaling,
                )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss_sum += loss.item()
            train_preds.extend(predictions.cpu().tolist())
            train_labels.extend(matched_labels.cpu().tolist())
            progress_bar.update(1)

        # --- Validation loop ---
        peft_model.eval()
        val_preds, val_labels = [], []
        for batch in val_dataloader:
            matched_labels = batch["labels"].to(device)
            rna_batch = {
                "input_ids": batch["rna_input_ids"].to(device),
                "attention_mask": batch["rna_attention_mask"].to(device),
                "labels_for_MLM": batch["rna_labels_for_MLM"].to(device),
            }
            protein_batch = {
                "input_ids": batch["protein_input_ids"].to(device),
                "attention_mask": batch["protein_attention_mask"].to(device),
                "labels_for_MLM": batch["protein_labels_for_MLM"].to(device),
            }
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, predictions, _ = peft_model(
                    rna_batch, protein_batch, matched_labels,
                    task_mask_lm=True, mlm_scaling_factor=args.mlm_scaling,
                )
            val_preds.extend(predictions.cpu().tolist())
            val_labels.extend(matched_labels.cpu().tolist())

        # --- Metrics ---
        for phase, y_true, y_pred in [("train", train_labels, train_preds), ("val", val_labels, val_preds)]:
            metrics_path = result_path / f"{phase}_metrics.csv"
            if not metrics_path.exists():
                metrics_path.write_text("Accuracy,F1,MCC,Recall,Precision,Specificity,Fold,Epoch\n")
            with metrics_path.open("a") as f:
                f.write(
                    f"{accuracy_score(y_true, y_pred)},{f1_score(y_true, y_pred)},"
                    f"{matthews_corrcoef(y_true, y_pred)},{recall_score(y_true, y_pred)},"
                    f"{precision_score(y_true, y_pred)},{specificity_score(y_true, y_pred)},"
                    f"{args.fold},{epoch}\n"
                )


if __name__ == "__main__":
    main()
