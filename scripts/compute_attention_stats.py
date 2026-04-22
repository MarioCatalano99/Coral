"""Extract per-head cross-attention statistics on the PDB complexes.

Writes `attention_heads_stats.csv` with per-complex, per-head contact / non-contact
attention means and the contact - non_contact difference. Used downstream by
`scripts/select_attention_heads.py` and `scripts/analyze_attention.py`.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coral.attention import AttentionAnalyzer
from coral.data import DataCollatorForContactPrediction
from coral.model import CoralConfig, CoralPretraining


def tokenize(examples, rna_tokenizer, protein_tokenizer,
             max_rna_length: Optional[int] = None, max_protein_length: Optional[int] = None):
    rna_tokenized = rna_tokenizer(
        examples["RNA_seqs"], truncation=bool(max_rna_length),
        max_length=max_rna_length, padding=False, return_special_tokens_mask=True,
    )
    protein_tokenized = protein_tokenizer(
        examples["Prot_seqs"], truncation=bool(max_protein_length),
        max_length=max_protein_length, padding=False, return_special_tokens_mask=True,
    )
    return {
        "rna_input_ids": rna_tokenized["input_ids"],
        "rna_attention_mask": rna_tokenized["attention_mask"],
        "rna_special_tokens_mask": rna_tokenized["special_tokens_mask"],
        "protein_input_ids": protein_tokenized["input_ids"],
        "protein_attention_mask": protein_tokenized["attention_mask"],
        "protein_special_tokens_mask": protein_tokenized["special_tokens_mask"],
    }


def prepare_dataloader(interactions_df, rna_tokenizer, protein_tokenizer, all_contacts,
                       batch_size=4, device=torch.device("cuda")):
    required_cols = ["RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs"]
    missing = [c for c in required_cols if c not in interactions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    interactions_df["RNA_seqs"] = interactions_df["RNA_seqs"].str.replace("U", "T")
    dataset = HFDataset.from_pandas(interactions_df)

    tokenized = dataset.map(
        tokenize, batched=True, num_proc=4,
        remove_columns=["RNA_seqs", "Prot_seqs"],
        fn_kwargs={"rna_tokenizer": rna_tokenizer, "protein_tokenizer": protein_tokenizer},
    )

    collator = DataCollatorForContactPrediction(
        rna_tokenizer=rna_tokenizer, protein_tokenizer=protein_tokenizer, all_contacts=all_contacts,
    )
    return DataLoader(
        tokenized, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=device.type == "cuda",
    )


def load_model(weights_dir, rna_encoder, protein_encoder, device):
    config = CoralConfig(
        rna_encoder_checkpoint=rna_encoder,
        protein_encoder_checkpoint=protein_encoder,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2,
    )
    base_model = CoralPretraining(config=config)
    peft_model = PeftModel.from_pretrained(base_model, weights_dir)
    merged = peft_model.merge_and_unload()
    merged.to(device).eval()
    return merged


def analyze_heads(heads, contact_labels, matrix_id):
    """Per-head contact-vs-noncontact summary for a single complex."""
    n_heads = heads.shape[0]

    attention_flat = heads.reshape(heads.shape[0], -1)
    contact_flat = contact_labels.flatten()

    contact_positions = contact_flat == 1
    contact_scores = attention_flat[:, contact_positions]
    non_contact_scores = attention_flat[:, ~contact_positions]

    if contact_scores.shape[1] < 3:
        return None

    return pd.DataFrame(
        {
            "matrix_id": [matrix_id] * n_heads,
            "head_id": [h + 1 for h in range(n_heads)],
            "n_contact": [len(contact_scores)] * n_heads,
            "n_non_contact": [len(non_contact_scores)] * n_heads,
            "contact_mean": contact_scores.mean(1).tolist(),
            "non_contact_mean": non_contact_scores.mean(1).tolist(),
            "contact_std": contact_scores.std(1).tolist(),
            "non_contact_std": non_contact_scores.std(1).tolist(),
            "imbalance_ratio": [len(non_contact_scores) / len(contact_scores)] * n_heads,
            "mean_diff": (contact_scores.mean(1) - non_contact_scores.mean(1)).tolist(),
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interactions-csv", default="data/pdb/pdb_predictednew.csv",
                   help="CSV of PDB-derived RNA-protein pairs")
    p.add_argument("--interfaces-json", default="data/pdb/multi_chain_interaction_interfaces.json",
                   help="JSON of contact annotations, keyed by '<RNA_id>_<Protein_id[-1]>'")
    p.add_argument("--weights", default="weights/coral-epoch-9")
    p.add_argument("--output", default="attention_heads_stats.csv")
    p.add_argument("--rna-encoder", default="./DNABERT-2-117M")
    p.add_argument("--protein-encoder", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    df = pd.read_csv(args.interactions_csv)
    df = df.loc[df.prediction == 1, ["RNA_id", "Protein_id", "RNA_seqs", "Prot_seqs"]]

    with open(args.interfaces_json) as f:
        all_contacts = json.load(f)

    rna_tokenizer = AutoTokenizer.from_pretrained(args.rna_encoder, trust_remote_code=True)
    protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_encoder, trust_remote_code=True)

    dataloader = prepare_dataloader(df, rna_tokenizer, protein_tokenizer, all_contacts,
                                    batch_size=args.batch_size, device=device)

    model = load_model(args.weights, args.rna_encoder, args.protein_encoder, device)
    analyzer = AttentionAnalyzer(model, rna_tokenizer, protein_tokenizer, device=device)

    all_rna_to_protein_atts, all_contact_labels = [], []
    for batch in tqdm(dataloader, total=len(dataloader)):
        rna_batch = {"input_ids": batch["rna_input_ids"].to(device),
                     "attention_mask": batch["rna_attention_mask"].to(device)}
        protein_batch = {"input_ids": batch["protein_input_ids"].to(device),
                         "attention_mask": batch["protein_attention_mask"].to(device)}
        contact_labels = batch["contact_labels"].to(device)

        out = analyzer.extract_cross_attention_scores(rna_batch, protein_batch, contact_labels)
        all_rna_to_protein_atts.extend(out["rna_to_protein_attentions"])
        all_contact_labels.extend(out["rna2protein_contact_labels"])

    rows = []
    for i, (att, contacts) in enumerate(zip(all_rna_to_protein_atts, all_contact_labels), start=1):
        stats_df = analyze_heads(att, contacts, matrix_id=i)
        if stats_df is not None:
            rows.append(stats_df)

    print(f"Valid matrices: {len(rows)}")
    pd.concat(rows, axis=0).to_csv(args.output, index=False)
    print(f"Wrote per-head statistics to {args.output}")


if __name__ == "__main__":
    main()
