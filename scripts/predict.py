"""Score a CSV of RNA-protein pairs with trained CORAL weights."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coral import paths
from coral.inference import predict_interactions
from coral.model import CoralConfig, CoralPretraining


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True,
                   help="CSV with columns: RNA_id, Protein_id, RNA_seqs, Prot_seqs (labels optional)")
    p.add_argument("--weights", default=str(paths.checkpoint(9)),
                   help="Path to the LoRA checkpoint directory")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--rna-encoder", default=str(paths.DNABERT_DIR),
                   help="DNABERT-2 local clone path")
    p.add_argument("--protein-encoder", default=paths.PROTEIN_ENCODER,
                   help="ESM-2 checkpoint (HF hub id or local path)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-rna-length", type=int, default=6000)
    p.add_argument("--max-protein-length", type=int, default=1250)
    p.add_argument("--save-attentions", action="store_true",
                   help="Also save a .npz of per-pair cross-attention arrays next to --output")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    pairs_df = pd.read_csv(args.input_csv)

    rna_tokenizer = AutoTokenizer.from_pretrained(args.rna_encoder, trust_remote_code=True)
    protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_encoder, trust_remote_code=True)

    config = CoralConfig(
        rna_encoder_checkpoint=args.rna_encoder,
        protein_encoder_checkpoint=args.protein_encoder,
    )

    print("Loading base encoders...")
    base_model = CoralPretraining(config=config)

    print(f"Loading trained weights from {args.weights}...")
    peft_model = PeftModel.from_pretrained(base_model, args.weights)

    merged_model = peft_model.merge_and_unload()
    merged_model.to(args.device).eval()

    prediction = predict_interactions(
        merged_model,
        pairs_df,
        rna_tokenizer,
        protein_tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        max_rna_length=args.max_rna_length,
        max_protein_length=args.max_protein_length,
        save_attentions=args.save_attentions,
    )

    if args.save_attentions:
        results_df, attentions = prediction
        attn_path = Path(args.output).with_suffix(".attentions.npz")
        np.savez_compressed(attn_path, **attentions)
        print(f"Attentions saved to {attn_path}")
    else:
        results_df = prediction

    results_df.to_csv(args.output, index=False)
    n_pos = int((results_df["prediction"] == 1).sum())
    print(f"Predictions saved to {args.output}  ({len(results_df)} pairs, {n_pos} predicted positive)")


if __name__ == "__main__":
    main()
