"""Population-level statistics for the selected cross-attention heads (e.g. [0, 7]).

For every correctly classified RNA-protein complex in the input set,
extract the cross-attention scores of the chosen heads, then run the three
hierarchical tests described in the paper Methods:

  1. One-sample t-test on per-complex (contact - non_contact) differences.
  2. Hierarchical bootstrap (resampling complexes with replacement, 10k iters).
  3. Per-complex Cohen's d distribution.
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
from scipy import stats
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


def prepare_dataloader(interactions_df, rna_tokenizer, protein_tokenizer,
                       all_contacts, batch_size=4, device=torch.device("cpu")):
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
    base = CoralPretraining(config=config)
    merged = PeftModel.from_pretrained(base, weights_dir).merge_and_unload()
    merged.to(device).eval()
    return merged


def analyze_multiple_matrices(attention_matrices_list, contact_labels_list):
    """Per-complex contact-vs-noncontact summary across all input pairs."""
    print("=== Multi-Matrix Attention Analysis ===")
    print(f"Number of matrices: {len(attention_matrices_list)}")

    per_matrix_stats = []
    all_contact_scores, all_non_contact_scores = [], []

    for i, (attention, contact) in enumerate(zip(attention_matrices_list, contact_labels_list)):
        attention_flat = attention.flatten()
        contact_flat = contact.flatten()
        contact_positions = contact_flat == 1
        contact_scores = attention_flat[contact_positions]
        non_contact_scores = attention_flat[~contact_positions]
        all_contact_scores.extend(contact_scores)
        all_non_contact_scores.extend(non_contact_scores)

        if len(contact_scores) > 0:
            stats_dict = {
                "matrix_id": i,
                "n_contact": len(contact_scores),
                "n_non_contact": len(non_contact_scores),
                "contact_mean": np.mean(contact_scores),
                "non_contact_mean": np.mean(non_contact_scores),
                "contact_std": np.std(contact_scores),
                "non_contact_std": np.std(non_contact_scores),
                "imbalance_ratio": len(non_contact_scores) / len(contact_scores),
                "mean_diff": np.mean(contact_scores) - np.mean(non_contact_scores),
            }
            if len(contact_scores) >= 3:
                t_stat, p_val = stats.ttest_ind(contact_scores, non_contact_scores, equal_var=False)
                stats_dict["t_stat"] = t_stat
                stats_dict["p_value"] = p_val
            else:
                stats_dict["t_stat"] = np.nan
                stats_dict["p_value"] = np.nan
            per_matrix_stats.append(stats_dict)

    df_stats = pd.DataFrame(per_matrix_stats)
    print(f"Matrices with contact positions: {len(df_stats)}")
    if len(df_stats) > 0:
        print(f"Average imbalance ratio: {df_stats['imbalance_ratio'].mean():.1f}:1")
    print(f"Total contact positions: {len(all_contact_scores)}")
    print(f"Total non-contact positions: {len(all_non_contact_scores)}\n")

    return df_stats, all_contact_scores, all_non_contact_scores


def meta_analysis(df_stats):
    """Per-complex one-sample t-test and sign test on mean-difference values."""
    valid_stats = df_stats.dropna(subset=["t_stat", "p_value"])
    if len(valid_stats) == 0:
        print("No matrices with sufficient data for individual t-tests")
        return None

    mean_diffs = valid_stats["mean_diff"].values
    t_meta, p_meta = stats.ttest_1samp(mean_diffs, 0, alternative="greater")
    positive_diffs = int(np.sum(mean_diffs > 0))
    sign_p = stats.binomtest(positive_diffs, len(mean_diffs), 0.5, alternative="greater").pvalue

    print(f"Matrices analyzed: {len(valid_stats)}")
    print(f"Mean of mean differences: {np.mean(mean_diffs):.5f}, std: {np.std(mean_diffs):.5f}")
    print(f"Meta t-test (mean diff vs 0): t={t_meta:.4f}, p={p_meta:.2e}")
    print(f"Matrices with contact > non-contact: {positive_diffs}/{len(mean_diffs)}")
    print(f"Sign test p-value: {sign_p:.2e}\n")

    return {
        "n_matrices": len(valid_stats),
        "mean_diff_overall": np.mean(mean_diffs),
        "meta_t": t_meta,
        "meta_p": p_meta,
        "positive_matrices": positive_diffs,
        "sign_test_p": sign_p,
    }


def hierarchical_bootstrap_analysis(df_stats, attention_matrices, contact_maps, n_bootstrap=10000):
    """Resample complexes with replacement; preserve within-complex dependence."""
    n_complexes = len(attention_matrices)
    observed_per_matrix_diffs = df_stats["mean_diff"].to_numpy()
    observed_mean_diff = float(np.mean(observed_per_matrix_diffs))
    print(f"Number of valid complexes: {len(observed_per_matrix_diffs)}")
    print(f"Observed mean difference across complexes: {observed_mean_diff:.6f}\n")

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        resampled_indices = np.random.choice(n_complexes, n_complexes, replace=True)
        per_matrix = []
        for idx in resampled_indices:
            attn, contact = attention_matrices[idx], contact_maps[idx]
            contact_scores = attn[contact == 1]
            non_contact_scores = attn[contact == 0]
            if len(contact_scores) == 0 or len(non_contact_scores) == 0:
                continue
            per_matrix.append(np.mean(contact_scores) - np.mean(non_contact_scores))
        bootstrap_diffs.append(np.mean(per_matrix))

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_95 = np.percentile(bootstrap_diffs, [2.5, 97.5])
    ci_99 = np.percentile(bootstrap_diffs, [0.5, 99.5])
    p_left = np.mean(bootstrap_diffs <= 0)
    p_right = np.mean(bootstrap_diffs >= 0)
    bootstrap_p = 2 * min(p_left, p_right)

    print(f"Observed mean difference: {observed_mean_diff}")
    print(f"Bootstrap mean: {np.mean(bootstrap_diffs)}")
    print(f"Bootstrap SD: {np.std(bootstrap_diffs)}")
    print(f"95% CI: [{ci_95[0]}, {ci_95[1]}]")
    print(f"99% CI: [{ci_99[0]}, {ci_99[1]}]")
    print(f"Bootstrap p-value: {bootstrap_p}")

    return {
        "observed_mean_diff": observed_mean_diff,
        "observed_per_matrix_diffs": observed_per_matrix_diffs,
        "bootstrap_diffs": bootstrap_diffs,
        "bootstrap_mean": float(np.mean(bootstrap_diffs)),
        "bootstrap_sd": float(np.std(bootstrap_diffs)),
        "ci_95": ci_95,
        "ci_99": ci_99,
        "bootstrap_p": bootstrap_p,
    }


def effect_size_analysis(df_stats):
    """Per-complex Cohen's d distribution."""
    valid_stats = df_stats.dropna(subset=["t_stat", "p_value"])
    effect_sizes = []
    for _, row in valid_stats.iterrows():
        pooled_std = np.sqrt((row["contact_std"] ** 2 + row["non_contact_std"] ** 2) / 2)
        if pooled_std > 0:
            effect_sizes.append(row["mean_diff"] / pooled_std)

    if effect_sizes:
        print(f"Mean effect size (Cohen's d): {np.mean(effect_sizes):.4f}, "
              f"std: {np.std(effect_sizes):.4f}")
        print(f"Median effect size: {np.median(effect_sizes):.4f}")
        print(f"Matrices with positive effect: {int(np.sum(np.array(effect_sizes) > 0))}/{len(effect_sizes)}\n")

    return {
        "effect_sizes": effect_sizes,
        "mean_effect": float(np.mean(effect_sizes)) if effect_sizes else 0.0,
        "median_effect": float(np.median(effect_sizes)) if effect_sizes else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interactions-csv", default="data/pdb/pdb_predictednew.csv")
    p.add_argument("--interfaces-json", default="data/pdb/multi_chain_interaction_interfaces.json")
    p.add_argument("--weights", default="weights/coral-epoch-9")
    p.add_argument("--output-stats", default="attention_stats.csv")
    p.add_argument("--heads", default="0,7",
                   help="Comma-separated list of cross-attention heads to average (default: 0,7)")
    p.add_argument("--rna-encoder", default="./DNABERT-2-117M")
    p.add_argument("--protein-encoder", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    subset = [int(h) for h in args.heads.split(",") if h.strip() != ""]
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
    rna2protein_pair_ids = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        rna_batch = {"input_ids": batch["rna_input_ids"].to(device),
                     "attention_mask": batch["rna_attention_mask"].to(device)}
        protein_batch = {"input_ids": batch["protein_input_ids"].to(device),
                         "attention_mask": batch["protein_attention_mask"].to(device)}
        contact_labels = batch["contact_labels"].to(device)

        out = analyzer.extract_cross_attention_scores(
            rna_batch, protein_batch, contact_labels, subset=subset
        )
        for i in range(len(out["rna_to_protein_attentions"])):
            if out["rna2protein_contact_labels"][i].sum() >= 3:
                all_rna_to_protein_atts.append(out["rna_to_protein_attentions"][i])
                all_contact_labels.append(out["rna2protein_contact_labels"][i])
                rna2protein_pair_ids.append(batch["contact_keys"][i])

    attention_stats, all_contact_scores, all_non_contact_scores = analyze_multiple_matrices(
        all_rna_to_protein_atts, all_contact_labels
    )

    print(f"contact map shape: {all_contact_labels[0].shape}")
    print(f"attention scores matrix shape: {all_rna_to_protein_atts[0].shape}")
    print(f"pooled contact scores mean: {np.mean(all_contact_scores)}")
    print(f"pooled non-contact scores mean: {np.mean(all_non_contact_scores)}")

    valid_stats = attention_stats.dropna(subset=["t_stat", "p_value"]).copy()
    valid_stats["complex_ids"] = valid_stats["matrix_id"].apply(lambda i: rna2protein_pair_ids[i])
    valid_stats.to_csv(args.output_stats, index=False)
    print(f"Per-complex stats written to {args.output_stats}\n")

    print("=== Approach 2: Meta-Analysis Across Matrices ===")
    meta_analysis(attention_stats)
    print("=== Approach 3: Hierarchical Bootstrap Analysis ===")
    hierarchical_bootstrap_analysis(
        attention_stats, all_rna_to_protein_atts, all_contact_labels, n_bootstrap=args.n_bootstrap
    )
    print("=== Approach 4: Effect Size Analysis ===")
    effect_size_analysis(attention_stats)


if __name__ == "__main__":
    main()
