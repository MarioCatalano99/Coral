#!/usr/bin/env python3
"""
Contact analysis — attention-vs-contact statistical test.

For each predicted-positive PDB complex with ≥3 residue contacts, run inference
with the chosen cross-attention heads (last layer, averaged), then per complex
compute

    Δ = mean(attn | contact) − mean(attn | non-contact)

with a Welch's t-test over the flattened RNA×protein attention matrix. Aggregate
across complexes via one-sample t (alt='greater'), a sign test, a hierarchical
bootstrap, and per-complex Cohen's d (the last two via `coral.stats`).

Head mode is set by `--selected-heads`:
    omitted              → average across all 12 heads → results/<model>/heads_all/
    selected_heads.json  → average across the screened subset → results/<model>/heads_best/

Outputs (under results/<model>/heads_{all,best}/):
    attention_stats.csv   one row per complex
    contact_vectors.npz   cached attention + contact-map matrices (object arrays);
                          lets the bootstrap/controls re-run without GPU
    meta_analysis.txt     captured stdout
"""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))  # no-op once `pip install -e .` has been run
from coral import paths
from coral.attention import AttentionAnalyzer
from coral.stats import cohens_d_per_unit, hierarchical_bootstrap

import _pipeline as pipe  # sibling module, on sys.path[0] when run as a script


def per_matrix_stat(attn: np.ndarray, contact: np.ndarray) -> Optional[dict]:
    """Welch's t-test of attention at contact vs non-contact positions.

    attn, contact: aligned (rna_len, prot_len) arrays; contact is 0/1.
    Returns None if the complex has <3 contact or <3 non-contact positions.
    """
    attn_flat = attn.reshape(-1)
    contact_flat = contact.reshape(-1)
    contact_scores = attn_flat[contact_flat == 1]
    non_contact_scores = attn_flat[contact_flat != 1]
    if contact_scores.size < 3 or non_contact_scores.size < 3:
        return None
    t_stat, p_val = stats.ttest_ind(contact_scores, non_contact_scores, equal_var=False)
    return {
        "n_contact": int(contact_scores.size),
        "n_non_contact": int(non_contact_scores.size),
        "contact_mean": float(contact_scores.mean()),
        "non_contact_mean": float(non_contact_scores.mean()),
        "contact_std": float(contact_scores.std()),
        "non_contact_std": float(non_contact_scores.std()),
        "imbalance_ratio": float(non_contact_scores.size / contact_scores.size),
        "mean_diff": float(contact_scores.mean() - non_contact_scores.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }


def collect_matrices(predictions_csv: Path,
                     interfaces: dict,
                     selected_heads: List[int],
                     checkpoint_dir: str,
                     device: torch.device,
                     batch_size: int,
                     rna_max_length: int,
                     protein_max_length: int):
    """Inference loop. Returns (attentions, contact_maps, complex_ids), each a
    list aligned by complex; only complexes with ≥3 contacts are kept."""
    interactions = pipe.filter_interactions(predictions_csv)
    print(f"Predicted-positive pairs: {len(interactions)}")

    rna_tok = AutoTokenizer.from_pretrained(str(paths.DNABERT_DIR), trust_remote_code=True)
    prot_tok = AutoTokenizer.from_pretrained(pipe.PROTEIN_ENCODER, trust_remote_code=True)

    dataloader = pipe.prepare_dataloader(
        interactions, rna_tok, prot_tok, interfaces,
        batch_size=batch_size,
        max_rna_length=rna_max_length,
        max_protein_length=protein_max_length,
        device=str(device),
    )
    model = pipe.load_pretrained_model(checkpoint_dir, str(paths.DNABERT_DIR),
                                       pipe.PROTEIN_ENCODER, device)
    analyzer = AttentionAnalyzer(model, rna_tok, prot_tok, device=device)

    attentions: List[np.ndarray] = []
    contact_maps: List[np.ndarray] = []
    complex_ids: List[str] = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        rna_in = {"input_ids": batch["rna_input_ids"].to(device),
                  "attention_mask": batch["rna_attention_mask"].to(device)}
        prot_in = {"input_ids": batch["protein_input_ids"].to(device),
                   "attention_mask": batch["protein_attention_mask"].to(device)}
        contact_labels = batch["contact_labels"].to(device)

        # subset=selected_heads → those heads averaged → 2D (rna_len, prot_len).
        out = analyzer.extract_cross_attention_scores(
            rna_in, prot_in, contact_labels, subset=selected_heads)
        # Geometric-mean fusion of both directions, applied to the head-averaged
        # matrices: sqrt(avg_S r2p * (avg_S p2r).T).
        r2p = out["rna_to_protein_attentions"]   # each (rna_len, prot_len)
        p2r = out["protein_to_rna_attentions"]   # each (prot_len, rna_len)
        attns = [pipe.geometric_mean_fuse(r, p) for r, p in zip(r2p, p2r)]
        contacts = out["rna2protein_contact_labels"]
        keys = batch["contact_keys"]

        for i in range(len(attns)):
            if (contacts[i] == 1).sum() < 3:
                continue
            attentions.append(attns[i])
            contact_maps.append(contacts[i])
            complex_ids.append(keys[i])

    return attentions, contact_maps, complex_ids


def build_stats_df(attentions: List[np.ndarray],
                   contact_maps: List[np.ndarray],
                   complex_ids: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for i, (attn, contact) in enumerate(zip(attentions, contact_maps)):
        stat = per_matrix_stat(attn, contact)
        if stat is None:
            continue
        stat["matrix_id"] = i
        stat["complex_id"] = complex_ids[i]
        rows.append(stat)
    cols = ["matrix_id", "complex_id", "n_contact", "n_non_contact",
            "contact_mean", "non_contact_mean", "contact_std", "non_contact_std",
            "imbalance_ratio", "mean_diff", "t_stat", "p_value"]
    return pd.DataFrame(rows, columns=cols)


def meta_analysis(stats_df: pd.DataFrame,
                  attentions: List[np.ndarray],
                  contact_maps: List[np.ndarray],
                  heads: List[int]) -> None:
    """One-sample t / sign test / hierarchical bootstrap / Cohen's d to stdout."""
    diffs = stats_df["mean_diff"].to_numpy()
    n = len(diffs)
    print(f"=== Contact attention meta-analysis ({n} complexes, heads={heads}) ===")
    print(f"Pooled contact mean    : {stats_df['contact_mean'].mean():.6e}")
    print(f"Pooled non-contact mean: {stats_df['non_contact_mean'].mean():.6e}")

    t_meta, p_meta = stats.ttest_1samp(diffs, 0, alternative="greater")
    print(f"\nOne-sample t-test (greater): t={t_meta:.4f}, p={p_meta:.3e}")
    print(f"  mean(Δ) = {diffs.mean():.6e}, std(Δ) = {diffs.std():.6e}")

    n_pos = int((diffs > 0).sum())
    sign_p = stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue
    print(f"Sign test: complexes with Δ>0: {n_pos}/{n}, p={sign_p:.3e}")

    print("\n=== Hierarchical bootstrap ===")
    boot = hierarchical_bootstrap(attentions, contact_maps, n_bootstrap=10000,
                                  rng=np.random.default_rng(42))
    print(f"Observed Δ̄: {boot['observed_mean_diff']:.6e}")
    print(f"Bootstrap mean: {boot['bootstrap_mean']:.6e}, sd: {boot['bootstrap_sd']:.6e}")
    print(f"95% CI: [{boot['ci_95'][0]:.6e}, {boot['ci_95'][1]:.6e}]")
    print(f"99% CI: [{boot['ci_99'][0]:.6e}, {boot['ci_99'][1]:.6e}]")
    print(f"Bootstrap two-sided p: {boot['p_two_sided']:.3e}")

    print("\n=== Per-complex Cohen's d ===")
    ds = cohens_d_per_unit(attentions, contact_maps)
    if ds:
        arr = np.asarray(ds)
        print(f"n = {len(arr)}, mean = {arr.mean():.4f}, "
              f"median = {float(np.median(arr)):.4f}, std = {arr.std():.4f}")
        print(f"Complexes with d > 0: {int((arr > 0).sum())}/{len(arr)}")


def save_contact_vectors(out_path: Path,
                         complex_ids: List[str],
                         attentions: List[np.ndarray],
                         contact_maps: List[np.ndarray]) -> None:
    np.savez_compressed(
        out_path,
        complex_ids=np.array(complex_ids),
        attentions=np.array(attentions, dtype=object),
        contact_maps=np.array(contact_maps, dtype=object),
    )
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path,
                    default=paths.PREDICTIONS_DIR / "pdb_predicted_interactions.csv",
                    help="CSV with [RNA_id, Protein_id, RNA_seqs, Prot_seqs, prediction]")
    ap.add_argument("--interfaces", type=Path,
                    default=paths.PDB_DIR / "multi_chain_interaction_interfaces.json",
                    help="JSON of residue contacts keyed by {RNA_id}_{chain}.")
    ap.add_argument("--selected-heads", type=Path, default=None,
                    help="Path to selected_heads.json. If omitted, average across all 12 heads.")
    ap.add_argument("--epoch", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rna-max-length", type=int, default=None)
    ap.add_argument("--protein-max-length", type=int, default=None)
    args = ap.parse_args()

    pipe.seed_everything(42)

    results_dir = paths.analysis_results("contact_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve heads + head_mode → output subdir ---
    head_mode = pipe.head_mode_from_selection(args.selected_heads)
    out_dir = pipe.head_subdir(results_dir, head_mode)
    if args.selected_heads is None:
        selected_heads = list(range(pipe.N_HEADS))
        print(f"No --selected-heads provided; averaging across all {pipe.N_HEADS} heads "
              f"(head_mode={head_mode}).")
    else:
        if not args.selected_heads.exists():
            raise FileNotFoundError(f"{args.selected_heads} not found.")
        selected_heads = json.loads(args.selected_heads.read_text())["selected"]
        print(f"Using selected heads from {args.selected_heads}: {selected_heads} "
              f"(head_mode={head_mode}).")

    with open(args.interfaces) as f:
        interfaces = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attentions, contact_maps, complex_ids = collect_matrices(
        predictions_csv=args.predictions,
        interfaces=interfaces,
        selected_heads=selected_heads,
        checkpoint_dir=str(paths.checkpoint(args.epoch)),
        device=device,
        batch_size=args.batch_size,
        rna_max_length=args.rna_max_length,
        protein_max_length=args.protein_max_length,
    )

    stats_df = build_stats_df(attentions, contact_maps, complex_ids)
    stats_csv = out_dir / "attention_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Wrote {stats_csv} ({len(stats_df)} rows)")

    save_contact_vectors(out_dir / "contact_vectors.npz",
                         complex_ids, attentions, contact_maps)

    meta_path = out_dir / "meta_analysis.txt"
    with open(meta_path, "w") as fh:
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams:
                    st.write(s)
            def flush(self):
                for st in self.streams:
                    st.flush()
        with redirect_stdout(_Tee(sys.stdout, fh)):
            meta_analysis(stats_df, attentions, contact_maps, selected_heads)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
