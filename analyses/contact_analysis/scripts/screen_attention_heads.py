#!/usr/bin/env python3
"""
Contact analysis — cross-attention head screening.

For every predicted-positive PDB complex with ≥3 residue contacts, run inference
with all 12 cross-attention heads (last layer) and compute, per head, the
per-complex difference

    Δ_h = mean(attn_h | contact) − mean(attn_h | non-contact)

over the flattened RNA×protein attention matrix. Then run Friedman across heads
on the per-complex Δ_h and a Wilcoxon best-vs-rest post-hoc with Benjamini–
Hochberg FDR (delegated to `coral.head_selection.select_heads_by_friedman_wilcoxon`).
Heads not significantly different from the best (by median Δ) are retained.

Merges the former `get_attention_heads_stats.py` (inference → per-head CSV) and
`attention_heads_analysis.py` (selection) into one script, matching the
RBD_analysis / Analysis_B blueprint.

Outputs (under results/<model>/):
    attention_heads_stats.csv   one row per (complex, head); head_id is 0-indexed
    selected_heads.json         {best_head, selected, friedman_chi2, friedman_p,
                                 wilcoxon_padj}

Re-run selection without re-running inference:
    python ... screen_attention_heads.py --from-csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))  # no-op once `pip install -e .` has been run

from coral import paths
from coral.attention import AttentionAnalyzer
from coral.head_selection import select_heads_by_friedman_wilcoxon

import _pipeline as pipe  # sibling module, on sys.path[0] when run as a script


def per_complex_head_stats(attn_per_head: np.ndarray,
                           contact_map: np.ndarray) -> List[dict]:
    """attn_per_head: (n_heads, rna_len, prot_len). contact_map: (rna_len, prot_len) 0/1.

    Returns one row per head, or [] if the complex has <3 contact positions.
    """
    contact_flat = contact_map.reshape(-1)
    attn_flat = attn_per_head.reshape(attn_per_head.shape[0], -1)  # (n_heads, rna*prot)

    is_contact = contact_flat == 1
    if is_contact.sum() < 3:
        return []
    contact = attn_flat[:, is_contact]        # (n_heads, n_contact)
    non_contact = attn_flat[:, ~is_contact]   # (n_heads, n_non_contact)

    rows = []
    for h in range(attn_flat.shape[0]):
        rows.append({
            "head_id": h,
            "n_contact": int(is_contact.sum()),
            "n_non_contact": int((~is_contact).sum()),
            "contact_mean": float(contact[h].mean()),
            "non_contact_mean": float(non_contact[h].mean()),
            "mean_diff": float(contact[h].mean() - non_contact[h].mean()),
        })
    return rows


def collect_head_stats(predictions_csv: Path,
                       interfaces: dict,
                       checkpoint_dir: str,
                       device: torch.device,
                       batch_size: int,
                       rna_max_length: int,
                       protein_max_length: int) -> pd.DataFrame:
    """Inference loop → long-form per-(complex × head) stats DataFrame."""
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

    rows: List[dict] = []
    matrix_id = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        rna_in = {"input_ids": batch["rna_input_ids"].to(device),
                  "attention_mask": batch["rna_attention_mask"].to(device)}
        prot_in = {"input_ids": batch["protein_input_ids"].to(device),
                   "attention_mask": batch["protein_attention_mask"].to(device)}
        contact_labels = batch["contact_labels"].to(device)

        # subset=None → per-head 3D attention (n_heads, rna_len, prot_len).
        out = analyzer.extract_cross_attention_scores(rna_in, prot_in, contact_labels, subset=None)
        # Geometric-mean fusion of both directions: fuse head h of r2p with
        # head h of p2r, per complex.
        r2p = out["rna_to_protein_attentions"]   # each (n_heads, rna_len, prot_len)
        p2r = out["protein_to_rna_attentions"]   # each (n_heads, prot_len, rna_len)
        attns = [pipe.geometric_mean_fuse(r, p) for r, p in zip(r2p, p2r)]
        contacts = out["rna2protein_contact_labels"]
        keys = batch["contact_keys"]

        for i in range(len(attns)):
            head_rows = per_complex_head_stats(attns[i], contacts[i])
            if not head_rows:
                continue
            for r in head_rows:
                r["complex_id"] = keys[i]
                r["matrix_id"] = matrix_id
            rows.extend(head_rows)
            matrix_id += 1

    cols = ["complex_id", "matrix_id", "head_id", "n_contact", "n_non_contact",
            "contact_mean", "non_contact_mean", "mean_diff"]
    df = pd.DataFrame(rows, columns=cols)
    print(f"Valid complexes: {df['matrix_id'].nunique()}")
    return df


def run_head_selection(stats_df: pd.DataFrame, fdr_alpha: float) -> dict:
    """Friedman + best-vs-rest Wilcoxon (BH FDR) with a human-readable report."""
    delta = stats_df.pivot(index="matrix_id", columns="head_id", values="mean_diff")
    n_complexes, n_heads = delta.shape
    print("=" * 72)
    print("ATTENTION HEAD DIFFERENTIAL CONTACT-POSITION ANALYSIS")
    print("=" * 72)
    print(f"\nPivot table: {n_complexes} complexes × {n_heads} heads (0-indexed)\n")

    res = select_heads_by_friedman_wilcoxon(delta, fdr_alpha=fdr_alpha)
    summary, posthoc = res["head_summary"], res["posthoc"]
    best_head = int(res["best_head"])
    retained = [int(h) for h in res["retained_heads"]]

    # --- Per-head summary (sorted by median Δ) ---
    print("-" * 72)
    print(f"{'Head':>6}  {'Mean':>12}  {'Median':>12}  {'Std':>12}  {'Cohen d':>10}  {'Rank':>6}")
    print("-" * 72)
    for _, row in summary.iterrows():
        print(f"{int(row['head']):>6}  {row['mean']:>12.6f}  {row['median']:>12.6f}  "
              f"{row['std']:>12.6f}  {row['cohens_d']:>10.6f}  {int(row['rank']):>6}")

    # --- Friedman ---
    print("\n" + "=" * 72)
    print("STEP 1: Friedman Test (non-parametric repeated-measures ANOVA)")
    print("=" * 72)
    print(f"\n  Heads compared : {n_heads}")
    print(f"  Complexes (N)  : {n_complexes}")
    print(f"  χ²({n_heads - 1})          : {res['friedman_chi2']:.1f}")
    print(f"  p-value        : {res['friedman_p']:.4e}")
    if res["friedman_p"] >= fdr_alpha:
        print("  → No significant variation detected. Stopping here.\n")
        return res
    print("  → Significant variation across heads detected. Proceeding to post-hoc.\n")

    # --- Post-hoc ---
    print("=" * 72)
    print("STEP 2: Best-vs-Rest Wilcoxon Signed-Rank Tests + BH FDR")
    print("=" * 72)
    print(f"\n  Best head by median Δ: Head {best_head} "
          f"(median = {delta[best_head].median():.6f})")
    print(f"  Comparing Head {best_head} against each of the other {len(posthoc)} heads:\n")
    print(f"  {'Head A':>6}  {'Head B':>6}  {'W stat':>12}  "
          f"{'p (raw)':>12}  {'p (BH adj)':>12}  {'Sig.':>6}")
    print("  " + "-" * 64)
    for _, row in posthoc.iterrows():
        sig = "***" if row["reject_H0"] else "n.s."
        print(f"  {int(row['head_A']):>6}  {int(row['head_B']):>6}  "
              f"{row['W_stat']:>12.1f}  {row['p_raw']:>12.4e}  "
              f"{row['p_adj_BH']:>12.4e}  {sig:>6}")

    # --- Selection ---
    print("\n" + "=" * 72)
    print("STEP 3: Head Selection for Downstream Analysis")
    print("=" * 72)
    ns_heads = [h for h in retained if h != best_head]
    print(f"\n  Best head: {best_head}")
    if ns_heads:
        print(f"  Head(s) NOT significantly different from best: {ns_heads}")
    else:
        print("  All other heads are significantly different from the best head.")
    print(f"\n  ✓ Selected heads for downstream analysis: {retained}")
    return res


def write_selected_heads(res: dict, out_path: Path) -> None:
    payload = {
        "best_head": int(res["best_head"]),
        "selected": sorted(int(h) for h in res["retained_heads"]),
        "friedman_chi2": res["friedman_chi2"],
        "friedman_p": res["friedman_p"],
        "wilcoxon_padj": {
            int(row["head_B"]): float(row["p_adj_BH"])
            for _, row in res["posthoc"].iterrows()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path,
                    default=paths.PREDICTIONS_DIR / "pdb_predicted_interactions.csv",
                    help="CSV with [RNA_id, Protein_id, RNA_seqs, Prot_seqs, prediction]")
    ap.add_argument("--interfaces", type=Path,
                    default=paths.PDB_DIR / "multi_chain_interaction_interfaces.json",
                    help="JSON of residue contacts keyed by {RNA_id}_{chain}.")
    ap.add_argument("--epoch", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rna-max-length", type=int, default=None)
    ap.add_argument("--protein-max-length", type=int, default=None)
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    ap.add_argument("--from-csv", action="store_true",
                    help="Skip inference and re-run selection from cached stats CSV.")
    args = ap.parse_args()

    pipe.seed_everything(42)

    results_dir = paths.analysis_results("contact_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = results_dir / "attention_heads_stats.csv"
    selected_json = results_dir / "selected_heads.json"

    if args.from_csv:
        if not stats_csv.exists():
            raise FileNotFoundError(f"--from-csv set but {stats_csv} does not exist.")
        stats_df = pd.read_csv(stats_csv)
        print(f"Loaded cached stats: {len(stats_df)} rows from {stats_csv}")
    else:
        with open(args.interfaces) as f:
            interfaces = json.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stats_df = collect_head_stats(
            predictions_csv=args.predictions,
            interfaces=interfaces,
            checkpoint_dir=str(paths.checkpoint(args.epoch)),
            device=device,
            batch_size=args.batch_size,
            rna_max_length=args.rna_max_length,
            protein_max_length=args.protein_max_length,
        )
        stats_df.to_csv(stats_csv, index=False)
        print(f"Wrote {stats_csv} ({len(stats_df)} rows)")

    res = run_head_selection(stats_df, fdr_alpha=args.fdr_alpha)
    write_selected_heads(res, selected_json)


if __name__ == "__main__":
    main()
