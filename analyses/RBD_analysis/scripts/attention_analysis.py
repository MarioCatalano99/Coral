#!/usr/bin/env python3
"""
RBD analysis — Step 5 (test + meta-analysis) and Step 7 (held-out sensitivity).

For each (protein, RNA) pair surviving the Step-2 filters, run inference
with the cross-attention heads chosen by `screen_attention_heads.py`,
marginalize over the RNA axis (mean), then average each protein's per-pair
vectors into one consensus profile. Per protein we compute

    Δ_protein = mean(attn | residue ∈ RBD) − mean(attn | residue ∉ RBD)

with a Welch's t-test, then aggregate across proteins via one-sample t
(alt='greater'), sign test, hierarchical bootstrap, and per-protein Cohen's d
(the latter two via `coral.stats`).

Step 7 sensitivity is a runtime flag: pass `--predictions <held-out csv>`.

Outputs (under results/<model>/heads_{all,best}/):
    rbd_attention_stats.csv         one row per protein
    rbd_meta_analysis.txt           captured stdout
    rbd_protein_vectors.npz         cached attention + mask, consumed by controls.py

heads_all  — averaged across all 12 heads (no --selected-heads).
heads_best — averaged across the screened subset (--selected-heads <json>).
"""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List

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
from coral.rbd import load_rbd_intervals
from coral.stats import cohens_d_per_unit, hierarchical_bootstrap

import _pipeline as pipe  # sibling module on sys.path[0] when run as script


def per_unit_stat(attn_vec: np.ndarray, residue_mask: np.ndarray) -> dict | None:
    """Welch's t-test of attention at RBD vs non-RBD residues."""
    rbd_scores = attn_vec[residue_mask == 1]
    non_rbd_scores = attn_vec[residue_mask == 0]
    if rbd_scores.size < 3 or non_rbd_scores.size < 3:
        return None
    t_stat, p_val = stats.ttest_ind(rbd_scores, non_rbd_scores, equal_var=False)
    return {
        "n_rbd": int(rbd_scores.size),
        "n_non_rbd": int(non_rbd_scores.size),
        "rbd_mean": float(rbd_scores.mean()),
        "non_rbd_mean": float(non_rbd_scores.mean()),
        "mean_diff": float(rbd_scores.mean() - non_rbd_scores.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }


def collect_pair_vectors(predictions_csv: Path,
                         rbd_intervals: dict,
                         protein_summary: pd.DataFrame,
                         selected_heads: List[int],
                         checkpoint_dir: str,
                         device: torch.device,
                         batch_size: int,
                         rna_max_length: int,
                         protein_max_length: int):
    """Inference loop. Returns (per_pair_attention_vectors, per_pair_masks,
    ordered_pair_ids, ordered_protein_ids)."""
    pairs_df = pipe.filter_pairs(predictions_csv, rbd_intervals, protein_summary)
    print(f"Pairs after filtering: {len(pairs_df)} "
          f"(unique proteins: {pairs_df['Protein_id'].nunique()})")
    if pairs_df.empty:
        raise RuntimeError("No pairs left after filtering — check inputs.")

    rna_tok = AutoTokenizer.from_pretrained(str(paths.DNABERT_DIR), trust_remote_code=True)
    prot_tok = AutoTokenizer.from_pretrained(pipe.PROTEIN_ENCODER, trust_remote_code=True)
    dataloader = pipe.prepare_dataloader(
        pairs_df, rna_tok, prot_tok, rbd_intervals,
        batch_size=batch_size,
        max_rna_length=rna_max_length,
        max_protein_length=protein_max_length,
        device=str(device),
    )
    model = pipe.load_pretrained_model(checkpoint_dir, str(paths.DNABERT_DIR),
                                       pipe.PROTEIN_ENCODER, device)
    analyzer = AttentionAnalyzer(model, rna_tok, prot_tok, device=device)

    attentions: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    pair_ids: List[str] = []
    protein_ids: List[str] = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        rna_in = {"input_ids": batch["rna_input_ids"].to(device),
                  "attention_mask": batch["rna_attention_mask"].to(device)}
        prot_in = {"input_ids": batch["protein_input_ids"].to(device),
                   "attention_mask": batch["protein_attention_mask"].to(device)}

        out = analyzer.extract_cross_attention_scores(
            rna_in, prot_in, subset=selected_heads,
        )
        rbd_labels_batch = batch["rbd_labels"].cpu().numpy()
        batch_pair_keys = batch["pair_keys"]
        batch_protein_keys = batch["protein_keys"]

        # With subset, AttentionAnalyzer returns shape (rna_residues, protein_residues).
        for i, attn_2d in enumerate(out["rna_to_protein_attentions"]):
            mask_full = rbd_labels_batch[i]
            mask = mask_full[mask_full != -100]
            n_res = min(attn_2d.shape[-1], mask.shape[0])
            if n_res == 0:
                continue
            attn_vec = attn_2d[:, :n_res].mean(axis=0).astype(np.float32)  # marginalize over RNA
            mask = mask[:n_res].astype(np.int8)
            attentions.append(attn_vec)
            masks.append(mask)
            pair_ids.append(batch_pair_keys[i])
            protein_ids.append(batch_protein_keys[i])

    return attentions, masks, pair_ids, protein_ids


def build_stats_df(attentions: List[np.ndarray],
                   masks: List[np.ndarray],
                   protein_ids: List[str],
                   protein_summary: pd.DataFrame,
                   n_partners: List[int]) -> tuple:
    """Run the per-protein Welch's test and assemble the long-form stats DataFrame.

    Returns (stats_df, kept_attentions, kept_masks, kept_protein_ids) — only
    proteins with ≥3 RBD and ≥3 non-RBD residues survive (matches pre-refactor).
    """
    rows: List[dict] = []
    kept_attn, kept_masks, kept_proteins = [], [], []

    for i, (attn, mask) in enumerate(zip(attentions, masks)):
        stat = per_unit_stat(attn, mask)
        if stat is None:
            continue
        stat["protein_id"] = protein_ids[i]
        stat["n_partners"] = int(n_partners[i])
        rows.append(stat)
        kept_attn.append(attn)
        kept_masks.append(mask)
        kept_proteins.append(protein_ids[i])

    cols = ["protein_id", "n_partners", "n_rbd", "n_non_rbd",
            "rbd_mean", "non_rbd_mean", "mean_diff", "t_stat", "p_value"]
    stats_df = pd.DataFrame(rows, columns=cols)

    # --- Annotate dominant_rbd from the protein summary for downstream stratification.
    dominant = (protein_summary
                .set_index("protein_id")["dominant_rbd"]
                .to_dict())
    stats_df["dominant_rbd"] = stats_df["protein_id"].map(dominant).fillna("")

    return stats_df, kept_attn, kept_masks, kept_proteins


def meta_analysis(stats_df: pd.DataFrame,
                  attentions: List[np.ndarray],
                  masks: List[np.ndarray]) -> None:
    """Print one-sample t / sign test / bootstrap / Cohen's d to stdout."""
    diffs = stats_df["mean_diff"].to_numpy()
    n = len(diffs)
    print(f"\n=== Meta-analysis ({n} proteins) ===")

    t_meta, p_meta = stats.ttest_1samp(diffs, 0, alternative="greater")
    print(f"One-sample t-test (greater): t={t_meta:.4f}, p={p_meta:.3e}")
    print(f"  mean(Δ) = {diffs.mean():.6e}, std(Δ) = {diffs.std():.6e}")

    n_pos = int((diffs > 0).sum())
    sign_p = stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue
    print(f"Sign test: proteins with Δ>0: {n_pos}/{n}, p={sign_p:.3e}")

    print("\n=== Hierarchical bootstrap ===")
    boot = hierarchical_bootstrap(attentions, masks, n_bootstrap=10000,
                                  rng=np.random.default_rng(42))
    print(f"Observed Δ̄: {boot['observed_mean_diff']:.6e}")
    print(f"Bootstrap mean: {boot['bootstrap_mean']:.6e}, sd: {boot['bootstrap_sd']:.6e}")
    print(f"95% CI: [{boot['ci_95'][0]:.6e}, {boot['ci_95'][1]:.6e}]")
    print(f"99% CI: [{boot['ci_99'][0]:.6e}, {boot['ci_99'][1]:.6e}]")
    print(f"Bootstrap two-sided p: {boot['p_two_sided']:.3e}")

    print("\n=== Per-protein Cohen's d ===")
    ds = cohens_d_per_unit(attentions, masks)
    if ds:
        arr = np.asarray(ds)
        print(f"n = {len(arr)}, mean = {arr.mean():.4f}, "
              f"median = {float(np.median(arr)):.4f}, std = {arr.std():.4f}")
        print(f"Proteins with d > 0: {int((arr > 0).sum())}/{len(arr)}")


def save_protein_vectors(out_path: Path,
                         protein_ids: List[str],
                         attentions: List[np.ndarray],
                         masks: List[np.ndarray]) -> None:
    np.savez_compressed(
        out_path,
        protein_ids=np.array(protein_ids),
        attentions=np.array(attentions, dtype=object),
        masks=np.array(masks, dtype=object),
    )
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path,
                    default=paths.PREDICTIONS_DIR / "all_predicted_interactions.csv",
                    help="CSV with [RNA_id, Protein_id, RNA_seqs, Prot_seqs, prediction]")
    ap.add_argument("--min-partners", type=int, default=1,
                    help="Drop proteins backed by fewer TP partners than this.")
    ap.add_argument("--epoch", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rna-max-length", type=int, default=6000)
    ap.add_argument("--protein-max-length", type=int, default=1250)
    ap.add_argument("--selected-heads", type=Path, default=None,
                    help="Path to selected_heads.json. If omitted, average across all 12 heads.")
    args = ap.parse_args()

    pipe.seed_everything(42)

    results_dir = paths.analysis_results("RBD_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve selected heads + head_mode → output subdir ---
    head_mode = pipe.head_mode_from_selection(args.selected_heads)
    out_dir = pipe.head_subdir(results_dir, head_mode)
    if args.selected_heads is None:
        selected_heads = list(range(pipe.N_HEADS))
        print(f"No --selected-heads provided; averaging across all {len(selected_heads)} heads "
              f"(head_mode={head_mode}).")
    else:
        if not args.selected_heads.exists():
            raise FileNotFoundError(f"{args.selected_heads} not found.")
        selected_heads = json.loads(args.selected_heads.read_text())["selected"]
        print(f"Using selected heads from {args.selected_heads}: {selected_heads} "
              f"(head_mode={head_mode}).")

    # --- Load shared annotation inputs (top level — head-mode-independent) ---
    intervals_path = results_dir / "rbd_annotations.rbd_intervals.tsv"
    summary_path = results_dir / "rbd_annotations.protein_summary.tsv"
    rbd_intervals = load_rbd_intervals(intervals_path)
    protein_summary = pd.read_csv(summary_path, sep="\t")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Inference: always per-pair ---
    pair_attentions, pair_masks, pair_ids, pair_protein_ids = collect_pair_vectors(
        predictions_csv=args.predictions,
        rbd_intervals=rbd_intervals,
        protein_summary=protein_summary,
        selected_heads=selected_heads,
        checkpoint_dir=str(paths.checkpoint(args.epoch)),
        device=device,
        batch_size=args.batch_size,
        rna_max_length=args.rna_max_length,
        protein_max_length=args.protein_max_length,
    )

    # --- Aggregate pairs → per-protein consensus profiles ---
    attentions, masks, protein_ids, n_partners = pipe.aggregate_to_protein(
        pair_attentions, pair_masks, pair_protein_ids,
    )
    attentions, masks, protein_ids, n_partners = pipe.drop_low_partner_proteins(
        attentions, masks, protein_ids, n_partners, args.min_partners,
    )
    print(f"Aggregated {len(pair_attentions)} pairs → {len(protein_ids)} proteins.")

    # --- Build per-protein stats DataFrame, run meta-analysis ---
    stats_df, kept_attn, kept_masks, kept_proteins = build_stats_df(
        attentions, masks, protein_ids, protein_summary, n_partners,
    )

    stats_csv = out_dir / "rbd_attention_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Wrote {stats_csv} ({len(stats_df)} rows)")

    save_protein_vectors(out_dir / pipe.VECTORS_FILENAME,
                         kept_proteins, kept_attn, kept_masks)

    meta_path = out_dir / "rbd_meta_analysis.txt"
    with open(meta_path, "w") as fh:
        # Tee: print to both file and console.
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams:
                    st.write(s)
            def flush(self):
                for st in self.streams:
                    st.flush()
        with redirect_stdout(_Tee(sys.stdout, fh)):
            meta_analysis(stats_df, kept_attn, kept_masks)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
