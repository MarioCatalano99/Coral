#!/usr/bin/env python3
"""
RBD analysis — Step 6 controls.

Reads the per-protein attention vectors cached by `attention_analysis.py` and:

1. Position-shuffle null: for each protein, cyclically rotate the RBD mask
   along the protein by a random offset, recompute Δ_protein, and average
   across proteins. Repeat 1000× to build a null distribution. The cyclic
   shift preserves number, length, and gap structure of intervals — only
   placement changes — so this rules out "attention is just elevated in
   the middle of the protein"-style confounds.

2. Family stratification: split proteins by `dominant_rbd` and re-run a small
   meta-analysis (one-sample t, Cohen's d on per-protein Δ) per family with
   ≥ N proteins. Cohen's d is reported with a percentile 95% CI from an
   ordinary row bootstrap — each row is one protein, so rows are already
   independent and need no clustering.

The `--heads` flag selects which upstream cache to consume: 'all' = average
over all 12 heads, 'best' = average over the screened subset. Must match the
head_mode used by attention_analysis.py.

Outputs (under results/<model>/heads_{all,best}/):
    rbd_null_distribution.csv   one row per shuffle iteration
                                (aggregate Δ̄ over proteins)
    rbd_family_stratified.csv   one row per RBD family
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))  # no-op once `pip install -e .` has been run
from coral import paths

# Deliberately does NOT import _pipeline: this script is pure post-processing on
# the cached npz, and _pipeline drags in torch/peft/the model definition.
VECTORS_FILENAME = "rbd_protein_vectors.npz"


def position_shuffle_null(attentions: list[np.ndarray],
                          masks: list[np.ndarray],
                          n_iter: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Cyclic-shift null distribution of aggregate Δ̄ across proteins."""
    null_diffs = np.empty(n_iter)
    for it in range(n_iter):
        per_protein = []
        for attn, mask in zip(attentions, masks):
            L = mask.shape[0]
            if L < 2:
                continue
            # Non-zero offset so the shuffled mask actually moves.
            offset = int(rng.integers(1, L))
            shuffled = np.roll(mask, offset)
            rbd = attn[shuffled == 1]
            non_rbd = attn[shuffled == 0]
            if rbd.size == 0 or non_rbd.size == 0:
                continue
            per_protein.append(rbd.mean() - non_rbd.mean())
        null_diffs[it] = np.mean(per_protein) if per_protein else np.nan

    null_diffs = null_diffs[~np.isnan(null_diffs)]
    return null_diffs


def cohens_d(diffs: np.ndarray) -> float:
    """One-sample Cohen's d against zero. Returns 0.0 when n<2 or std==0."""
    if diffs.size < 2:
        return 0.0
    std = diffs.std(ddof=1)
    return float(diffs.mean() / std) if std > 0 else 0.0


def bootstrap_d_ci(sub: pd.DataFrame,
                   n_boot: int,
                   rng: np.random.Generator) -> tuple[float, float]:
    """Percentile 95% CI for Cohen's d via an ordinary row bootstrap.

    Each row is one protein's consensus profile, so rows are independent and
    need no clustering."""
    diffs = sub["mean_diff"].to_numpy()
    n = diffs.size
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = cohens_d(diffs[rng.integers(0, n, n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def family_stratified(stats_df: pd.DataFrame,
                      min_proteins: int,
                      n_boot: int,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Per-family one-sample t-test on per-protein mean_diff plus a bootstrap CI
    for Cohen's d. One row of the upstream stats DF is one protein, so
    `n_proteins` is just the family's row count."""
    rows = []
    for family, sub in stats_df.groupby("dominant_rbd"):
        if family == "" or len(sub) < min_proteins:
            continue
        diffs = sub["mean_diff"].to_numpy()
        t, p = stats.ttest_1samp(diffs, 0, alternative="greater")
        d = cohens_d(diffs)
        d_lo, d_hi = bootstrap_d_ci(sub, n_boot, rng)
        rows.append({
            "family": family,
            "n_proteins": int(sub["protein_id"].nunique()),
            "mean_diff": float(diffs.mean()),
            "t_stat": float(t),
            "p_value": float(p),
            "cohens_d": d,
            "d_lo": d_lo,
            "d_hi": d_hi,
        })
    return pd.DataFrame(rows).sort_values("cohens_d", ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", choices=["all", "best"], default="all",
                    help="Which upstream cache to consume: 'all' (averaged over all 12 heads) "
                         "or 'best' (averaged over the screened subset). Must match the "
                         "head_mode used by attention_analysis.py.")
    ap.add_argument("--n-iter", type=int, default=1000,
                    help="Number of position-shuffle iterations.")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="Number of bootstrap resamples for the per-family Cohen's d CI.")
    ap.add_argument("--min-family-proteins", type=int, default=10,
                    help="Minimum proteins required to report a family.")
    args = ap.parse_args()

    results_dir = paths.analysis_results("RBD_analysis")
    out_dir = results_dir / f"heads_{args.heads}"

    vectors_path = out_dir / VECTORS_FILENAME
    stats_csv = out_dir / "rbd_attention_stats.csv"

    hint = " --selected-heads ..." if args.heads == "best" else ""
    if not vectors_path.exists():
        raise FileNotFoundError(
            f"{vectors_path} not found. Run attention_analysis.py{hint} first."
        )
    if not stats_csv.exists():
        raise FileNotFoundError(
            f"{stats_csv} not found. Run attention_analysis.py{hint} first."
        )

    data = np.load(vectors_path, allow_pickle=True)
    attentions = list(data["attentions"])
    masks = list(data["masks"])
    protein_ids = list(data["protein_ids"])
    print(f"Loaded {len(protein_ids)} proteins from {vectors_path}")

    # --- Position-shuffle null ---
    print(f"\nPosition-shuffle null: {args.n_iter} iterations...")
    rng = np.random.default_rng(42)
    null_diffs = position_shuffle_null(attentions, masks, args.n_iter, rng)

    stats_df = pd.read_csv(stats_csv)
    observed_mean_diff = float(stats_df["mean_diff"].mean())
    # (k+1)/(N+1) convention (Phipson & Smyth 2010): a permutation p is bounded
    # below by 1/(N+1), so report that floor instead of a misleading 0.0000.
    # Keeps this in step with the "p < 0.001" label in Figure 6 panel B.
    n_exceed = int((null_diffs >= observed_mean_diff).sum())
    p_emp = (n_exceed + 1) / (len(null_diffs) + 1)
    p_str = "< 0.001" if p_emp < 1e-3 else f"{p_emp:.4f}"
    print(f"Observed Δ̄: {observed_mean_diff:.6e}")
    print(f"Null Δ̄: mean={null_diffs.mean():.6e}, sd={null_diffs.std():.6e}")
    print(f"Empirical p (Δ̄_real ≥ shuffle): {p_str}")

    null_path = out_dir / "rbd_null_distribution.csv"
    pd.DataFrame({"iteration": np.arange(len(null_diffs)),
                  "shuffled_mean_diff": null_diffs}).to_csv(null_path, index=False)
    print(f"Wrote {null_path}")

    # --- Family stratification ---
    print(f"\nFamily stratification (min {args.min_family_proteins} proteins/family, "
          f"{args.n_boot} bootstrap resamples)...")
    boot_rng = np.random.default_rng(43)
    fam_df = family_stratified(stats_df, args.min_family_proteins,
                               args.n_boot, boot_rng)
    if fam_df.empty:
        print("  no families meet the minimum-proteins threshold.")
    else:
        print(fam_df.to_string(index=False))

    fam_path = out_dir / "rbd_family_stratified.csv"
    fam_df.to_csv(fam_path, index=False)
    print(f"Wrote {fam_path}")


if __name__ == "__main__":
    main()
