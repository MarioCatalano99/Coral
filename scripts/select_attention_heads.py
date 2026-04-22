"""Identify which cross-attention heads systematically prioritise contact positions.

Reads `attention_heads_stats.csv` produced by `scripts/compute_attention_stats.py`,
runs a Friedman test across all heads, then Wilcoxon signed-rank post-hocs
(Benjamini-Hochberg FDR) of the best-median head against every other head.
"""

import argparse
import sys

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", default="attention_heads_stats.csv",
                   help="Per-head statistics CSV produced by compute_attention_stats.py")
    p.add_argument("--fdr-alpha", type=float, default=0.05,
                   help="FDR threshold for Benjamini-Hochberg (default: 0.05)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("ATTENTION HEAD DIFFERENTIAL CONTACT-POSITION ANALYSIS")
    print("=" * 72)

    df = pd.read_csv(args.input_csv)
    df["head_id"] = df["head_id"] - 1  # 0-indexed

    print(
        f"\nLoaded {len(df):,} rows  |  "
        f"{df['matrix_id'].nunique()} complexes  |  "
        f"{df['head_id'].nunique()} heads (0-indexed)"
    )

    delta = df.pivot(index="matrix_id", columns="head_id", values="mean_diff")
    heads = sorted(delta.columns)
    n_heads = len(heads)
    n_complexes = len(delta)
    print(f"Pivot table: {n_complexes} complexes x {n_heads} heads\n")

    # --- Summary by median delta ---
    print("-" * 72)
    print(f"{'Head':>6}  {'Mean':>12}  {'Median':>12}  {'Std':>12}  {'Cohen d':>10}  {'Rank':>6}")
    print("-" * 72)

    rows = []
    for h in heads:
        vals = delta[h]
        mean_d, median_d, std_d = vals.mean(), vals.median(), vals.std()
        cohens_d = mean_d / std_d if std_d > 0 else 0.0
        rows.append({"head": h, "mean": mean_d, "median": median_d, "std": std_d, "cohens_d": cohens_d})

    summary = pd.DataFrame(rows).sort_values("median", ascending=False)
    summary["rank"] = range(1, len(summary) + 1)
    summary["head"] = summary["head"].astype(int)
    summary["rank"] = summary["rank"].astype(int)
    for _, row in summary.iterrows():
        print(
            f"{int(row['head']):>6}  {row['mean']:>12.6f}  {row['median']:>12.6f}  "
            f"{row['std']:>12.6f}  {row['cohens_d']:>10.6f}  {int(row['rank']):>6}"
        )
    print()

    # --- Friedman test ---
    print("=" * 72)
    print("STEP 1: Friedman Test (non-parametric repeated-measures ANOVA)")
    print("=" * 72)

    arrays = [delta[h].values for h in heads]
    chi2, p_friedman = friedmanchisquare(*arrays)
    print(f"\n  Heads compared : {n_heads}")
    print(f"  Complexes (N)  : {n_complexes}")
    print(f"  chi2({n_heads - 1})          : {chi2:.1f}")
    print(f"  p-value        : {p_friedman:.4e}")

    if p_friedman >= args.fdr_alpha:
        print("  -> No significant variation across heads. Stopping.")
        sys.exit(0)
    print("  -> Significant variation across heads detected. Proceeding to post-hoc.\n")

    # --- Best-vs-rest post-hocs ---
    print("=" * 72)
    print("STEP 2: Best-vs-Rest Wilcoxon Signed-Rank Tests + BH FDR")
    print("=" * 72)

    best_head = int(summary.iloc[0]["head"])
    other_heads = [h for h in heads if h != best_head]

    print(f"\n  Best head by median delta: Head {best_head} "
          f"(median = {delta[best_head].median():.6f})")
    print(f"  Comparing Head {best_head} against each of the other {len(other_heads)} heads:\n")

    test_rows, raw_pvals = [], []
    for h in other_heads:
        stat, pval = wilcoxon(delta[best_head].values, delta[h].values, alternative="two-sided")
        raw_pvals.append(pval)
        test_rows.append({"head_A": best_head, "head_B": h, "W_stat": stat, "p_raw": pval})

    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=args.fdr_alpha, method="fdr_bh")
    results = pd.DataFrame(test_rows)
    results["p_adj_BH"] = pvals_corrected
    results["reject_H0"] = reject
    results = results.sort_values("p_adj_BH").reset_index(drop=True)

    print(f"  {'Head A':>6}  {'Head B':>6}  {'W stat':>12}  "
          f"{'p (raw)':>12}  {'p (BH adj)':>12}  {'Sig.':>6}")
    print("  " + "-" * 64)
    for _, row in results.iterrows():
        sig_marker = "***" if row["reject_H0"] else "n.s."
        print(
            f"  {int(row['head_A']):>6}  {int(row['head_B']):>6}  "
            f"{row['W_stat']:>12.1f}  {row['p_raw']:>12.4e}  "
            f"{row['p_adj_BH']:>12.4e}  {sig_marker:>6}"
        )

    # --- Retained heads ---
    print("\n" + "=" * 72)
    print("STEP 3: Head Selection for Downstream Analysis")
    print("=" * 72)

    not_rejected = results[~results["reject_H0"]]
    retained_heads = sorted([best_head] + [int(h) for h in not_rejected["head_B"]])
    print(f"\n  Best head: {best_head}")
    if len(not_rejected) > 0:
        ns_heads = [int(h) for h in not_rejected["head_B"]]
        print(f"  Head(s) NOT significantly different from best: {ns_heads}")
    else:
        print("  All other heads are significantly different from the best head.")
    print(f"\n  Selected heads for downstream analysis: {retained_heads}")

    # --- Averaged attention across retained heads ---
    print("\n" + "=" * 72)
    print("STEP 4: Average Attention from Selected Heads")
    print("=" * 72)

    df_selected = df[df["head_id"].isin(retained_heads)].copy()
    averaged = (
        df_selected.groupby("matrix_id")
        .agg(
            contact_mean=("contact_mean", "mean"),
            non_contact_mean=("non_contact_mean", "mean"),
            mean_diff=("mean_diff", "mean"),
        )
        .reset_index()
    )

    print(f"\n  Averaged scores across heads {retained_heads} for {len(averaged)} complexes:")
    print(f"    Mean  contact attention    : {averaged['contact_mean'].mean():.6e}")
    print(f"    Mean  non-contact attention: {averaged['non_contact_mean'].mean():.6e}")
    print(f"    Mean  delta (contact - non): {averaged['mean_diff'].mean():.6e}")
    print(f"    Median delta               : {averaged['mean_diff'].median():.6e}")

    print("\n" + "=" * 72)
    print("APPENDIX: Cohen's d by Head (sorted descending)")
    print("=" * 72)
    cohens = summary[["head", "cohens_d"]].sort_values("cohens_d", ascending=False)
    print(f"\n  {'Head':>6}  {'Cohen d':>12}")
    print("  " + "-" * 22)
    for _, row in cohens.iterrows():
        print(f"  {int(row['head']):>6}  {row['cohens_d']:>12.6f}")

    print("\n" + "=" * 72)
    print("Analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
