"""Per-head screening shared across attention analyses.

The screening recipe mirrors what `contact_analysis/attention_heads_analysis.py`
introduced: Friedman across heads on per-unit Δ; if significant, Wilcoxon
signed-rank post-hoc of best-by-median head vs each other head with
Benjamini–Hochberg FDR. Heads not significantly different from the best
are retained.
"""

from typing import List

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def select_heads_by_friedman_wilcoxon(
    delta_matrix: pd.DataFrame,
    fdr_alpha: float = 0.05,
) -> dict:
    """Pick heads by Friedman + best-vs-rest Wilcoxon with BH FDR.

    Args:
        delta_matrix: DataFrame indexed by unit (e.g. matrix_id, pair_id),
            columns = head ids, values = per-unit mean_diff.
        fdr_alpha: significance threshold used for Friedman + BH-adjusted
            Wilcoxon.

    Returns dict with:
        best_head        — head with the largest median Δ
        retained_heads   — sorted [best_head] + heads not BH-rejected
        friedman_chi2    — χ² statistic
        friedman_p       — Friedman p-value
        posthoc          — DataFrame of Wilcoxon best-vs-rest with W stat,
                           raw p, BH-adjusted p, reject flag
        head_summary     — DataFrame of per-head mean/median/std/Cohen's d,
                           sorted by median descending
    """
    heads: List = sorted(delta_matrix.columns)
    n_heads = len(heads)

    # Per-head summary (sorted by median Δ descending).
    rows = []
    for h in heads:
        vals = delta_matrix[h].dropna()
        std = float(vals.std())
        rows.append({
            "head": h,
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": std,
            "cohens_d": float(vals.mean() / std) if std > 0 else 0.0,
        })
    head_summary = (pd.DataFrame(rows)
                    .sort_values("median", ascending=False)
                    .reset_index(drop=True))
    head_summary["rank"] = head_summary.index + 1

    # Friedman across all heads.
    arrays = [delta_matrix[h].dropna().values for h in heads]
    chi2, p_friedman = friedmanchisquare(*arrays)

    posthoc = pd.DataFrame(columns=["head_A", "head_B", "W_stat",
                                    "p_raw", "p_adj_BH", "reject_H0"])
    retained = [head_summary.iloc[0]["head"]]

    if p_friedman < fdr_alpha:
        best_head = head_summary.iloc[0]["head"]
        others = [h for h in heads if h != best_head]

        rows = []
        raw_p = []
        for h in others:
            stat, pval = wilcoxon(delta_matrix[best_head].values,
                                  delta_matrix[h].values,
                                  alternative="two-sided")
            raw_p.append(pval)
            rows.append({"head_A": best_head, "head_B": h,
                         "W_stat": float(stat), "p_raw": float(pval)})

        reject, p_adj, _, _ = multipletests(raw_p, alpha=fdr_alpha,
                                            method="fdr_bh")
        posthoc = pd.DataFrame(rows)
        posthoc["p_adj_BH"] = p_adj
        posthoc["reject_H0"] = reject
        posthoc = posthoc.sort_values("p_adj_BH").reset_index(drop=True)

        retained = [best_head] + posthoc.loc[~posthoc["reject_H0"],
                                             "head_B"].tolist()

    return {
        "best_head": head_summary.iloc[0]["head"],
        "retained_heads": sorted(retained),
        "friedman_chi2": float(chi2),
        "friedman_p": float(p_friedman),
        "posthoc": posthoc,
        "head_summary": head_summary,
    }
