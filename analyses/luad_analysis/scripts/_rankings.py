"""Isoform rankings and descriptive cross-RBP comparisons on explicit backgrounds."""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def rank_transcripts(transcripts: pd.DataFrame, eligibility: pd.DataFrame,
                     predictions: pd.DataFrame, scope: str = "primary",
                     top_fraction: float = 0.01) -> pd.DataFrame:
    """Rank each RBP's eligible isoforms, retaining ties and fixed-cardinality flags."""
    keys = ["transcript_id", "Protein_id"]
    selected = eligibility.loc[eligibility[f"{scope}_eligible"], keys]
    if selected.duplicated().any() or transcripts.transcript_id.duplicated().any():
        raise ValueError("duplicate eligibility or transcript keys")
    columns = ["transcript_id", "RNA_id", "gene_id", "gene_name", "transcript_biotype", "sequence_length_nt"]
    frame = selected.merge(transcripts[columns], on="transcript_id", how="left", validate="many_to_one")
    frame = frame.merge(predictions, on=["RNA_id", "Protein_id"], how="left", validate="many_to_one")
    if frame.isna().any().any() or len(frame) != len(selected):
        raise ValueError("incomplete eligible-transcript score propagation")
    frame = frame.sort_values(["Protein_id", "logit_margin", "transcript_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    groups = frame.groupby("Protein_id", sort=False)
    frame["rank"] = groups.cumcount() + 1
    frame["ranking_total"] = groups.transcript_id.transform("size")
    frame["top_count"] = np.ceil(frame.ranking_total * top_fraction).astype(int)
    frame["percentile"] = 100 * (frame.ranking_total - frame["rank"]) / (frame.ranking_total - 1).clip(lower=1)
    # Average ranks keep equally scored isoforms equal in percentile contrasts.
    frame["score_percentile"] = 100 * (frame.ranking_total - groups.logit_margin.rank(ascending=False, method="average")) / (frame.ranking_total - 1).clip(lower=1)
    frame["top_candidate"] = frame["rank"] <= frame.top_count
    cutoff = frame.loc[frame["rank"] == frame.top_count].set_index("Protein_id").logit_margin
    frame["tie_inclusive_candidate"] = frame.logit_margin >= frame.Protein_id.map(cutoff)
    frame["scope"] = scope
    return frame


def gene_max_rankings(ranked: pd.DataFrame) -> pd.DataFrame:
    """Summarize genes by their maximum eligible-isoform score, for description only."""
    frame = ranked.drop_duplicates(["Protein_id", "gene_id"]).copy()
    frame = frame.drop(columns=["rank", "ranking_total", "percentile", "score_percentile", "top_count", "top_candidate", "tie_inclusive_candidate"])
    frame = frame.rename(columns={"transcript_id": "gene_score_transcript_id"})
    frame = frame.sort_values(["Protein_id", "logit_margin", "gene_id"], ascending=[True, False, True], kind="stable")
    frame["rank"] = frame.groupby("Protein_id").cumcount() + 1
    frame["ranking_total"] = frame.groupby("Protein_id").gene_id.transform("size")
    frame["percentile"] = 100 * (frame.ranking_total - frame["rank"]) / (frame.ranking_total - 1).clip(lower=1)
    return frame


def compare_rankings(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return full-ranking correlations, top-set overlap and isoform percentile contrasts."""
    correlations, overlaps, contrasts = [], [], []
    for scope, scoped in ranked.groupby("scope", sort=False):
        for left, right in combinations(sorted(scoped.Protein_id.unique()), 2):
            a = scoped.loc[scoped.Protein_id == left].set_index("transcript_id")
            b = scoped.loc[scoped.Protein_id == right].set_index("transcript_id")
            common = sorted(set(a.index) & set(b.index))
            identity = {"scope": scope, "protein_a": left, "protein_b": right}
            for unit, column in (("transcript", None), ("sequence", "RNA_id")):
                ids = common if column is None else a.loc[common].drop_duplicates(column).index.tolist()
                x, y = a.loc[ids].logit_margin, b.loc[ids].logit_margin
                variable = len(ids) >= 2 and x.nunique() > 1 and y.nunique() > 1
                correlations.append({**identity, "unit": unit, "background": "pairwise_primary_intersection" if scope == "primary" else "pairwise_sensitivity_intersection",
                    "common_count": len(ids), "spearman_rho": float(spearmanr(x, y).statistic) if variable else np.nan,
                    "kendall_tau_b": float(kendalltau(x, y).statistic) if variable else np.nan,
                    "status": "ok" if variable else "insufficient_or_constant"})
            contrast = a.loc[common, ["RNA_id", "gene_id", "gene_name"]].reset_index()
            for key, value in identity.items():
                contrast[key] = value
            contrast["percentile_a"] = a.loc[common].score_percentile.to_numpy()
            contrast["percentile_b"] = b.loc[common].score_percentile.to_numpy()
            contrast["percentile_a_minus_b"] = contrast.percentile_a - contrast.percentile_b
            contrast["percentile_background"] = "each_RBP_own_universe_average_score_ranks"
            # Also remove background-composition effects by ranking the same isoforms.
            for label, source in (("a", a), ("b", b)):
                contrast[f"common_percentile_{label}"] = 100 * (len(common) - source.loc[common].logit_margin.rank(ascending=False, method="average").to_numpy()) / max(len(common) - 1, 1)
            contrast["common_percentile_a_minus_b"] = contrast.common_percentile_a - contrast.common_percentile_b
            contrasts.append(contrast)
            for policy in ("top_candidate", "tie_inclusive_candidate"):
                for unit, col in (("transcript", "transcript_id"), ("sequence", "RNA_id"), ("gene", "gene_id")):
                    aa, bb = a.reset_index(), b.reset_index()
                    aset, bset = set(aa.loc[aa[policy], col]), set(bb.loc[bb[policy], col])
                    shared, union = len(aset & bset), len(aset | bset)
                    overlaps.append({**identity, "selection": policy, "unit": unit, "count_a": len(aset), "count_b": len(bset),
                        "intersection": shared, "union": union, "jaccard": shared / union if union else np.nan,
                        "fraction_a_shared": shared / len(aset) if aset else np.nan, "fraction_b_shared": shared / len(bset) if bset else np.nan})
    return pd.DataFrame(correlations), pd.DataFrame(overlaps), pd.concat(contrasts, ignore_index=True)
