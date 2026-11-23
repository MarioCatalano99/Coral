"""Statistics helpers shared across attention analyses.

Two functions, both shape-agnostic in the per-unit arrays:

  - hierarchical_bootstrap: resample units (e.g. complexes, (protein, RNA)
    pairs) with replacement, recompute the per-unit Δ = mean(score|label==1)
    − mean(score|label==0), and return the distribution of the bootstrap
    averages.

  - cohens_d_per_unit: per-unit Cohen's d using the pooled-std formula
    sqrt((std₁² + std₀²) / 2).

  - bootstrap_over_units: cluster bootstrap of an aggregate (mean/median/…)
    over *precomputed* per-unit scalars, e.g. a per-complex AUPRC distribution.

The score and label arrays must be aligned and the same shape; numpy
boolean indexing flattens them, so 1D residue masks and 2D label matrices
both work without a code change.
"""

from typing import Callable, List, Optional, Sequence

import numpy as np


Array = np.ndarray


def _per_unit_mean_diff(scores: Array, labels: Array) -> Optional[float]:
    """mean(scores | labels==1) − mean(scores | labels==0); None if empty."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    return float(pos.mean() - neg.mean())


def hierarchical_bootstrap(
    per_unit_scores: Sequence[Array],
    per_unit_labels: Sequence[Array],
    n_bootstrap: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Cluster bootstrap over units of analysis.

    For each iteration, resample `n_units` indices with replacement, compute
    the per-unit Δ for each resampled unit, and average. Returns the
    distribution and the two-sided empirical p-value against Δ̄ = 0.
    """
    n_units = len(per_unit_scores)
    assert n_units == len(per_unit_labels)
    rng = rng if rng is not None else np.random.default_rng()

    observed_per_unit = np.array(
        [d for d in (_per_unit_mean_diff(s, l)
                     for s, l in zip(per_unit_scores, per_unit_labels))
         if d is not None]
    )
    observed_mean_diff = float(observed_per_unit.mean())

    bootstrap_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_units, size=n_units)
        diffs = []
        for j in idx:
            d = _per_unit_mean_diff(per_unit_scores[j], per_unit_labels[j])
            if d is not None:
                diffs.append(d)
        bootstrap_diffs[i] = np.mean(diffs) if diffs else np.nan

    bootstrap_diffs = bootstrap_diffs[~np.isnan(bootstrap_diffs)]

    ci_95 = np.percentile(bootstrap_diffs, [2.5, 97.5])
    ci_99 = np.percentile(bootstrap_diffs, [0.5, 99.5])
    p_two_sided = 2.0 * min(np.mean(bootstrap_diffs <= 0),
                            np.mean(bootstrap_diffs >= 0))

    return {
        "observed_mean_diff": observed_mean_diff,
        "observed_per_unit_diffs": observed_per_unit,
        "bootstrap_mean_diffs": bootstrap_diffs,
        "bootstrap_mean": float(bootstrap_diffs.mean()),
        "bootstrap_sd": float(bootstrap_diffs.std()),
        "ci_95": tuple(float(x) for x in ci_95),
        "ci_99": tuple(float(x) for x in ci_99),
        "p_two_sided": float(p_two_sided),
    }


def bootstrap_over_units(
    values: Sequence[float],
    n_bootstrap: int = 10000,
    agg: Callable[[Array], float] = np.mean,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Cluster bootstrap of an aggregate over independent units.

    Unlike `hierarchical_bootstrap`, which recomputes the label-1-vs-label-0
    Δ from raw (score, label) arrays, this takes *precomputed* per-unit scalars
    (e.g. one AUPRC per complex) and resamples the units with replacement,
    recomputing `agg` each iteration. So any metric — AUPRC, AUROC, lift — can be
    summarized with CIs without re-deriving it inside the bootstrap.

    Args:
        values: one scalar per unit (NaNs dropped).
        n_bootstrap: number of resamples.
        agg: aggregate applied to the resampled values (default mean; pass
            np.median for the median).
        rng: optional numpy Generator for reproducibility.

    Returns dict with observed agg, bootstrap mean/sd, and 95/99% percentile CIs.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = vals.size
    if n == 0:
        raise ValueError("bootstrap_over_units: no finite values.")
    rng = rng if rng is not None else np.random.default_rng()

    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = agg(vals[idx])

    ci_95 = np.percentile(boot, [2.5, 97.5])
    ci_99 = np.percentile(boot, [0.5, 99.5])
    return {
        "observed": float(agg(vals)),
        "n_units": int(n),
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_sd": float(boot.std()),
        "ci_95": tuple(float(x) for x in ci_95),
        "ci_99": tuple(float(x) for x in ci_99),
    }


def cohens_d_per_unit(
    per_unit_scores: Sequence[Array],
    per_unit_labels: Sequence[Array],
) -> List[float]:
    """Per-unit Cohen's d using pooled-std denominator. Skips degenerate units."""
    out: List[float] = []
    for scores, labels in zip(per_unit_scores, per_unit_labels):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.size == 0 or neg.size == 0:
            continue
        pooled = np.sqrt((pos.std() ** 2 + neg.std() ** 2) / 2.0)
        if pooled <= 0:
            continue
        out.append(float((pos.mean() - neg.mean()) / pooled))
    return out
