"""Step 3: do the three panels overlap, and is CORAL's ranking protein-specific?

Expectation from biology: the paralogues IGF2BP1 and IGF2BP3 (RRMx2 + KHx4) should share many
candidates; LIN28B (cold-shock domain + zinc knuckles) should share few. If every pair of logit
margins correlates near 1, the ranking reflects the RNA rather than the protein.

For each pair, both panels are restricted to the pair's common universe before testing
(one-sided hypergeometric, nominal). This is done at transcript and at gene level. Descriptive
step; it reads no clinical data.

Writes tables/overlap.tsv and tables/margin_correlation.tsv.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import LOGGER, OUT_TABLES, load_config, setup_logging, write_table  # noqa: E402


def overlap_row(names: tuple[str, str], level: str, universe: set[str], panels: list[set[str]]) -> dict[str, object]:
    """Intersection statistics of two panels already restricted to their common universe."""
    n = len(universe)
    sizes = [len(p) for p in panels]
    shared = len(set.intersection(*panels))
    expected = n * np.prod([s / n for s in sizes])
    return {"rbps": "&".join(names), "level": level, "common_universe": n,
            **{f"panel_{i + 1}_in_common": s for i, s in enumerate(sizes)},
            "shared": shared, "expected": expected, "fold_enrichment": shared / expected if expected else np.nan,
            "jaccard": shared / len(set.union(*panels)) if any(sizes) else np.nan,
            "p_hypergeometric": float(hypergeom.sf(shared - 1, n, sizes[0], sizes[1]))}


def overlaps(rankings: pd.DataFrame, names: tuple[str, str]) -> list[dict[str, object]]:
    """Transcript- and gene-level overlap of two panels in their common universe."""
    frames = [rankings.loc[rankings.rbp.eq(name)] for name in names]
    common = set.intersection(*(set(f.transcript_id) for f in frames))
    restricted = [f.loc[f.transcript_id.isin(common)] for f in frames]
    genes_common = set(restricted[0].gene_id)
    return [
        overlap_row(names, "transcript", common, [set(f.loc[f.in_panel, "transcript_id"]) for f in restricted]),
        overlap_row(names, "gene", genes_common, [set(f.loc[f.in_panel, "gene_id"]) for f in restricted]),
    ]


def paired_margins(rankings: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Logit margins of two RBPs over their common universe."""
    cols = ["transcript_id", "logit_margin"]
    left = rankings.loc[rankings.rbp.eq(a), cols]
    right = rankings.loc[rankings.rbp.eq(b), cols]
    return left.merge(right, on="transcript_id", suffixes=(f"_{a}", f"_{b}"), validate="one_to_one")


def main() -> None:
    setup_logging()
    rbps = list(load_config()["rbps"])
    rankings = pd.read_parquet(OUT_TABLES / "rankings.parquet")
    pairs = list(combinations(rbps, 2))

    table = pd.DataFrame([row for pair in pairs for row in overlaps(rankings, pair)])
    write_table(table, OUT_TABLES / "overlap.tsv")
    for row in table.loc[table.level.eq("transcript")].itertuples():
        LOGGER.info("%s: %d shared transcripts (%.1f-fold expected, Jaccard %.3f)", row.rbps, row.shared, row.fold_enrichment, row.jaccard)

    correlations = []
    for a, b in pairs:
        merged = paired_margins(rankings, a, b)
        rho = spearmanr(merged[f"logit_margin_{a}"], merged[f"logit_margin_{b}"]).statistic
        correlations.append({"rbps": f"{a}&{b}", "n_transcripts": len(merged), "spearman_rho": float(rho)})
        LOGGER.info("%s & %s: margin Spearman rho %.3f over %d transcripts", a, b, rho, len(merged))
    write_table(pd.DataFrame(correlations), OUT_TABLES / "margin_correlation.tsv")


if __name__ == "__main__":
    main()
