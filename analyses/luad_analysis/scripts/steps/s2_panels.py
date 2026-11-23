"""Step 2: which transcripts does CORAL rank in the top 1% for each RBP?

CORAL scores an (RNA, protein) pair from sequence alone, so a transcript's score is the same in
every patient. Each universe is ranked by logit margin (descending; ties by transcript ID), and
the top ceil(1%) forms the fixed panel. The panel description compares length, GC content and
expression with the universe. No clinical file is read.

Writes tables/rankings.parquet and tables/panel_summary.tsv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (DATA, LOGGER, OUT_TABLES, REPO, ceil_fraction, checkpoint_hash, load_config,  # noqa: E402
                    read_predictions, setup_logging, write_table)


def load_scores(checkpoint: str) -> pd.DataFrame:
    """Return validated CORAL predictions made with the configured checkpoint."""
    directory = DATA / "luad_predictions" / Path(checkpoint).name
    predictions, manifests = read_predictions(directory, DATA / "luad_rna_sequences.parquet", DATA / "luad_proteins.tsv")
    recorded = json.loads(manifests[0].read_text())["checkpoint_sha256"]
    actual = checkpoint_hash(REPO / checkpoint)
    if recorded != actual:
        raise ValueError(f"predictions in {directory} were made with checkpoint {recorded}, configured one is {actual}")
    LOGGER.info("predictions: %d pairs, checkpoint %s", len(predictions), actual[:12])
    return predictions[["RNA_id", "Protein_id", "logit_margin", "interaction_probability"]]


def rank_universe(universe: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    """Rank by descending logit margin, ties broken by transcript ID; flag the top ceil(fraction)."""
    order = np.lexsort((universe.transcript_id.to_numpy(), -universe.logit_margin.to_numpy()))
    ranked = universe.iloc[order].reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["in_panel"] = ranked["rank"] <= ceil_fraction(top_fraction, len(ranked))
    return ranked


def describe(rbp: str, ranked: pd.DataFrame, gene_id: str) -> dict[str, object]:
    """Summarize what kind of transcripts the panel holds, against its universe."""
    panel = ranked.loc[ranked.in_panel]
    return {
        "rbp": rbp, "universe_transcripts": len(ranked), "panel_transcripts": len(panel),
        "panel_genes": panel.gene_id.nunique(), "max_isoforms_per_gene": int(panel.groupby("gene_id").size().max()),
        "panel_transcripts_with_shared_sequence": int(panel.RNA_id.duplicated(keep=False).sum()),
        "rbp_own_isoforms_in_panel": int(panel.gene_id.eq(gene_id).sum()),
        "panel_margin_min": panel.logit_margin.min(), "panel_margin_median": panel.logit_margin.median(),
        "panel_probability_min": panel.interaction_probability.min(),
        "panel_probability_median": panel.interaction_probability.median(),
        "universe_fraction_probability_ge_0_5": float((ranked.interaction_probability >= 0.5).mean()),
        "panel_length_median": panel.sequence_length_nt.median(), "universe_length_median": ranked.sequence_length_nt.median(),
        "panel_gc_median": panel.gc_fraction.median(), "universe_gc_median": ranked.gc_fraction.median(),
        "panel_gc_percentile_median": float(ranked.gc_fraction.rank(pct=True)[ranked.in_panel].median()),
        "margin_gc_spearman": float(ranked.logit_margin.corr(ranked.gc_fraction, method="spearman")),
        "panel_mean_log2_tpm1_median": panel.mean_log2_tpm1.median(),
        "universe_mean_log2_tpm1_median": ranked.mean_log2_tpm1.median(),
    }


def main() -> None:
    setup_logging()
    config = load_config()
    scores = load_scores(config["checkpoint"])
    universes = pd.read_parquet(OUT_TABLES / "universes.parquet")

    rankings, summary = [], []
    for rbp, spec in config["rbps"].items():
        universe = universes.loc[universes.rbp.eq(rbp)]
        protein_scores = scores.loc[scores.Protein_id.eq(spec["protein_id"])].drop(columns="Protein_id")
        merged = universe.merge(protein_scores, on="RNA_id", how="left", validate="many_to_one")
        if merged.logit_margin.isna().any():
            raise ValueError(f"{rbp}: {int(merged.logit_margin.isna().sum())} universe transcripts have no prediction")
        ranked = rank_universe(merged, config["panel"]["top_fraction"])
        rankings.append(ranked)
        summary.append(describe(rbp, ranked, spec["gene_id"]))
        LOGGER.info("%s: panel of %d transcripts (%d genes); median GC %.3f vs %.3f in the universe", rbp,
                    summary[-1]["panel_transcripts"], summary[-1]["panel_genes"], summary[-1]["panel_gc_median"],
                    summary[-1]["universe_gc_median"])

    write_table(pd.concat(rankings, ignore_index=True), OUT_TABLES / "rankings.parquet")
    write_table(pd.DataFrame(summary), OUT_TABLES / "panel_summary.tsv")


if __name__ == "__main__":
    main()
