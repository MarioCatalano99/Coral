"""Main figure (Figure 5 of the paper).

Panels (display only; every number comes from a step table, except the per-stage medians and their bootstrap CIs
in panel c):
- a (s3): overlap of the top-1% panels for each RBP pair.
- b (s4): the IGF2BP1 panel's stage correlation against its 9,999 matched random panels (histogram).
- c (s4, s6): median IGF2BP1 panel score by stage in IGF2BP1-positive and IGF2BP1-negative tumors.
- d (s5): N and T effects, univariate and conditional, against GC-matched random panels.

Writes figures/main/{svg,jpg}/: one file per panel plus the assembled figure_main.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_FIGURES, OUT_TABLES, STAGES, load_config, setup_logging  # noqa: E402
from plotting import plt, save_svg_jpg  # noqa: E402

MAIN = OUT_FIGURES / "main"
FULL_SIZE = (7.2, 5.4)  # inches; two-column journal width
PANEL_SIZE = (3.6, 2.7)
STREAM = 99  # seed stream [seed, 99, k] for the display-only bootstrap in panel c

NAVY, VERMILLION, AMBER, TEAL = "#255F85", "#D8654A", "#D99B35", "#3B8C88"
INK, SLATE, GRID, HISTOGRAM = "#263238", "#667782", "#D9E1E5", "#AEBCC5"
POSITIVE, NEGATIVE = NAVY, SLATE  # IGF2BP1-positive / -negative tumors
DASHED = (0, (4, 3))


def style() -> None:
    """Print-size style: white surface, ink axes, light grid."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "axes.titleweight": "normal", "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.6, "legend.frameon": False, "lines.linewidth": 1.5, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def format_p(value: float) -> str:
    """P with three decimals, or a one-decimal ×10 power below 0.001."""
    if value < 1e-3:
        exponent = int(np.floor(np.log10(value)))
        superscript = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{value / 10 ** exponent:.1f}×10{superscript}"
    return f"{value:.3f}"


def num(value: float, sign: bool = False) -> str:
    """Two decimals with a typographic minus (and an explicit plus when sign is set)."""
    return f"{value:{'+' if sign else ''}.2f}".replace("-", "−")


# --- Inputs shared by several panels ---

def stage_scores() -> pd.DataFrame:
    """IGF2BP1 panel score of every staged tumor, with IGF2BP1 status (s6)."""
    scores = pd.read_csv(OUT_TABLES / "dependence_scores.tsv", sep="\t")
    return scores.loc[scores.stage.notna()]


def stage_effects(rbp: str) -> dict[bool, pd.Series]:
    """Stage correlation in positive (s4) and negative (s6) tumors, keyed by IGF2BP1 status."""
    s4 = pd.read_csv(OUT_TABLES / "stage_association.tsv", sep="\t").set_index("rbp").loc[rbp]
    s6 = pd.read_csv(OUT_TABLES / "dependence.tsv", sep="\t")
    negative = s6.loc[s6.statistic.eq("stage") & s6.group.eq(f"{rbp}-negative")].iloc[0]
    positive = pd.Series({"rho": s4.spearman_rho, "q_bh": s4.q_bh, "patients": s4.patients})
    return {True: positive, False: negative}


# --- Panels ---

def panel_overlap(fig: Figure | SubFigure) -> None:
    """a: Jaccard overlap of the top-1% transcript sets, within each pair's common universe."""
    table = pd.read_csv(OUT_TABLES / "overlap.tsv", sep="\t")
    pairs = table.loc[table.level.eq("transcript")].reset_index(drop=True)
    ax = fig.subplots()
    y = np.arange(len(pairs))[::-1]
    ax.barh(y, pairs.jaccard, height=0.56, color=[NAVY, AMBER, TEAL], edgecolor="white", linewidth=0.8, zorder=2)
    for yi, row in zip(y, pairs.itertuples()):
        noun = "transcript" if row.shared == 1 else "transcripts"
        ax.text(row.jaccard + 0.012, yi, f"{row.shared} shared {noun}\n{row.fold_enrichment:.1f}× random; "
                f"P {format_p(row.p_hypergeometric)}", ha="left", va="center", fontsize=6.6, color=INK)
    ax.set_yticks(y, pairs.rbps.str.replace("&", "–"))
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("Jaccard overlap of top-1% transcript sets")
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)


def panel_stage_null(fig: Figure | SubFigure, rbp: str) -> None:
    """b: the panel's stage correlation against its matched random panels."""
    s4 = pd.read_csv(OUT_TABLES / "stage_association.tsv", sep="\t").set_index("rbp").loc[rbp]
    null = pd.read_parquet(OUT_TABLES / "null_distribution.parquet")
    null = null.loc[null.rbp.eq(rbp)]
    plain = null.loc[null.control.eq("expression_length"), "spearman_rho"].to_numpy()
    bins = np.histogram_bin_edges(np.concatenate([plain, [s4.spearman_rho]]), bins=40)

    ax = fig.subplots()
    counts, _, _ = ax.hist(plain, bins=bins, color=HISTOGRAM, edgecolor="white", linewidth=0.5,
                           label="Matched random panels")
    ax.set_ylim(0, counts.max() * 1.55)  # the legend sits above the bars
    ax.axvline(np.median(plain), color=SLATE, linestyle=DASHED, linewidth=1.2, label="Random-panel median")
    ax.axvline(s4.spearman_rho, color=VERMILLION, linewidth=2.0,
               label=f"CORAL {rbp} panel\ncompetitive P = {s4.competitive_p_expression_length:.4f}\n"
                     f"(GC-matched: {s4.competitive_p_expression_length_gc:.4f})")
    ax.set_xlabel(f"Spearman ρ with stage, {rbp}-positive tumors")
    ax.set_ylabel("Random panels")
    ax.legend(loc="upper right", handlelength=1.6)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def panel_context_trend(fig: Figure | SubFigure, rbp: str, seed: int, bootstraps: int) -> None:
    """c: median panel score by stage (95% bootstrap CI), both groups on one axis."""
    scores, effects = stage_scores(), stage_effects(rbp)
    ax = fig.subplots()
    for k, (positive, marker, offset) in enumerate(((True, "o", -0.06), (False, "s", 0.06))):
        part = scores.loc[scores.positive.eq(positive)]
        rng = np.random.default_rng([seed, STREAM, k])
        medians, low, high = [], [], []
        for s in STAGES.values():
            values = part.panel_score[part.stage.eq(s)].to_numpy()
            resampled = np.median(values[rng.integers(0, len(values), size=(bootstraps, len(values)))], axis=1)
            medians.append(np.median(values))
            low.append(np.percentile(resampled, 2.5))
            high.append(np.percentile(resampled, 97.5))
        medians, low, high = map(np.asarray, (medians, low, high))
        effect = effects[positive]
        ax.errorbar(np.array(list(STAGES.values())) + offset, medians, yerr=[medians - low, high - medians],
                    fmt=f"{marker}-", color=POSITIVE if positive else NEGATIVE, markersize=4.5, capsize=2.5,
                    linewidth=1.5, label=f"{rbp}-{'positive' if positive else 'negative'} (n = {len(part)}), "
                                         f"ρ = {num(effect.rho, not positive)}")
    counts = scores.groupby(["stage", "positive"]).size().unstack()
    ax.set_xticks(list(STAGES.values()), [f"{name}\n{counts.loc[v, True]} / {counts.loc[v, False]}"
                                          for name, v in STAGES.items()])
    ax.set_xlim(0.6, 4.4)
    ax.set_xlabel("Pathological stage (patients: positive / negative)")
    ax.set_ylabel("Median panel score, log2(TPM + 1)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.25 * (top - bottom))  # room for the legend


def panel_tnm(fig: Figure | SubFigure) -> None:
    """d: univariate and conditional N and T effects over GC-matched random panels."""
    table = pd.read_csv(OUT_TABLES / "tnm_association.tsv", sep="\t").set_index("statistic")
    ax = fig.subplots()
    rows = {"N": (1.0, "N", "N | T"), "T": (0.0, "T", "T | N")}
    for _, (y, univariate, conditional) in rows.items():
        for statistic, marker, offset in ((univariate, "o", 0.15), (conditional, "D", -0.15)):
            row = table.loc[statistic]
            ax.fill_between([row.null_low_expression_length_gc, row.null_high_expression_length_gc],
                            y + offset - 0.07, y + offset + 0.07, color=HISTOGRAM, alpha=0.7, linewidth=0, zorder=1)
            ax.errorbar(row.rho, y + offset, xerr=[[row.rho - row.rho_low], [row.rho_high - row.rho]], fmt=marker,
                        color=VERMILLION, markersize=4.5, capsize=2.5, linewidth=1.5, zorder=3)
            ax.text(1.01, y + offset, f"{row.competitive_p_expression_length_gc:#.2g}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.6, color=INK)
    ax.text(1.01, 1.0, "comp. P", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6, color=SLATE)
    ax.axvline(0, color=SLATE, linestyle=DASHED, linewidth=1)
    ax.set_yticks([1.0, 0.0], ["N", "T"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.6, 0.1)
    ax.set_xlabel("Spearman ρ with panel score (95% CI)")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=VERMILLION, label="Univariate"),
                       Line2D([], [], marker="D", linestyle="none", color=VERMILLION,
                              label="Conditional on the other (N | T, T | N)"),
                       Patch(facecolor=HISTOGRAM, alpha=0.7, label="GC-matched random panels, 95% range")],
              loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)


# --- Assembly and output ---

def letter(fig: Figure | SubFigure, text: str) -> None:
    fig.text(0.0, 1.0, text, ha="left", va="top", fontsize=11, fontweight="bold")


def build(panels: list[tuple[str, str, Callable[[Figure | SubFigure], None]]]) -> None:
    """Save every panel on its own, then the assembled 2 x 2 figure."""
    for name, stem, draw in panels:
        fig = plt.figure(figsize=PANEL_SIZE, layout="constrained")
        draw(fig)
        letter(fig, name)
        save_svg_jpg(fig, MAIN, stem)
    fig = plt.figure(figsize=FULL_SIZE, layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.08])
    for slot, (name, _, draw) in zip([*top.subfigures(1, 2), *bottom.subfigures(1, 2)], panels):
        draw(slot)
        letter(slot, name)
    save_svg_jpg(fig, MAIN, "figure_main")


def main() -> None:
    setup_logging()
    style()
    config = load_config()
    rbp, seed, bootstraps = config["followup"]["rbp"], config["stage"]["seed"], config["stage"]["bootstraps"]
    build([("a", "panel_a_target_overlap", panel_overlap),
           ("b", "panel_b_stage_null", lambda f: panel_stage_null(f, rbp)),
           ("c", "panel_c_stage_trend", lambda f: panel_context_trend(f, rbp, seed, bootstraps)),
           ("d", "panel_d_n_versus_t", panel_tnm)])


if __name__ == "__main__":
    main()
