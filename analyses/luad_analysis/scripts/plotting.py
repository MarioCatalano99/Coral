"""Figure style of the supplementary figures (s4-s8), and the one figure writer every figure goes through.

Colors follow the dataviz reference palette (light mode). The three RBPs take categorical slots
1-3, which are validated as an all-pairs set; two-level ordinal groups use the single-hue blue ramp.
The main figure (steps/figure_main.py) keeps its own print-size style and imports only the Agg-backed
pyplot and save_svg_jpg from here.

Every figure is written as editable SVG + 400-dpi JPEG under <figures dir>/{svg,jpg}/ (save_svg_jpg).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b4b2a9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])

SVG_HASH_SALT = "coral-luad"  # any fixed string; see save_svg_jpg


def rbp_colors(rbps: list[str]) -> dict[str, str]:
    """Fixed color per RBP in config order, so every figure gives an RBP the same hue."""
    return dict(zip(rbps, SERIES))


def apply_style() -> None:
    """Set a quiet matplotlib style: light surface, muted axes, no top/right spines."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def save_svg_jpg(fig: plt.Figure, directory: Path, stem: str) -> None:
    """Write <directory>/svg/<stem>.svg (editable text) and <directory>/jpg/<stem>.jpg at 400 dpi, then close.

    Used for the main and supplementary figures alike, so they share one output format. The SVG has no date
    stamp and a fixed ID salt, so redrawing unchanged data gives an identical file.
    """
    for kind, options in (("svg", {"metadata": {"Date": None}}), ("jpg", {"pil_kwargs": {"quality": 95, "subsampling": 0}})):
        path = directory / kind / f"{stem}.{kind}"
        path.parent.mkdir(parents=True, exist_ok=True)
        # matplotlib salts SVG element IDs with a fresh uuid4 unless svg.hashsalt is set
        with plt.rc_context({"svg.hashsalt": SVG_HASH_SALT}):
            fig.savefig(path, dpi=400, bbox_inches="tight", **options)
    logging.getLogger("luad_stage").info("wrote %s/{svg,jpg}/%s", directory.name, stem)
    plt.close(fig)


def default_annotation(controls: dict[str, str]) -> Callable[[dict], str]:
    """Right-margin text per forest row: BH q (primary rows only) and the competitive P for each null."""
    names = list(controls.items())

    def annotate(row: dict) -> str:
        q = f"q = {row['q_bh']:.2g}   " if pd.notna(row.get("q_bh")) else ""
        comp = " / ".join([f"{row[f'competitive_p_{names[0][0]}']:.2g}"]
                          + [f"{row[f'competitive_p_{c}']:.2g} ({legend})" for c, legend in names[1:]])
        return f"{q}comp. P = {comp}"
    return annotate


def forest_axes(ax: plt.Axes, table: pd.DataFrame, labels: list[str], controls: dict[str, str], colors: str | list[str],
                panel_label: str, annotation: Callable[[dict], str], fontsize: float = 8, legend: bool = True) -> None:
    """Panel estimate and 95% CI per row, drawn over the matched random panels' 95% range (one band per null).

    table needs rho, rho_low, rho_high and, per control, null_low_/null_high_<control>; annotation(row) gives the
    right-margin text. colors is one color or one per row (then the legend marker is neutral ink).
    """
    y = np.arange(len(table))[::-1].astype(float)
    row_colors = [colors] * len(table) if isinstance(colors, str) else list(colors)
    offsets = np.linspace(0.12, -0.12, len(controls)) if len(controls) > 1 else [0.0]
    handles = []
    for shift, alpha, (control, legend_label) in zip(offsets, (0.9, 0.45), controls.items()):
        ax.hlines(y + shift, table[f"null_low_{control}"], table[f"null_high_{control}"], color=NEUTRAL, alpha=alpha,
                  linewidth=5)
        handles.append(Line2D([], [], color=NEUTRAL, alpha=alpha, linewidth=5,
                              label=f"random panels, 95% range ({legend_label})"))
    ax.hlines(y, table.rho_low, table.rho_high, color=row_colors, linewidth=2)
    ax.scatter(table.rho, y, s=36, color=row_colors, zorder=3, edgecolors="white", linewidths=0.8)
    marker = row_colors[0] if len(set(row_colors)) == 1 else INK_SECONDARY
    handles.append(Line2D([], [], marker="o", linestyle="none", color=marker, markeredgecolor="white",
                          label=f"{panel_label}, 95% CI"))
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_yticks(y, labels)
    for yi, row in zip(y, table.to_dict("records")):
        ax.text(1.02, yi, annotation(row), transform=ax.get_yaxis_transform(), va="center", fontsize=fontsize,
                color=INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    if legend:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=fontsize - 0.5, ncol=1)


def forest(table: pd.DataFrame, labels: list[str], controls: dict[str, str], color: str, panel_label: str,
           xlabel: str) -> plt.Figure:
    """A standalone forest figure (see forest_axes). The caller saves it with save_svg_jpg."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 0.45 * len(table) + 1.3), constrained_layout=True)
    forest_axes(ax, table, labels, controls, color, panel_label, default_annotation(controls))
    ax.set_xlabel(xlabel)
    return fig
