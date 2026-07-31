"""Render Figure 6 panels for the CORAL paper (RBD_analysis result).

  a  exemplar protein attention tracks with RBD intervals marked
  b  per-protein Δ distribution against the position-shuffle null
  c  per-family Cohen's d forest plot
  d  per-protein Cohen's d ECDF

The unit of analysis is the protein (per-protein consensus profile). Reads
artifacts from `analyses/RBD_analysis/results/<model>/heads_best/`; writes 4
PDF + SVG + JPEG figures into `<results_dir>/figures/`.

Run from repo root with the `encodersenv` conda env active:
    python analyses/RBD_analysis/scripts/plot_figure6.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Make the repo root importable so we can use `coral.*` regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))  # no-op once `pip install -e .` has been run

from coral.paths import analysis_results  # noqa: E402
from coral.rbd import load_rbd_intervals  # noqa: E402
from coral.stats import cohens_d_per_unit  # noqa: E402

log = logging.getLogger("plot_figure6")


# --- Style + palette ------------------------------------------------------- #

PALETTE = {
    "observed": "#2563eb",
    "null": "#94a3b8",
    "pos": "#059669",
    "neg": "#dc2626",
    "rbd_bar": "#2563eb",
    "pad_bg": "#f0f0f0",
    "median": "#1e3a8a",
    "zero": "#64748b",
}

FAMILY_DISPLAY = {
    "RRM_1": "RRM",
    "KH_1": "KH",
    "zf-CCCH": "CCCH-ZnF",
    "zf-C2H2": "C2H2-ZnF",
    "dsrm": "dsRBD",
}

EXEMPLARS = [
    ("Q92879", "CELF1", "RRM"),
    ("Q9BYJ9", "YTHDF1", "YTH"),
    ("P22893", "ZFP36",  "zf-CCCH"),
    ("O95793", "STAU1",  "dsRBD"),
    ("Q92945", "KHSRP",  "KH"),
]

MM = 1.0 / 25.4  # mm -> inches


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    # Warn if Arial is not actually installed (Linux fallback to DejaVu Sans).
    from matplotlib.font_manager import findfont, FontProperties
    resolved = Path(findfont(FontProperties(family="Arial"))).name.lower()
    if "arial" not in resolved and "helvetica" not in resolved:
        log.info("Arial/Helvetica not found; falling back to %s.", resolved)


def warm_cmap():
    """White -> deep crimson sequential map; NaN cells in pad_bg gray."""
    cmap = LinearSegmentedColormap.from_list(
        "warm_attn",
        ["#ffffff", "#fff0e0", "#fdbb84", "#e34a33", "#7f0000"],
    )
    cmap = cmap.copy()
    cmap.set_bad(PALETTE["pad_bg"])
    return cmap


# --- Loading --------------------------------------------------------------- #

@dataclass
class Artifacts:
    stats: pd.DataFrame
    null: pd.DataFrame
    families: pd.DataFrame
    npz: dict                       # {"protein_ids", "attentions", "masks"}


def _load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "protein_ids": z["protein_ids"],
        "attentions": z["attentions"],
        "masks": z["masks"],
    }


def load_artifacts(results_dir: Path) -> Artifacts:
    sub = results_dir / "heads_best"
    return Artifacts(
        stats=pd.read_csv(sub / "rbd_attention_stats.csv"),
        null=pd.read_csv(sub / "rbd_null_distribution.csv"),
        families=pd.read_csv(sub / "rbd_family_stratified.csv"),
        npz=_load_npz(sub / "rbd_protein_vectors.npz"),
    )


# --- Panel A: exemplar protein heatmap ------------------------------------- #

def panel_a(intervals_dict: dict, npz: dict) -> plt.Figure:
    protein_ids = np.asarray(npz["protein_ids"])
    attentions = npz["attentions"]

    rows = []
    for pid, gene, family_label in EXEMPLARS:
        idx = np.where(protein_ids == pid)[0]
        if len(idx) == 0:
            log.warning("Panel A: %s missing from protein_ids; skipping.", pid)
            continue
        vec = np.asarray(attentions[idx[0]], dtype=np.float32)
        intervals = intervals_dict.get(pid, [])
        rows.append({"pid": pid, "gene": gene, "family": family_label,
                     "vec": vec, "intervals": intervals, "length": len(vec)})

    if not rows:
        raise RuntimeError("No exemplar proteins found in NPZ.")

    max_len = max(r["length"] for r in rows)

    # 5 stacked tracks + colorbar lane; small extra row at the very bottom
    # for the shared x-axis label.
    fig = plt.figure(figsize=(180 * MM, 80 * MM))
    gs = fig.add_gridspec(
        nrows=len(rows), ncols=2,
        width_ratios=[60, 1.2],
        height_ratios=[1] * len(rows),
        hspace=0.95, wspace=0.04,
        left=0.18, right=0.95, top=0.92, bottom=0.13,
    )

    cmap = warm_cmap()
    im = None
    for i, r in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        # Min-max normalize per protein.
        v = r["vec"]
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        if vmax > vmin:
            v_norm = (v - vmin) / (vmax - vmin)
        else:
            v_norm = np.zeros_like(v)
        # Pad to max_len with NaN; cmap.set_bad renders NaN as pad_bg.
        padded = np.full((1, max_len), np.nan, dtype=np.float32)
        padded[0, :r["length"]] = v_norm
        masked = np.ma.masked_invalid(padded)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                       interpolation="nearest", extent=[0, max_len, 0, 1])

        # RBD bars just above the track. Annotate the family on the first bar.
        bar_y = 1.05
        bar_h = 0.25
        for j, (s, e) in enumerate(r["intervals"]):
            # 1-based inclusive -> 0-based half-open.
            s0, e0 = max(0, s - 1), min(r["length"], e)
            if e0 <= s0:
                continue
            ax.add_patch(Rectangle(
                (s0, bar_y), e0 - s0, bar_h,
                facecolor=PALETTE["rbd_bar"], edgecolor="none",
                clip_on=False, transform=ax.transData,
            ))
            if j == 0:
                ax.text((s0 + e0) / 2, bar_y + bar_h + 0.05, r["family"],
                        fontsize=6.5, ha="center", va="bottom",
                        color=PALETTE["rbd_bar"], clip_on=False)

        # Track frame: keep only bottom spine; remove ticks except on bottom-most.
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max_len)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.grid(False)

        if i < len(rows) - 1:
            ax.tick_params(axis="x", labelbottom=False, length=0)
        else:
            ax.set_xticks(np.arange(0, max_len + 1, 100))
            ax.tick_params(axis="x", length=2)
            ax.set_xlabel("Residue position")

        # Left label: "GENE (family) — L=N aa"
        ax.set_ylabel(
            f"{r['gene']} ({r['family']})\nL={r['length']} aa",
            rotation=0, ha="right", va="center", labelpad=8, fontsize=8,
        )

    # Shared colorbar.
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Per-protein normalized attention", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    fig.text(0.01, 0.96, "A", fontsize=12, fontweight="bold")
    return fig


# --- Panel B: Δ distribution with shuffle null as a band ------------------- #

def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10_000,
                       alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = values[idx].mean(axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def _sci_inner(value: float) -> str:
    """Compact mathtext for a scientific value, without $ delimiters."""
    if value == 0 or abs(value) < 1e-300:
        return "0"
    sign = "-" if value < 0 else ""
    val = abs(value)
    exp = int(np.floor(np.log10(val)))
    coeff = val / 10 ** exp
    if abs(coeff - round(coeff)) < 0.05:
        coeff_str = f"{round(coeff):.0f}"
    else:
        coeff_str = f"{coeff:.1f}"
    return rf"{sign}{coeff_str}{{\times}}10^{{{exp}}}"


_SCI_FORMATTER = FuncFormatter(lambda v, _pos: f"${_sci_inner(v)}$" if v != 0 else "0")


def panel_b(stats: pd.DataFrame, null: pd.DataFrame) -> plt.Figure:
    """Single-axis Δ distribution: observed histogram with the position-shuffle
    null drawn as a shaded 95% band at 0.

    The null is ~30× narrower than the observed distribution, so on one
    continuous linear x-axis it collapses to a thin band at 0 while the observed
    mass sits well to its right. This replaces the earlier broken-axis layout.
    x-tick labels use compact scientific notation.
    """
    obs = stats["mean_diff"].to_numpy()
    null_vals = null["shuffled_mean_diff"].to_numpy()
    n_total = len(obs)
    n_pos = int((obs > 0).sum())
    obs_mean = float(obs.mean())
    # Empirical p via the (k+1)/(N+1) convention (Phipson & Smyth 2010): with
    # zero null exceedances this is 1/(N+1) < 1/N, so the label honestly reads
    # "p < 0.001" instead of the floored "= 0.001".
    p_emp = (int((null_vals >= obs_mean).sum()) + 1) / (len(null_vals) + 1)
    ci_lo, ci_hi = _bootstrap_mean_ci(obs)

    fig, ax = plt.subplots(figsize=(85 * MM, 62 * MM))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.30)

    bins_obs = np.histogram_bin_edges(obs, bins="fd")
    if len(bins_obs) < 8:
        bins_obs = np.histogram_bin_edges(obs, bins="auto")

    # --- Observed distribution --- #
    ax.hist(obs, bins=bins_obs, density=True, color=PALETTE["observed"],
            alpha=0.6, edgecolor=PALETTE["observed"], linewidth=0.4)

    # --- Position-shuffle null as a shaded 95% band at 0 --- #
    n_lo, n_hi = np.percentile(null_vals, [2.5, 97.5])
    ax.axvspan(n_lo, n_hi, color=PALETTE["null"], alpha=0.6, linewidth=0)
    ax.axvline(float(null_vals.mean()), color="#475569", linewidth=0.8)
    ax.axvline(0, color=PALETTE["zero"], linewidth=0.5, linestyle=":")

    # --- Observed mean line + Δ annotation + bootstrap CI bar --- #
    ax.axvline(obs_mean, color=PALETTE["observed"], linestyle="--", linewidth=1.0)
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    # Legend sits upper-right, where the right tail is low, so it never overlaps
    # the bars.
    legend_loc = "upper right"
    ax.annotate(rf"$\bar{{\Delta}} = {_sci_inner(obs_mean)}$",
                xy=(obs_mean, 0.55), xycoords=blended,
                xytext=(5, 0), textcoords="offset points",
                fontsize=7, color=PALETTE["observed"], va="center")
    ax.hlines(0.06, ci_lo, ci_hi, transform=blended,
              color=PALETTE["observed"], linewidth=2.5, alpha=0.7)

    # --- Axes cosmetics: one continuous x-axis spanning both distributions --- #
    lo = min(obs.min(), null_vals.min())
    hi = max(obs.max(), null_vals.max())
    span = hi - lo
    ax.set_xlim(lo - 0.03 * span, hi + 0.03 * span)
    ax.set_yticks([])
    ax.set_ylabel("Density", labelpad=4)
    ax.tick_params(axis="x", labelsize=7, length=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(_SCI_FORMATTER)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)

    handles = [
        Patch(facecolor=PALETTE["observed"], alpha=0.6,
              edgecolor=PALETTE["observed"], label="Observed"),
        Patch(facecolor=PALETTE["null"], alpha=0.6, label="Position-shuffle null"),
    ]
    ax.legend(handles=handles, fontsize=7, loc=legend_loc, frameon=False)
    ax.set_title(f"Observed (N = {n_total:,})", fontsize=7,
                 color=PALETTE["observed"], fontweight="bold")

    # --- Shared x-label + significance row --- #
    fig.text(0.5, 0.13, "Δ (inside − outside RBD)",
             ha="center", va="center", fontsize=9)
    p_label = "p < 0.001" if p_emp < 1e-3 else f"p = {p_emp:.3f}"
    fig.text(0.5, 0.04,
             f"{p_label}   |   {n_pos}/{n_total} proteins Δ > 0",
             ha="center", va="center", fontsize=7, color="#1e293b")
    fig.text(0.01, 0.96, "B", fontsize=12, fontweight="bold")
    return fig


# --- Panel C: family forest plot ------------------------------------------- #

def panel_c(families: pd.DataFrame, d_cap: float = 10.0) -> plt.Figure:
    df = families.sort_values("cohens_d", ascending=False).reset_index(drop=True)
    n = len(df)

    height = max(55 * MM, 0.30 * n + 1.4)
    fig, ax = plt.subplots(figsize=(75 * MM, height), constrained_layout=True)

    display_names = [FAMILY_DISPLAY.get(f, f) for f in df["family"]]
    y = np.arange(n)[::-1]  # top family at the top of the axis

    for yi, (_, row) in zip(y, df.iterrows()):
        d = float(row["cohens_d"])
        lo = float(row["d_lo"])
        hi = float(row["d_hi"])
        color = PALETTE["pos"] if d >= 0 else PALETTE["neg"]
        # Cap the visual extent at d_cap; mark truncation with a chevron.
        hi_clip = min(hi, d_cap)
        d_clip = min(d, d_cap)
        ax.hlines(yi, lo, hi_clip, color=color, linewidth=1.4, alpha=0.85)
        ax.scatter([d_clip], [yi], color=color, s=18, zorder=3,
                   edgecolor="white", linewidth=0.4)
        if hi > d_cap:
            ax.text(d_cap, yi, " ›", color=color, fontsize=9,
                    va="center", ha="left")

    ax.axvline(0, color=PALETTE["zero"], linestyle="--", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel("Cohen's d")
    ax.set_xlim(min(-0.5, df["d_lo"].min() - 0.3), d_cap + 0.6)
    ax.grid(axis="x", alpha=0.4)
    ax.grid(axis="y", visible=False)

    # Annotate n right of each row.
    blended = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.text(1.02, yi, f"n={int(row['n_proteins'])}",
                transform=blended, fontsize=6.5, va="center",
                ha="left", color="#475569")

    ax.set_title(f"Per-protein ({n} families)", fontsize=8.5)

    fig.text(0.01, 0.97, "C", fontsize=12, fontweight="bold")
    return fig


# --- Panel D: per-unit Cohen's d ECDF -------------------------------------- #

def _per_protein_cohens_d(npz: dict) -> np.ndarray:
    """Per-protein Cohen's d via the shared `coral.stats` helper.

    Delegates to `cohens_d_per_unit` so this ECDF shows exactly the
    distribution whose mean/median the meta-analysis txt reports. That helper's
    pooled SD is the *unweighted* sqrt((σ²₁ + σ²₀) / 2); do not reintroduce an
    n-weighted pooled variance here — it inflates d (~0.2–0.3, ~5–25% per
    protein on the current caches) because the minority RBD group carries the
    larger variance, and the figure would drift from the reported numbers.
    """
    scores = [np.asarray(v, dtype=np.float64) for v in npz["attentions"]]
    labels = [np.asarray(m) for m in npz["masks"]]
    return np.asarray(cohens_d_per_unit(scores, labels), dtype=np.float64)


def panel_d(npz: dict) -> plt.Figure:
    d = _per_protein_cohens_d(npz)
    n = len(d)
    if n == 0:
        raise RuntimeError("No usable proteins for Cohen's d ECDF.")
    d_sorted = np.sort(d)
    y = np.arange(1, n + 1) / n
    median = float(np.median(d))
    n_pos = int((d > 0).sum())
    pct_pos = 100.0 * n_pos / n

    fig, ax = plt.subplots(figsize=(70 * MM, 60 * MM), constrained_layout=True)
    ax.step(d_sorted, y, where="post", color=PALETTE["observed"], linewidth=1.3)

    ax.axvline(0, color=PALETTE["zero"], linestyle="--", linewidth=0.7)
    ax.axvline(median, color=PALETTE["median"], linestyle="--", linewidth=0.9)
    ax.text(median, 0.04, f" median d = {median:.2f}",
            color=PALETTE["median"], fontsize=7, va="bottom", ha="left",
            rotation=90)

    ax.text(0.03, 0.96,
            f"{n_pos}/{n} proteins\n({pct_pos:.1f}%) d > 0",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=PALETTE["observed"], lw=0.5, alpha=0.9))

    # Rug along the bottom (uniform light blue, low opacity).
    ax.eventplot(d, lineoffsets=-0.025, linelengths=0.025,
                 colors=PALETTE["observed"], alpha=0.18, linewidths=0.4)

    ax.set_xlabel("Cohen's d (per protein)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(-0.06, 1.02)
    ax.set_title(f"Per-protein consensus (N = {n:,})", fontsize=8.5)
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", visible=False)

    fig.text(0.01, 0.96, "D", fontsize=12, fontweight="bold")
    return fig


# --- IO + orchestration ---------------------------------------------------- #

def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    pdf_dir = out_dir / "pdf"
    svg_dir = out_dir / "svg"
    jpg_dir = out_dir / "jpeg"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    jpg_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{stem}.pdf"
    svg_path = svg_dir / f"{stem}.svg"
    jpg_path = jpg_dir / f"{stem}.jpg"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=400)
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(jpg_path, format="jpeg", bbox_inches="tight", dpi=400,
                facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)
    log.info("Wrote %s + .svg", pdf_path.name)


def _try_panel(name: str, fn, *args, **kwargs) -> Optional[plt.Figure]:
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        log.warning("Skipping %s: missing input %s", name, exc)
    except Exception as exc:  # noqa: BLE001 - report and continue
        log.exception("Skipping %s: %s", name, exc)
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override the output directory (default results/<model>/figures).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Silence the noisy per-PDF font-subsetting chatter from matplotlib.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    setup_style()

    results_dir = analysis_results("RBD_analysis")
    if not results_dir.exists():
        log.error("Results directory not found: %s", results_dir)
        return 2
    out_dir = args.out_dir or (results_dir / "figures")
    log.info("Results dir: %s", results_dir)
    log.info("Output dir:  %s", out_dir)

    intervals_path = results_dir / "rbd_annotations.rbd_intervals.tsv"
    intervals_dict = (load_rbd_intervals(intervals_path)
                      if intervals_path.exists() else {})
    if not intervals_dict:
        log.warning("RBD intervals TSV missing or empty at %s", intervals_path)

    try:
        arts = load_artifacts(results_dir)
    except FileNotFoundError as exc:
        log.error("Missing input %s (run the analysis scripts first)", exc)
        return 2

    fig = _try_panel("Panel A", panel_a, intervals_dict, arts.npz)
    if fig is not None:
        save_fig(fig, out_dir, "fig6a_heatmap")

    fig = _try_panel("Panel B", panel_b, arts.stats, arts.null)
    if fig is not None:
        save_fig(fig, out_dir, "fig6b_delta")

    fig = _try_panel("Panel C", panel_c, arts.families)
    if fig is not None:
        save_fig(fig, out_dir, "fig6c_forest")

    fig = _try_panel("Panel D", panel_d, arts.npz)
    if fig is not None:
        save_fig(fig, out_dir, "fig6d_ecdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
