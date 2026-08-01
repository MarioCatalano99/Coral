"""Render Figure 5 panels for the CORAL paper (contact_analysis result).

Panels a, c, d, e are rendered here. **Panel b is intentionally deferred** — it
is the 3D, attention-colored structure view, which needs PDB-file
identification, sequence→structure contact mapping, and PyMOL rendering, so it
lives in its own standalone script rather than this matplotlib renderer.

  a  three-complex exemplar attention heatmaps with contact stars
  c  per-head contact bias (mean Δ + bootstrap CI), selected heads highlighted
  d  per-complex AUROC ECDF (heads_best vs heads_all), exemplars marked
  e  per-complex Cohen's d ECDF (heads_best vs heads_all)

Reads artifacts from `analyses/contact_analysis/results/<model>/`; writes 4 SVG +
4 JPEG figures into `<results_dir>/figures/{svg,jpeg}/`.

Run from repo root with the `encodersenv` conda env active:
    python analyses/contact_analysis/scripts/plot_figure5.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_auc_score

# Make the repo root importable so we can use `coral.*` regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))  # no-op once `pip install -e .` has been run

from coral.paths import analysis_results  # noqa: E402

log = logging.getLogger("plot_figure5")


# --- Style + palette ------------------------------------------------------- #

PALETTE = {
    "best": "#2563eb",      # heads_best (selected subset)
    "all": "#94a3b8",       # heads_all
    "gold": "#FFD400",      # contact stars / exemplar markers
    "zero": "#64748b",      # reference lines
    "nan": "#f0f0f0",       # masked / NaN cells
}

# (complex_id, gene, rna, fold) — overridable via --exemplars.
EXEMPLARS = [
    ("4QIL|C_A", "Roquin ROQ", "CDE hairpin",  "winged-helix"),
    ("3MOJ|A_B", "YxiN",       "23S rRNA",     "DEAD-box RBD"),
    ("2UWM|C_A", "SelB",       "SECIS hairpin", "Sec-specific EF"),

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


def blue_cmap():
    """White -> deep blue sequential map; NaN cells in light grey."""
    cmap = LinearSegmentedColormap.from_list(
        "blue_attn",
        ["#f7fbff", "#d8e7f5", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
    )
    cmap = cmap.copy()
    cmap.set_bad(PALETTE["nan"])
    return cmap


# --- Loading + shared stats ------------------------------------------------ #

def _load_npz(path: Path) -> dict:
    """Load a contact_vectors.npz. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    return {
        "complex_ids": z["complex_ids"],
        "attentions": z["attentions"],
        "contact_maps": z["contact_maps"],
    }


def per_complex_auroc(npz: dict) -> dict[str, float]:
    """AUROC of attention ranking contact cells above non-contact cells, per complex.

    Flatten the attention matrix (scores) and contact_map>0 (labels) over all
    (RNA-token, residue) cells; AUROC = P(score of a random contact cell > score
    of a random non-contact cell). Complexes with <3 contacts are skipped.
    """
    out: dict[str, float] = {}
    for cid, a, c in zip(npz["complex_ids"], npz["attentions"], npz["contact_maps"]):
        scores = np.asarray(a, float).ravel()
        labels = (np.asarray(c, float).ravel() > 0).astype(int)
        if labels.sum() < 3 or labels.sum() == labels.size:
            continue
        out[str(cid)] = float(roc_auc_score(labels, scores))
    return out


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted values, cumulative fraction) for a step ECDF."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(v) + 1) / len(v)
    return v, y


def _selected_label(selected: list[int]) -> str:
    """Human-readable label for the selected-head subset, e.g. 'heads 0 & 7'."""
    if not selected:
        return "selected heads"
    if len(selected) == 1:
        return f"head {selected[0]}"
    return "heads " + " & ".join(str(h) for h in selected)


def _resolve_exemplars(arg: Optional[str]) -> list[tuple[str, str, str, str]]:
    """Default trio, or CLI override (keeps default labels where the id matches)."""
    if not arg:
        return EXEMPLARS
    by_cid = {e[0]: e for e in EXEMPLARS}
    out: list[tuple[str, str, str, str]] = []
    for cid in (c.strip() for c in arg.split(",")):
        if not cid:
            continue
        out.append(by_cid.get(cid, (cid, cid.split("|")[0], "", "")))
    return out


# --- Panel a: three-complex exemplar heatmaps ------------------------------ #

def panel_a(results_dir: Path, exemplars: list[tuple[str, str, str, str]],
            pad: int = 12) -> plt.Figure:
    npz = _load_npz(results_dir / "heads_best" / "contact_vectors.npz")
    auroc = per_complex_auroc(npz)
    stats = pd.read_csv(results_dir / "heads_best" / "attention_stats.csv")
    pooled = np.sqrt((stats["contact_std"] ** 2 + stats["non_contact_std"] ** 2) / 2.0)
    cohens_d = dict(zip(stats["complex_id"].astype(str),
                        (stats["mean_diff"] / pooled).to_numpy()))
    by_cid = {str(c): (a, m) for c, a, m in
              zip(npz["complex_ids"], npz["attentions"], npz["contact_maps"])}

    rows = []
    for cid, gene, rna, fold in exemplars:
        if cid not in by_cid:
            log.warning("Panel a: %s missing from complex_ids; skipping.", cid)
            continue
        att, cmap_arr = by_cid[cid]
        rows.append({
            "cid": cid, "gene": gene, "rna": rna, "fold": fold,
            "att": np.asarray(att, dtype=np.float32),
            "cmap": np.asarray(cmap_arr, dtype=np.float32),
        })
    if not rows:
        raise RuntimeError("No exemplar complexes found in NPZ.")

    cmap = blue_cmap()
    n = len(rows)
    fig = plt.figure(figsize=(180 * MM, 32 * MM * n + 12 * MM))
    gs = fig.add_gridspec(
        nrows=n, ncols=2, width_ratios=[60, 1.2],
        hspace=0.95, wspace=0.04,
        left=0.16, right=0.93, top=0.90, bottom=0.16,
    )

    im = None
    for i, r in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        att, cm = r["att"], r["cmap"]
        n_rna = att.shape[0]

        # --- Crop the protein (column) axis to the contact-spanning window --- #
        col_has_contact = np.where(cm.sum(axis=0) > 0)[0]
        if col_has_contact.size:
            c0 = max(0, int(col_has_contact.min()) - pad)
            c1 = min(att.shape[1], int(col_has_contact.max()) + pad + 1)
        else:
            c0, c1 = 0, att.shape[1]
        A = att[:, c0:c1]
        C = cm[:, c0:c1]

        # --- Normalize per matrix with a 2nd-98th percentile clip to [0,1] --- #
        lo, hi = np.nanpercentile(A, [2, 98])
        if hi > lo:
            A_norm = np.clip((A - lo) / (hi - lo), 0.0, 1.0)
        else:
            A_norm = np.zeros_like(A)
        masked = np.ma.masked_invalid(A_norm)

        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                       interpolation="nearest", extent=[c0, c1, n_rna, 0])

        # --- Stars at annotated contacts (cell centers) --- #
        rr, cc = np.where(C > 0)
        if rr.size:
            ax.scatter(c0 + cc + 0.5, rr + 0.5, marker="*", s=70,
                       facecolor=PALETTE["gold"], edgecolor="black",
                       linewidth=0.5, zorder=5)

        # --- Axes cosmetics --- #
        ax.set_yticks([])
        ax.set_ylim(n_rna, 0)
        ax.set_xlim(c0, c1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(axis="x", length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.grid(False)

        ylab = f"{r['gene']}\n· {r['rna']}" if r["rna"] else r["gene"]
        ax.set_ylabel(ylab, rotation=0, ha="right", va="center",
                      labelpad=8, fontsize=8)
        if i == n - 1:
            ax.set_xlabel("Protein residue")

        pdb = r["cid"].split("|")[0]
        head = f"{pdb} · {r['fold']}" if r["fold"] else pdb
        n_contact = int((C > 0).sum())
        auc = auroc.get(r["cid"], float("nan"))
        d = cohens_d.get(r["cid"], float("nan"))
        ax.set_title(f"{head}  (AUROC {auc:.2f}, d {d:.2f}, {n_contact} contacts)",
                     loc="left", fontsize=8)

    # --- Shared colorbar --- #
    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Per-matrix normalized attention", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    fig.text(0.01, 0.97, "a", fontsize=12, fontweight="bold")
    fig.text(0.16, 0.045, "★ = annotated structural contact (≤3.4 Å)",
             fontsize=6.5, color="#1e293b")
    fig.text(0.16, 0.012,
             "y-axis: RNA BPE tokens (k-mers), not single nucleotides",
             fontsize=6.5, color="#475569")
    return fig


# --- Panel c: per-head contact bias ---------------------------------------- #

def panel_c(results_dir: Path, n_boot: int = 5000, seed: int = 42) -> plt.Figure:
    df = pd.read_csv(results_dir / "attention_heads_stats.csv")
    sel_path = results_dir / "selected_heads.json"
    selected = (json.loads(sel_path.read_text())["selected"]
                if sel_path.exists() else [])

    # Per-complex × head matrix so bootstrap resamples whole complexes.
    piv = df.pivot_table(index="complex_id", columns="head_id", values="mean_diff")
    heads = list(piv.columns)
    M = piv.to_numpy()
    M = M[~np.isnan(M).any(axis=1)]
    if M.shape[0] == 0:
        raise RuntimeError("No complexes with complete per-head stats.")

    rng = np.random.default_rng(seed)
    n = M.shape[0]
    boot = np.empty((n_boot, M.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = M[idx].mean(axis=0)
    mean = M.mean(axis=0)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)

    heads_arr = np.asarray(heads)
    sel_mask = np.array([h in set(selected) for h in heads])
    yerr = np.vstack([mean - lo, hi - mean])

    fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM), constrained_layout=True)
    ax.axhline(0, color=PALETTE["zero"], linestyle="--", linewidth=0.7)

    if (~sel_mask).any():
        ax.errorbar(heads_arr[~sel_mask], mean[~sel_mask],
                    yerr=yerr[:, ~sel_mask], fmt="o", color=PALETTE["all"],
                    ecolor=PALETTE["all"], elinewidth=1.0, capsize=2,
                    markersize=4, label="other heads")
    if sel_mask.any():
        ax.errorbar(heads_arr[sel_mask], mean[sel_mask],
                    yerr=yerr[:, sel_mask], fmt="o", color=PALETTE["best"],
                    ecolor=PALETTE["best"], elinewidth=1.2, capsize=2,
                    markersize=5, zorder=3, label="selected heads")
        for h, m, h_hi in zip(heads_arr[sel_mask], mean[sel_mask], hi[sel_mask]):
            ax.annotate(f"head {h}", xy=(h, h_hi), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=6.5, color=PALETTE["best"])

    ax.set_xticks(heads)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Mean Δ (contact − non-contact)")
    ax.set_title("Contact bias is head-specialized", fontsize=8.5)
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")

    fig.text(0.01, 0.97, "c", fontsize=12, fontweight="bold")
    return fig


# --- Panel d: per-complex AUROC ECDF --------------------------------------- #

def panel_d(results_dir: Path,
            exemplars: list[tuple[str, str, str, str]]) -> plt.Figure:
    npz_best = _load_npz(results_dir / "heads_best" / "contact_vectors.npz")
    npz_all = _load_npz(results_dir / "heads_all" / "contact_vectors.npz")
    sel_path = results_dir / "selected_heads.json"
    selected = (json.loads(sel_path.read_text())["selected"]
                if sel_path.exists() else [])

    auroc_best = per_complex_auroc(npz_best)
    auroc_all = per_complex_auroc(npz_all)
    vb = np.array(list(auroc_best.values()))
    va = np.array(list(auroc_all.values()))
    if vb.size == 0:
        raise RuntimeError("No complexes with usable AUROC.")

    fig, ax = plt.subplots(figsize=(75 * MM, 65 * MM), constrained_layout=True)

    xa, ya = _ecdf(va)
    ax.step(xa, ya, where="post", color=PALETTE["all"], linewidth=1.3,
            label="all 12 heads")
    xb, yb = _ecdf(vb)
    ax.step(xb, yb, where="post", color=PALETTE["best"], linewidth=1.5,
            label=_selected_label(selected))

    ax.axvline(0.5, color=PALETTE["zero"], linestyle="--", linewidth=0.7)

    # Mark exemplars on the heads_best curve. They tend to cluster at high
    # AUROC, so cascade the labels downward to avoid overlap.
    drawn = 0
    for cid, *_ in exemplars:
        a = auroc_best.get(cid)
        if a is None:
            continue
        height = float(np.searchsorted(xb, a, side="right")) / len(xb)
        ax.scatter([a], [height], marker="o", s=28, facecolor=PALETTE["gold"],
                   edgecolor="black", linewidth=0.5, zorder=5)
        ax.annotate(cid.split("|")[0], xy=(a, height), xytext=(5, -6 - 11 * drawn),
                    textcoords="offset points", fontsize=6.5, color="#1e293b")
        drawn += 1

    median = float(np.median(vb))
    pct = 100.0 * float((vb > 0.5).mean())
    ax.set_title(f"median {median:.2f}; {pct:.0f}% > chance", fontsize=8.5)
    ax.set_xlabel("Per-complex AUROC")
    ax.set_ylabel("Cumulative fraction of complexes")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")

    fig.text(0.01, 0.97, "d", fontsize=12, fontweight="bold")
    return fig


# --- Panel e: per-complex Cohen's d ECDF ----------------------------------- #

def _per_complex_cohens_d(stats: pd.DataFrame) -> np.ndarray:
    """Per-complex Cohen's d with the *unweighted* pooled SD.

    d = mean_diff / sqrt((contact_std² + non_contact_std²) / 2). This matches
    `coral.stats.cohens_d_per_unit` and the reported meta-analysis numbers; do
    not switch to an n-weighted pooled variance — that inflates d and drifts
    the figure away from the reported median.
    """
    pooled = np.sqrt((stats["contact_std"] ** 2 + stats["non_contact_std"] ** 2) / 2.0)
    d = stats["mean_diff"] / pooled
    return d.to_numpy()[np.isfinite(d.to_numpy())]


def panel_e(results_dir: Path) -> plt.Figure:
    sel_path = results_dir / "selected_heads.json"
    selected = (json.loads(sel_path.read_text())["selected"]
                if sel_path.exists() else [])
    db = _per_complex_cohens_d(pd.read_csv(results_dir / "heads_best" / "attention_stats.csv"))
    da = _per_complex_cohens_d(pd.read_csv(results_dir / "heads_all" / "attention_stats.csv"))
    if db.size == 0:
        raise RuntimeError("No complexes with usable Cohen's d.")

    fig, ax = plt.subplots(figsize=(75 * MM, 65 * MM), constrained_layout=True)

    xa, ya = _ecdf(da)
    ax.step(xa, ya, where="post", color=PALETTE["all"], linewidth=1.3,
            label="all 12 heads")
    xb, yb = _ecdf(db)
    ax.step(xb, yb, where="post", color=PALETTE["best"], linewidth=1.5,
            label=_selected_label(selected))

    ax.axvline(0.0, color=PALETTE["zero"], linestyle="--", linewidth=0.7)

    med_b, med_a = float(np.median(db)), float(np.median(da))
    pos_b = 100.0 * float((db > 0).mean())
    pos_a = 100.0 * float((da > 0).mean())
    ax.set_title(f"median d {med_b:.2f} vs {med_a:.2f}; "
                 f"{pos_b:.0f}% vs {pos_a:.0f}% d > 0", fontsize=8.0)
    ax.set_xlabel("Per-complex Cohen's d")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(-1.2, 1.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.4)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")

    fig.text(0.01, 0.97, "e", fontsize=12, fontweight="bold")
    return fig


# --- IO + orchestration ---------------------------------------------------- #

def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    svg_dir = out_dir / "svg"
    jpeg_dir = out_dir / "jpeg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    svg_path = svg_dir / f"{stem}.svg"
    jpeg_path = jpeg_dir / f"{stem}.jpeg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(jpeg_path, format="jpeg", dpi=400, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)
    log.info("Wrote %s + .jpeg", svg_path.name)


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
    parser.add_argument("--exemplars", default=None,
                        help="Comma-separated complex ids for Panel a (default trio).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Silence the noisy per-figure font-subsetting chatter from matplotlib.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    setup_style()

    results_dir = analysis_results("contact_analysis")
    if not results_dir.exists():
        log.error("Results directory not found: %s", results_dir)
        return 2
    out_dir = args.out_dir or (results_dir / "figures")
    exemplars = _resolve_exemplars(args.exemplars)
    log.info("Results dir: %s", results_dir)
    log.info("Output dir:  %s", out_dir)

    fig = _try_panel("Panel a", panel_a, results_dir, exemplars)
    if fig is not None:
        save_fig(fig, out_dir, "fig5a_heatmap")

    fig = _try_panel("Panel c", panel_c, results_dir)
    if fig is not None:
        save_fig(fig, out_dir, "fig5c_headspec")

    fig = _try_panel("Panel d", panel_d, results_dir, exemplars)
    if fig is not None:
        save_fig(fig, out_dir, "fig5d_auroc_ecdf")

    fig = _try_panel("Panel e", panel_e, results_dir)
    if fig is not None:
        save_fig(fig, out_dir, "fig5e_cohend_ecdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
