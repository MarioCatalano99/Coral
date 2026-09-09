#!/usr/bin/env python3
"""Render paired Section 8 QC heatmaps with distinct RBP-unavailable cells."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "coral-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, TwoSlopeNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from _analysis_io import RESULTS_ROOT, CONFIG_PATH, provenance, staged_outputs, verify_manifest, load_config


LOGGER = logging.getLogger("coral.luad.expression_context.plot")
def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    """Render and reopen both formats before publishing either figure file."""
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix=".expression-context-", dir=base.parent) as stage:
            for suffix in (".png", ".pdf"):
                temporary = Path(stage) / f"{base.name}{suffix}"
                fig.savefig(temporary, dpi=240 if suffix == ".png" else None, bbox_inches="tight")
                if suffix == ".png":
                    pixels = plt.imread(temporary)
                    if not pixels.size or not np.isfinite(pixels).all():
                        raise ValueError("rendered PNG is empty or non-finite")
                else:
                    data = temporary.read_bytes()
                    if not data.startswith(b"%PDF-") or not data.rstrip().endswith(b"%%EOF"):
                        raise ValueError("rendered PDF is incomplete")
            for suffix in (".png", ".pdf"):
                path = base.with_suffix(suffix)
                os.replace(Path(stage) / path.name, path)
                outputs.append(path)
    finally:
        plt.close(fig)
    return outputs


def paired_matrix(
    changes: pd.DataFrame, rbp_order: list[str], prefix: str
) -> pd.DataFrame:
    """Build patient rows with adjacent normal/tumor columns for each RBP."""
    patients = sorted(changes["patient_id"].unique())
    columns: dict[str, np.ndarray] = {}
    for rbp in rbp_order:
        group = changes.loc[changes["rbp_gene"] == rbp].set_index("patient_id").loc[patients]
        for sample_type in ("normal", "tumor"):
            values = group[f"{sample_type}_{prefix}"].to_numpy(float)
            columns[f"{rbp}\n{sample_type}"] = values
    return pd.DataFrame(columns, index=patients)


def validate_paired_values(changes: pd.DataFrame, config: dict[str, Any]) -> None:
    """Reject missing gates and inconsistent precomputed display values."""
    expected = int(config["expected_samples"]["matched_pairs"])
    if not changes.groupby("rbp_gene")["patient_id"].nunique().eq(expected).all():
        raise ValueError("paired table lacks the expected patients for each RBP")
    if not changes.groupby("patient_id")["rbp_gene"].nunique().eq(3).all():
        raise ValueError("paired table does not share the same patients across RBPs")
    for cohort in ("normal", "tumor"):
        available = changes[f"{cohort}_rbp_available"]
        if available.isna().any() or available.dtype != bool:
            raise ValueError("paired availability flags must be complete booleans")
        columns = [f"{cohort}_{name}" for name in (
            "rbp_gene_tpm", "rbp_log2_tpm", "active_edge_count", "active_edge_fraction",
        )]
        if not np.isfinite(changes[columns].to_numpy(float)).all():
            raise ValueError("paired display values must be finite")
        counts = changes[f"{cohort}_active_edge_count"]
        fraction = changes[f"{cohort}_active_edge_fraction"]
        if not fraction.between(0, 1).all() or not counts.mod(1).eq(0).all():
            raise ValueError("paired active fraction or integer count is invalid")
        if not np.allclose(fraction, counts / changes["candidate_edge_count"], atol=1e-9, rtol=0):
            raise ValueError("paired active fractions disagree with recorded counts")
        if not counts.loc[~available].eq(0).all():
            raise ValueError("paired active counts must be zero when the RBP gate is closed")
        expected_log = np.log2(changes[f"{cohort}_rbp_gene_tpm"] + config["expression_pseudocount"])
        if not np.allclose(changes[f"{cohort}_rbp_log2_tpm"], expected_log, atol=1e-8, rtol=0):
            raise ValueError("paired log expression disagrees with raw TPM")


def draw_heatmap(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    cmap: str | Colormap,
    colorbar_label: str,
    norm: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ytick_step: int = 1,
    annotate: bool = False,
) -> None:
    """Render a labeled heatmap with no plotting dependency beyond Matplotlib."""
    image = axis.imshow(
        frame.to_numpy(float), aspect="auto", interpolation="nearest", cmap=cmap,
        norm=norm, vmin=vmin, vmax=vmax,
    )
    axis.grid(False)
    axis.set_xticks(range(len(frame.columns)), frame.columns)
    indexes = list(range(0, len(frame), ytick_step))
    axis.set_yticks(indexes, [frame.index[index] for index in indexes])
    axis.figure.colorbar(image, ax=axis, label=colorbar_label, fraction=0.046, pad=0.04)
    if annotate:
        for row in range(len(frame)):
            for column in range(len(frame.columns)):
                value = float(frame.iat[row, column])
                color = "white" if value > 0.5 else "black"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7)


def plot_paired_heatmaps(
    changes: pd.DataFrame, config: dict[str, Any], rbp_order: list[str]
) -> plt.Figure:
    """Plot absolute paired RBP proxy expression and active-edge fractions."""
    pseudocount = float(config["expression_pseudocount"])
    expression = paired_matrix(changes, rbp_order, "rbp_log2_tpm")
    available = paired_matrix(changes, rbp_order, "rbp_available").astype(bool)
    active = paired_matrix(changes, rbp_order, "active_edge_fraction")
    # Masking changes only the display; stored inactive states remain explicit zeros.
    active = active.where(available)
    threshold = np.log2(float(config["rbp_tpm_cutoff"]) + pseudocount)
    lower, upper = np.quantile(expression.to_numpy(), config["figures"]["expression_color_quantiles"])
    lower, upper = min(float(lower), threshold - 0.01), max(float(upper), threshold + 0.01)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, 256))
    colors[0] = matplotlib.colors.to_rgba(config["figures"]["rbp_available_zero_color"])
    active_cmap = ListedColormap(colors)
    active_cmap.set_bad(config["figures"]["rbp_unavailable_color"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 12), constrained_layout=True)
    draw_heatmap(
        axes[0],
        expression,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=lower, vcenter=threshold, vmax=upper),
        colorbar_label=f"log2(protein-coding isoform TPM sum + {pseudocount:g})",
        ytick_step=5,
    )
    draw_heatmap(
        axes[1],
        active,
        cmap=active_cmap,
        vmin=0,
        vmax=1,
        colorbar_label="fraction of each RBP’s candidate isoforms available",
        ytick_step=len(active) + 1,
    )
    axes[1].set_yticks([])
    axes[0].set_title("RBP mRNA availability proxy")
    axes[1].set_title("Expression-contextualized candidate edges")
    axes[1].legend(
        handles=[
            Patch(facecolor=config["figures"]["rbp_unavailable_color"],
                  label=f"RBP below {config['rbp_tpm_cutoff']:g} TPM: gate closed"),
            Patch(facecolor=config["figures"]["rbp_available_zero_color"],
                  label=f"RBP at least {config['rbp_tpm_cutoff']:g} TPM; zero targets available"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=9,
    )
    for axis in axes:
        axis.set_xlabel("")
        axis.set_ylabel("Matched patients (stable ID order)" if axis is axes[0] else "")
        axis.tick_params(axis="x", rotation=0)
    fig.suptitle("Paired LUAD expression context — oncofetal-expression QC", fontsize=15)
    return fig




def main() -> None:
    """Verify source tables and publish figures together with current provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = args.output_dir / "run_manifests/expression_context.json"
    analysis = verify_manifest(manifest, args.output_dir)
    if analysis["config"] != config:
        raise ValueError("plot and analysis configurations differ")
    table = args.output_dir / "tables/expression_context_paired_changes.tsv"
    changes = pd.read_csv(table, sep="\t")
    if changes[["patient_id", "Protein_id"]].duplicated().any():
        raise ValueError("duplicate paired-change rows")
    validate_paired_values(changes, config)
    rbp_order = ["IGF2BP1", "IGF2BP3", "LIN28B"]
    plt.style.use("seaborn-v0_8-whitegrid")
    metadata = provenance([args.config, manifest, table], config)
    metadata["software"]["matplotlib"] = matplotlib.__version__
    metadata["definitions"] = {"patient_order": "patient_id ascending; no clustering",
        "expression": "log2(protein-coding isoform TPM sum + pseudocount)",
        "active_fraction": "within each RBP-specific top-1% isoform set; gray means RBP unavailable",
        "role": "paired oncofetal-expression QC; all-tumor conditioned results are in tables"}
    with staged_outputs(args.output_dir, args.output_dir / "run_manifests/expression_context_figures.json", metadata) as stage:
        save_figure(plot_paired_heatmaps(changes, config, rbp_order), stage / "figures/expression_context_paired_heatmaps")


if __name__ == "__main__":
    main()
