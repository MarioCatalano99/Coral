"""Step 4: does expression of each RBP's CORAL panel vary with AJCC pathological stage?

- Patients: the RBP-positive tumors with a known pathological stage (I-IV, substages collapsed).
- Score: log2(median panel TPM + 1) per patient.
- Test: Spearman rho with ordinal stage; two-sided P from 99,999 stage permutations; 95% CI from
  2,000 patient bootstraps; BH across the three RBPs.
- Specificity: 9,999 random panels drawn from the same universe, matched on cohort-mean expression
  decile x length quintile, scored in the same patients; competitive P on |rho|.
- GC robustness: the same, with strata that also match GC-content quintile, because CORAL's IGF2BP
  panels are GC-rich and RNA-seq GC bias varies between samples.

Writes tables/stage_association.tsv, tables/patient_scores.tsv, tables/null_distribution.parquet and
figures/Supplementary/{svg,jpg}/SupplementaryFigure{3,4} (stage scores; null distributions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (CONTROLS, LOGGER, OUT_SUPPLEMENTARY, OUT_TABLES, STAGES, bh, bootstrap_rho,  # noqa: E402
                    competitive_p, load_config, load_log_expression, null_panels, panel_scores, permutation_p,
                    row_spearman, setup_logging, stage_table, to_tpm, write_table)
from plotting import INK_SECONDARY, NEUTRAL, apply_style, rbp_colors, save_svg_jpg  # noqa: E402


def analyse(rbp: str, index: int, ranked: pd.DataFrame, cohort: pd.DataFrame, stages: pd.DataFrame,
            settings: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Association, bootstrap CI and both matched-panel controls for one RBP."""
    patients = cohort.merge(stages, on="patient_id", how="inner", validate="one_to_one").reset_index(drop=True)
    LOGGER.info("%s: %d positive tumors, %d with known stage", rbp, len(cohort), len(patients))
    universe = ranked.sort_values("transcript_id").reset_index(drop=True)
    tpm = to_tpm(load_log_expression(patients.sample_id.tolist(), universe.transcript_id.tolist()).to_numpy())
    panel = np.flatnonzero(universe.in_panel.to_numpy())
    stage = patients.stage.to_numpy(float)
    seed = settings["seed"]

    score = panel_scores(tpm, panel)[0]
    rho = float(row_spearman(score, stage)[0])
    low, high, usable = bootstrap_rho(score, stage, settings["bootstraps"], np.random.default_rng([seed, index, 1]))
    row = {"rbp": rbp, "patients": len(patients),
           **{f"stage_{name}": int((stage == value).sum()) for name, value in STAGES.items()},
           "panel_transcripts": len(panel), "spearman_rho": rho, "rho_low": low, "rho_high": high, "bootstraps_used": usable,
           "p": permutation_p(score, stage, settings["permutations"], np.random.default_rng([seed, index, 2]))}

    nulls = []
    for control in CONTROLS:
        null = null_panels(universe, panel, settings, index, control)
        rho_null = row_spearman(panel_scores(tpm, null), stage)
        row[f"competitive_p_{control}"] = competitive_p(rho, rho_null)
        row[f"null_median_rho_{control}"] = float(np.median(rho_null))
        row[f"null_median_gc_{control}"] = float(np.median(universe.gc_fraction.to_numpy()[null].mean(axis=1)))
        nulls.append(pd.DataFrame({"rbp": rbp, "control": control, "panel": np.arange(len(rho_null)), "spearman_rho": rho_null}))
    row["panel_mean_gc"] = float(universe.gc_fraction.to_numpy()[panel].mean())
    LOGGER.info("%s: rho %.3f [%.3f, %.3f], P=%.3g, competitive P %.3g (expression x length), %.3g (+ GC)", rbp, rho, low,
                high, row["p"], row["competitive_p_expression_length"], row["competitive_p_expression_length_gc"])
    scores = patients.assign(rbp=rbp, panel_score=score)[["rbp", "sample_id", "patient_id", "rbp_coding_tpm", "stage", "panel_score"]]
    return row, scores, pd.concat(nulls, ignore_index=True)


# --- Figures (display only; every number shown comes from the tables) ---

def figures(results: pd.DataFrame, scores: pd.DataFrame, null: pd.DataFrame, rbps: list[str]) -> None:
    apply_style()
    colors = rbp_colors(rbps)
    jitter = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(rbps), figsize=(3.2 * len(rbps), 2.9), constrained_layout=True)
    for ax, row in zip(axes, results.itertuples()):
        group = scores.loc[scores.rbp.eq(row.rbp)]
        values = [group.panel_score[group.stage.eq(s)].to_numpy() for s in STAGES.values()]
        ax.boxplot(values, positions=list(STAGES.values()), widths=0.5, showfliers=False,
                   medianprops={"color": INK_SECONDARY, "linewidth": 1.5}, boxprops={"color": NEUTRAL},
                   whiskerprops={"color": NEUTRAL}, capprops={"color": NEUTRAL})
        ax.scatter(group.stage + jitter.uniform(-0.15, 0.15, len(group)), group.panel_score, s=10,
                   color=colors[row.rbp], alpha=0.7, linewidths=0)
        ax.set_xticks(list(STAGES.values()), [f"{name}\n(n={count})" for name, count in
                                               zip(STAGES, [getattr(row, f"stage_{name}") for name in STAGES])])
        ax.set_title(row.rbp)
        ax.text(0.97, 0.97, f"ρ = {row.spearman_rho:.2f} [{row.rho_low:.2f}, {row.rho_high:.2f}]\nq = {row.q_bh:.2g}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel("AJCC pathological stage")
        ax.grid(axis="x", visible=False)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.2 * (top - bottom))  # headroom for the annotation
    axes[0].set_ylabel("Panel score, log2(median TPM + 1)")
    save_svg_jpg(fig, OUT_SUPPLEMENTARY, "SupplementaryFigure3")  # stage scores

    titles = {"expression_length": "matched on expression × length", "expression_length_gc": "also matched on GC"}
    fig, axes = plt.subplots(len(CONTROLS), len(rbps), figsize=(3.2 * len(rbps), 2.4 * len(CONTROLS)), constrained_layout=True)
    for (r, control), (c, row) in ((rc, cr) for rc in enumerate(CONTROLS) for cr in enumerate(results.itertuples())):
        ax = axes[r, c]
        values = null.loc[null.rbp.eq(row.rbp) & null.control.eq(control), "spearman_rho"]
        ax.hist(values, bins=60, color=NEUTRAL, edgecolor="none")
        ax.axvline(row.spearman_rho, color=colors[row.rbp], linewidth=2)
        right = row.spearman_rho < values.median()
        ax.text(0.97 if right else 0.03, 0.97, f"competitive P = {getattr(row, f'competitive_p_{control}'):.2g}",
                transform=ax.transAxes, va="top", ha="right" if right else "left", fontsize=8, color=INK_SECONDARY)
        ax.set_title(f"{row.rbp}: {titles[control]}", fontsize=9)
        ax.set_xlabel("Spearman ρ with stage (random panels)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)  # headroom for the annotation
    for ax in axes[:, 0]:
        ax.set_ylabel("Random panels")
    save_svg_jpg(fig, OUT_SUPPLEMENTARY, "SupplementaryFigure4")  # null distributions


def main() -> None:
    setup_logging()
    config = load_config()
    settings, rbps = config["stage"], list(config["rbps"])
    if "--figures-only" in sys.argv[1:]:
        figures(pd.read_csv(OUT_TABLES / "stage_association.tsv", sep="\t"), pd.read_csv(OUT_TABLES / "patient_scores.tsv", sep="\t"),
                pd.read_parquet(OUT_TABLES / "null_distribution.parquet"), rbps)
        return
    stages = stage_table()
    cohorts = pd.read_csv(OUT_TABLES / "cohorts.tsv", sep="\t")
    rankings = pd.read_parquet(OUT_TABLES / "rankings.parquet")

    rows, scores, nulls = [], [], []
    for index, rbp in enumerate(rbps):
        cohort = cohorts.loc[cohorts.rbp.eq(rbp) & cohorts.positive, ["sample_id", "patient_id", "rbp_coding_tpm"]]
        row, patient_scores, null = analyse(rbp, index, rankings.loc[rankings.rbp.eq(rbp)], cohort, stages, settings)
        rows.append(row)
        scores.append(patient_scores)
        nulls.append(null)

    results = pd.DataFrame(rows)
    results.insert(results.columns.get_loc("p") + 1, "q_bh", bh(results.p.to_numpy()))
    results.insert(results.columns.get_loc("q_bh") + 1, "significant", results.q_bh < settings["alpha"])
    scores, null = pd.concat(scores, ignore_index=True), pd.concat(nulls, ignore_index=True)
    write_table(results, OUT_TABLES / "stage_association.tsv")
    write_table(scores, OUT_TABLES / "patient_scores.tsv")
    write_table(null, OUT_TABLES / "null_distribution.parquet")
    figures(results, scores, null, rbps)


if __name__ == "__main__":
    main()
