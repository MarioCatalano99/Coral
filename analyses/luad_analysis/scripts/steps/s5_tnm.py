"""Step 5: is the IGF2BP1 stage gradient carried by pathological N, by T, or by both?

- Patients: the IGF2BP1-positive tumors with known N and T. Score: the s4 panel score, log2(median panel TPM + 1).
- Primary family (BH over 4): Spearman rho of the score with N and with T, plus the partial effects N | T and
  T | N (rank residuals). P from 99,999 permutations: plain label permutations for the univariate effects,
  Freedman-Lane for the partial ones. 95% CIs from 2,000 patient bootstraps, the same resamples for all four.
- Specificity: each statistic is recomputed for s4's 9,999 matched random panels (both controls), which gives a
  competitive P for each.

Reading, fixed before running:
- N-driven: the N | T CI excludes 0 and the T | N CI includes 0. T-driven: the reverse. Both: both CIs exclude 0.
  Not separable: neither does.
- CORAL-specific part: the competitive P of each partial effect.

Descriptive only, no P: rho(N, T).

T is the only tumor-size proxy (GDC records no size in cm). Its size cut-offs changed across the AJCC 5th-7th
editions mixed in this cohort, while N's did not.

Writes tables/tnm_association.tsv, tables/tnm_patients.tsv and
figures/Supplementary/{svg,jpg}/SupplementaryFigure7 (scores by N and T).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (LOGGER, OUT_SUPPLEMENTARY, OUT_TABLES, bh, effect, load_config, partial_spearman,  # noqa: E402
                    row_spearman, scores_with_nulls, setup_logging, tnm_table, write_table)
from plotting import BLUE_RAMP, INK_SECONDARY, NEUTRAL, apply_style, save_svg_jpg  # noqa: E402

STREAM = 5  # seed stream [seed, 5, k]; s4 uses [seed, 0-2, k]
TERMS = {  # statistic -> (variable of interest, controls)
    "N": ("n_category", []),
    "T": ("t_category", []),
    "N | T": ("n_category", ["t_category"]),
    "T | N": ("t_category", ["n_category"]),
}


def associations(data: pd.DataFrame, score: np.ndarray, nulls: dict[str, np.ndarray], settings: dict) -> pd.DataFrame:
    """The primary family (4 effects) and rho(N, T), one row each."""
    seed, n = settings["seed"], len(data)
    indices = np.random.default_rng([seed, STREAM, 1]).integers(0, n, size=(settings["bootstraps"], n))
    rows = []
    for k, (term, (interest, controls)) in enumerate(TERMS.items()):
        row, _ = effect(score, data[interest].to_numpy(float), data[controls].to_numpy(float), nulls,
                        indices, settings["permutations"], np.random.default_rng([seed, STREAM, 2 + k]))
        rows.append({"family": "primary", "statistic": term, **row})
        LOGGER.info("%s: rho %.3f [%.3f, %.3f], P=%.3g, competitive P %.3g / %.3g", term, row["rho"], row["rho_low"],
                    row["rho_high"], row["p"], row["competitive_p_expression_length"],
                    row["competitive_p_expression_length_gc"])
    results = pd.DataFrame(rows)
    results.insert(results.columns.get_loc("p") + 1, "q_bh", bh(results.p.to_numpy()))

    n_t = float(row_spearman(data.n_category.to_numpy(float), data.t_category.to_numpy(float))[0])
    descriptive = pd.DataFrame([{"family": "descriptive", "statistic": "rho(N, T)", "patients": n, "rho": n_t}])
    LOGGER.info("rho(N, T) = %.3f", n_t)
    return pd.concat([results, descriptive], ignore_index=True)


# --- Figures (display only; every number shown comes from the tables) ---

def figure(data: pd.DataFrame, rbp: str) -> None:
    apply_style()
    jitter = np.random.default_rng(0)
    shades = (BLUE_RAMP(0.3), BLUE_RAMP(0.85))  # ordinal two-level groups: light = lower category

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True, sharey=True)
    views = [("n_category", "Pathological N", "t_category", ("T1", "T2+")),
             ("t_category", "Pathological T", "n_category", ("N0", "N+"))]
    for ax, (axis_var, axis_label, split_var, split_labels) in zip(axes, views):
        levels = sorted(data[axis_var].unique())
        high = data[split_var] > data[split_var].min()
        for offset, (mask, label, shade) in zip((-0.18, 0.18), ((~high, split_labels[0], shades[0]),
                                                                (high, split_labels[1], shades[1]))):
            part = data.loc[mask]
            values = [part.panel_score[part[axis_var].eq(level)].to_numpy() for level in levels]
            positions = [level + offset for level in levels]
            kept = [(p, v) for p, v in zip(positions, values) if len(v)]
            ax.boxplot([v for _, v in kept], positions=[p for p, _ in kept], widths=0.3, showfliers=False,
                       medianprops={"color": INK_SECONDARY, "linewidth": 1.5}, boxprops={"color": NEUTRAL},
                       whiskerprops={"color": NEUTRAL}, capprops={"color": NEUTRAL})
            ax.scatter(part[axis_var] + offset + jitter.uniform(-0.08, 0.08, len(part)), part.panel_score, s=12,
                       color=shade, linewidths=0.4, edgecolors="white", label=label)
        counts = data[axis_var].value_counts()
        prefix = axis_label[-1]
        ax.set_xticks(levels, [f"{prefix}{int(level)}\n(n={counts[level]})" for level in levels])
        ax.set_xlabel(axis_label)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", fontsize=8, handletextpad=0.2)
    axes[0].set_ylabel("Panel score, log2(median TPM + 1)")
    fig.suptitle(f"{rbp}-positive tumors", fontsize=10, fontweight="bold")
    save_svg_jpg(fig, OUT_SUPPLEMENTARY, "SupplementaryFigure7")  # scores by N and T


def main() -> None:
    setup_logging()
    config = load_config()
    settings, rbp = config["stage"], config["followup"]["rbp"]
    if "--figures-only" in sys.argv[1:]:
        figure(pd.read_csv(OUT_TABLES / "tnm_patients.tsv", sep="\t"), rbp)
        return

    roster = pd.read_csv(OUT_TABLES / "patient_scores.tsv", sep="\t")
    roster = roster.loc[roster.rbp.eq(rbp)].reset_index(drop=True)
    score, nulls = scores_with_nulls(rbp, list(config["rbps"]).index(rbp), roster, settings)

    patients = roster.merge(tnm_table(), on="patient_id", how="left", validate="one_to_one")
    complete = patients[["n_category", "t_category"]].notna().all(axis=1).to_numpy()
    LOGGER.info("%s: %d of %d staged patients have both N and T", rbp, int(complete.sum()), len(patients))
    data = patients.loc[complete].reset_index(drop=True)
    score, nulls = score[complete], {control: matrix[:, complete] for control, matrix in nulls.items()}

    stage_rho = float(partial_spearman(score, data.stage.to_numpy(float), np.empty((len(data), 0)))[0])
    LOGGER.info("check: rho(score, stage) in these patients = %.4f", stage_rho)

    results = associations(data, score, nulls, settings)
    write_table(results, OUT_TABLES / "tnm_association.tsv")
    write_table(data[["sample_id", "patient_id", "stage", "n_category", "t_category", "panel_score"]],
                OUT_TABLES / "tnm_patients.tsv")
    figure(data, rbp)


if __name__ == "__main__":
    main()
