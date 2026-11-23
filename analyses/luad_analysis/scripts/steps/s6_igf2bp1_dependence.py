"""Step 6: does the IGF2BP1-panel stage gradient require IGF2BP1?

The fixed IGF2BP1 panel, its universe and s4's 9,999 matched random panels (both controls) are scored in every
primary tumor. Every statistic carries a competitive P against those panels.

Primary family (BH over 3):
1. rho(score, stage) in the IGF2BP1-negative tumors (coding TPM <= 1): new patients, same panel.
2. rho(score, log2(IGF2BP1 coding TPM + 1)) across all tumors. Asks whether the candidates co-vary with IGF2BP1
   more than comparable transcripts do; uses no clinical data.
3. Partial rho(score, stage | IGF2BP1 TPM) in the IGF2BP1-positive tumors (Freedman-Lane P).

Descriptive (no BH):
- the s4 statistic recomputed here (positives);
- the difference in rho, positive minus negative, with an independent-group bootstrap CI;
- rho(IGF2BP1 TPM, stage) in positives.

Reading, fixed before running:
- IGF2BP1 dependence predicts a competitive excess (panel rho beyond the random panels) in positives that is absent
  in negatives, and test 2 beating its null.
- A compositional or global explanation predicts a similar rho and a similar excess in negatives.
- "Absent" is judged from the CI and the competitive P, not from P > 0.05, because the negative group is ~5x larger.

Writes tables/dependence.tsv, tables/dependence_stage_counts.tsv, tables/dependence_scores.tsv and
figures/Supplementary/{svg,jpg}/SupplementaryFigure{5,6} (scores by group; null distributions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (CONTROLS, LOGGER, OUT_SUPPLEMENTARY, OUT_TABLES, STAGES, bh, bootstrap_partial, effect,  # noqa: E402
                    load_config, percentile_ci, row_spearman, scores_with_nulls, setup_logging, stage_table, write_table)
from plotting import INK_SECONDARY, NEUTRAL, apply_style, rbp_colors, save_svg_jpg  # noqa: E402

STREAM = 6  # seed stream [seed, 6, k]


def no_controls(n: int) -> np.ndarray:
    """An (n, 0) control matrix: the effect is then a plain Spearman rho."""
    return np.empty((n, 0))


def run_effects(tumors: pd.DataFrame, score: np.ndarray, nulls: dict[str, np.ndarray], settings: dict) -> pd.DataFrame:
    """Every primary and descriptive effect, one row each."""
    seed, n_boot, n_perm = settings["seed"], settings["bootstraps"], settings["permutations"]
    staged = tumors.stage.notna().to_numpy()
    positive = tumors.positive.to_numpy()
    level = np.log2(tumors.rbp_coding_tpm.to_numpy() + 1.0)
    stage = tumors.stage.to_numpy(float)

    def boot(mask: np.ndarray, k: int) -> np.ndarray:
        n = int(mask.sum())
        return np.random.default_rng([seed, STREAM, k]).integers(0, n, size=(n_boot, n))

    def run(name: str, family: str, group: str, mask: np.ndarray, interest: np.ndarray, controls: np.ndarray | None,
            k: int) -> tuple[dict, np.ndarray]:
        c = no_controls(int(mask.sum())) if controls is None else controls[mask][:, None]
        row, replicates = effect(score[mask], interest[mask], c, {key: m[:, mask] for key, m in nulls.items()},
                                 boot(mask, k), n_perm, np.random.default_rng([seed, STREAM, 100 + k]))
        LOGGER.info("%s (%s, n=%d): rho %.3f [%.3f, %.3f], P=%.3g, competitive P %.3g / %.3g", name, group, row["patients"],
                    row["rho"], row["rho_low"], row["rho_high"], row["p"], row["competitive_p_expression_length"],
                    row["competitive_p_expression_length_gc"])
        return {"family": family, "statistic": name, "group": group, **row}, replicates

    everyone = np.ones(len(tumors), dtype=bool)
    neg, neg_reps = run("stage", "primary", "IGF2BP1-negative", staged & ~positive, stage, None, 1)
    coupling, _ = run("IGF2BP1 TPM", "primary", "all tumors", everyone, level, None, 2)
    adjusted, _ = run("stage | IGF2BP1 TPM", "primary", "IGF2BP1-positive", staged & positive, stage, level, 3)
    pos, pos_reps = run("stage", "descriptive", "IGF2BP1-positive", staged & positive, stage, None, 4)
    rows = [neg, coupling, adjusted, pos]

    level_stage = staged & positive
    rho_level = float(row_spearman(level[level_stage], stage[level_stage])[0])
    low, high, used = percentile_ci(bootstrap_partial(level[level_stage], stage[level_stage],
                                                      no_controls(int(level_stage.sum())), boot(level_stage, 8)))
    rows.append({"family": "descriptive", "statistic": "rho(IGF2BP1 TPM, stage)", "group": "IGF2BP1-positive",
                 "patients": int(level_stage.sum()), "rho": rho_level, "rho_low": low, "rho_high": high,
                 "bootstraps_used": used})
    low, high, used = percentile_ci(pos_reps - neg_reps)
    rows.append({"family": "descriptive", "statistic": "stage rho, positive - negative", "group": "both",
                 "patients": pos["patients"] + neg["patients"], "rho": pos["rho"] - neg["rho"], "rho_low": low,
                 "rho_high": high, "bootstraps_used": used})
    LOGGER.info("rho(IGF2BP1 TPM, stage) in positives: %.3f; stage rho positive - negative: %.3f [%.3f, %.3f]",
                rho_level, pos["rho"] - neg["rho"], low, high)

    results = pd.DataFrame(rows)
    primary = results.family.eq("primary").to_numpy()
    results.insert(results.columns.get_loc("p") + 1, "q_bh", np.nan)
    results.loc[primary, "q_bh"] = bh(results.p[primary].to_numpy())
    for control in CONTROLS:
        results[f"excess_rho_{control}"] = results.rho - results[f"null_median_rho_{control}"]
    return results


# --- Figures (display only; every number shown comes from the tables) ---

def figures(results: pd.DataFrame, scores: pd.DataFrame, null_rho: pd.DataFrame, rbp: str, color: str) -> None:
    apply_style()
    jitter = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), constrained_layout=True)
    low, high = scores.panel_score.min(), scores.panel_score.max()
    groups = (("IGF2BP1-positive", scores.positive, color), ("IGF2BP1-negative", ~scores.positive, NEUTRAL))
    for ax, (group, mask, shade) in zip(axes[:2], groups):
        part = scores.loc[mask & scores.stage.notna()]
        values = [part.panel_score[part.stage.eq(s)].to_numpy() for s in STAGES.values()]
        ax.boxplot(values, positions=list(STAGES.values()), widths=0.5, showfliers=False,
                   medianprops={"color": INK_SECONDARY, "linewidth": 1.5}, boxprops={"color": NEUTRAL},
                   whiskerprops={"color": NEUTRAL}, capprops={"color": NEUTRAL})
        ax.scatter(part.stage + jitter.uniform(-0.15, 0.15, len(part)), part.panel_score, s=8, color=shade, alpha=0.7,
                   linewidths=0)
        ax.set_xticks(list(STAGES.values()), [f"{name}\n(n={int(part.stage.eq(v).sum())})" for name, v in STAGES.items()])
        row = results.loc[results.statistic.eq("stage") & results.group.eq(group)].iloc[0]
        ax.text(0.97, 0.97, f"ρ = {row.rho:.2f} [{row.rho_low:.2f}, {row.rho_high:.2f}]\n"
                f"comp. P = {row.competitive_p_expression_length:.2g} / {row.competitive_p_expression_length_gc:.2g} (+ GC)",
                transform=ax.transAxes, va="top", ha="right", fontsize=7.5, color=INK_SECONDARY)
        ax.set_title(group)
        ax.set_xlabel("AJCC pathological stage")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(low - 0.05 * (high - low), high + 0.3 * (high - low))  # shared scale, headroom for the annotation
    axes[0].set_ylabel("Panel score, log2(median TPM + 1)")
    fig.suptitle(f"{rbp} panel", fontsize=10, fontweight="bold")

    ax = axes[2]
    for group, mask, shade in groups[::-1]:
        part = scores.loc[mask]
        ax.scatter(np.log2(part.rbp_coding_tpm + 1), part.panel_score, s=8, color=shade, alpha=0.7, linewidths=0,
                   label=f"{group} (n={len(part)})")
    row = results.loc[results.statistic.eq("IGF2BP1 TPM") & results.group.eq("all tumors")].iloc[0]
    ax.text(0.97, 0.03, f"ρ = {row.rho:.2f}\ncomp. P = {row.competitive_p_expression_length:.2g} / "
            f"{row.competitive_p_expression_length_gc:.2g} (+ GC)", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=7.5, color=INK_SECONDARY)
    ax.set_xlabel(f"{rbp} coding TPM, log2(TPM + 1)")
    ax.set_title("All tumors")
    ax.legend(loc="upper left", fontsize=7, markerscale=1.5, handletextpad=0.1)
    save_svg_jpg(fig, OUT_SUPPLEMENTARY, "SupplementaryFigure5")  # scores by stage and IGF2BP1 level

    panels = [("stage", "IGF2BP1-positive"), ("stage", "IGF2BP1-negative"), ("IGF2BP1 TPM", "all tumors")]
    columns = {("stage", "IGF2BP1-positive"): "ρ with stage, IGF2BP1-positive",
               ("stage", "IGF2BP1-negative"): "ρ with stage, IGF2BP1-negative",
               ("IGF2BP1 TPM", "all tumors"): "ρ with IGF2BP1 TPM, all tumors"}
    rows = {"expression_length": "Random panels\n(expression × length)", "expression_length_gc": "Random panels\n(+ GC)"}
    fig, axes = plt.subplots(len(CONTROLS), len(panels), figsize=(9.6, 4.6), constrained_layout=True)
    for r, control in enumerate(CONTROLS):
        for c, (statistic, group) in enumerate(panels):
            ax = axes[r, c]
            row = results.loc[results.statistic.eq(statistic) & results.group.eq(group)].iloc[0]
            values = null_rho.loc[null_rho.statistic.eq(statistic) & null_rho.group.eq(group) & null_rho.control.eq(control),
                                  "rho"]
            ax.hist(values, bins=60, color=NEUTRAL, edgecolor="none")
            ax.axvline(row.rho, color=color, linewidth=2)
            if r == 0:
                ax.set_title(columns[(statistic, group)], fontsize=9)
            right = row.rho < values.median()
            ax.text(0.97 if right else 0.03, 0.97, f"comp. P = {getattr(row, f'competitive_p_{control}'):.2g}",
                    transform=ax.transAxes, va="top", ha="right" if right else "left", fontsize=8, color=INK_SECONDARY)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
            ax.set_xlabel("Spearman ρ (random panels)")
    for ax, control in zip(axes[:, 0], CONTROLS):
        ax.set_ylabel(rows[control])
    save_svg_jpg(fig, OUT_SUPPLEMENTARY, "SupplementaryFigure6")  # null distributions


def main() -> None:
    setup_logging()
    config = load_config()
    settings, rbp = config["stage"], config["followup"]["rbp"]
    color = rbp_colors(list(config["rbps"]))[rbp]
    if "--figures-only" in sys.argv[1:]:
        figures(pd.read_csv(OUT_TABLES / "dependence.tsv", sep="\t"), pd.read_csv(OUT_TABLES / "dependence_scores.tsv", sep="\t"),
                pd.read_parquet(OUT_TABLES / "dependence_null.parquet"), rbp, color)
        return

    cohorts = pd.read_csv(OUT_TABLES / "cohorts.tsv", sep="\t")
    tumors = cohorts.loc[cohorts.rbp.eq(rbp), ["sample_id", "patient_id", "rbp_coding_tpm", "positive"]]
    tumors = tumors.merge(stage_table(), on="patient_id", how="left", validate="one_to_one").reset_index(drop=True)
    LOGGER.info("%d tumors: %d %s-positive, %d negative; %d with known stage", len(tumors), int(tumors.positive.sum()),
                rbp, int((~tumors.positive).sum()), int(tumors.stage.notna().sum()))
    score, nulls = scores_with_nulls(rbp, list(config["rbps"]).index(rbp), tumors, settings)

    results = run_effects(tumors, score, nulls, settings)
    scores = tumors.assign(panel_score=score)

    # null correlations behind the competitive Ps that the figure shows
    staged, positive = tumors.stage.notna().to_numpy(), tumors.positive.to_numpy()
    level = np.log2(tumors.rbp_coding_tpm.to_numpy() + 1.0)
    shown = [("stage", "IGF2BP1-positive", staged & positive, tumors.stage.to_numpy(float)),
             ("stage", "IGF2BP1-negative", staged & ~positive, tumors.stage.to_numpy(float)),
             ("IGF2BP1 TPM", "all tumors", np.ones(len(tumors), dtype=bool), level)]
    null_rho = pd.concat([pd.DataFrame({"statistic": statistic, "group": group, "control": control,
                                        "panel": np.arange(len(matrix)), "rho": row_spearman(matrix[:, mask], x[mask])})
                          for statistic, group, mask, x in shown for control, matrix in nulls.items()], ignore_index=True)

    counts = (scores.assign(group=np.where(scores.positive, f"{rbp}-positive", f"{rbp}-negative"))
              .groupby(["group", scores.stage.fillna(0).astype(int)]).size().rename("tumors").reset_index()
              .replace({"stage": {0: "unknown", **{v: k for k, v in STAGES.items()}}}))
    write_table(results, OUT_TABLES / "dependence.tsv")
    write_table(counts, OUT_TABLES / "dependence_stage_counts.tsv")
    write_table(scores, OUT_TABLES / "dependence_scores.tsv")
    write_table(null_rho, OUT_TABLES / "dependence_null.parquet")
    figures(results, scores, null_rho, rbp, color)


if __name__ == "__main__":
    main()
