"""Step 8: does tumor purity explain the IGF2BP1 panel's gradient, and in particular its excess over random panels?

- Patients: IGF2BP1-positive tumors with known stage and a consensus purity estimate. CPE is from Aran et al. 2015,
  Supplementary Data 1, which combines ESTIMATE, ABSOLUTE, LUMP and IHC; it is mapped by 15-character sample barcode.
- Primary (1 test): partial rho(score, outcome | CPE), with Freedman-Lane P and a bootstrap CI. Its competitive P
  applies the same CPE adjustment to each of s4's 9,999 matched random panels (both controls). The outcome is
  config.followup.outcome, fixed after s5 (stage).

Descriptive:
- the unadjusted effect in the same patients;
- the other clinical variable (N) given CPE;
- rho(score, CPE), with competitive P;
- rho(CPE, outcome).

Reading, fixed before running: the question is whether the excess (competitive P) survives adjustment, not only the
raw effect. CPE is a scalar; it cannot resolve specific stromal, immune or malignant-cell states.

Writes tables/composition.tsv, tables/composition_patients.tsv and
figures/Supplementary/{svg,jpg}/SupplementaryFigure8 (forest).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (LOGGER, OUT_SUPPLEMENTARY, OUT_TABLES, bootstrap_partial, effect, load_config,  # noqa: E402
                    percentile_ci, row_spearman, scores_with_nulls, setup_logging, source, tnm_table, write_table)
from plotting import forest, rbp_colors, save_svg_jpg  # noqa: E402

STREAM = 8  # seed stream [seed, 8, k]
OUTCOME_COLUMNS = {"stage": "stage", "N": "n_category"}
CONTROL_LABELS = {"expression_length": "expression × length", "expression_length_gc": "+ GC"}
PURITY_COLUMNS = ["Sample ID", "Cancer type", "ESTIMATE", "ABSOLUTE", "LUMP", "IHC", "CPE"]


def purity_table() -> pd.DataFrame:
    """CPE per LUAD sample (15-character barcode) from the pinned Aran et al. workbook."""
    with warnings.catch_warnings():  # openpyxl warns about an unused workbook extension; the data are unaffected
        warnings.filterwarnings("ignore", message="Unknown extension is not supported")
        sheet = pd.read_excel(source("purity"), sheet_name="Supp Data 1", header=3, engine="openpyxl")
    if list(sheet.columns[:len(PURITY_COLUMNS)]) != PURITY_COLUMNS:
        raise ValueError(f"unexpected purity workbook columns: {list(sheet.columns)}")
    luad = sheet.loc[sheet["Cancer type"].eq("LUAD")]
    return pd.DataFrame({"sample_id": luad["Sample ID"].astype(str).str.slice(0, 15),
                         "purity_source_id": luad["Sample ID"].astype(str),
                         "cpe": pd.to_numeric(luad.CPE, errors="coerce")})


def join_purity(patients: pd.DataFrame) -> pd.DataFrame:
    """Attach CPE; a sample with more than one source row (several vials) is an error, a missing one is reported."""
    purity = purity_table()
    counts = purity.sample_id.value_counts()
    ambiguous = sorted(set(counts[counts > 1].index) & set(patients.sample_id))
    if ambiguous:
        raise ValueError(f"purity source has several rows for analysed samples: {ambiguous}")
    purity = purity.loc[purity.sample_id.isin(patients.sample_id)]  # duplicates elsewhere in the source are irrelevant
    joined = patients.merge(purity, on="sample_id", how="left", validate="one_to_one")
    LOGGER.info("CPE: %d of %d patients mapped (%d without a source row, %d with a missing CPE)",
                int(joined.cpe.notna().sum()), len(joined), int(joined.purity_source_id.isna().sum()),
                int((joined.purity_source_id.notna() & joined.cpe.isna()).sum()))
    return joined


# --- Figure (display only; every number shown comes from the table) ---

def figure(results: pd.DataFrame, rbp: str, outcome: str, color: str) -> None:
    """Supplementary Figure 8. rho(CPE, outcome) is left out: it has no random-panel null, so it stays table-only."""
    shown = results.loc[results.statistic.ne(f"rho(CPE, {outcome})")].reset_index(drop=True)
    save_svg_jpg(forest(shown, list(shown.statistic), CONTROL_LABELS, color, f"{rbp} panel",
                        "Spearman ρ with the panel score"), OUT_SUPPLEMENTARY, "SupplementaryFigure8")


def main() -> None:
    setup_logging()
    config = load_config()
    settings, followup = config["stage"], config["followup"]
    rbp, outcome = followup["rbp"], followup["outcome"]
    if outcome not in OUTCOME_COLUMNS:
        raise ValueError(f"config.followup.outcome must be one of {sorted(OUTCOME_COLUMNS)} (set after s5), got {outcome!r}")
    color = rbp_colors(list(config["rbps"]))[rbp]
    if "--figures-only" in sys.argv[1:]:
        figure(pd.read_csv(OUT_TABLES / "composition.tsv", sep="\t"), rbp, outcome, color)
        return

    roster = pd.read_csv(OUT_TABLES / "patient_scores.tsv", sep="\t")
    roster = roster.loc[roster.rbp.eq(rbp)].reset_index(drop=True)
    score, nulls = scores_with_nulls(rbp, list(config["rbps"]).index(rbp), roster, settings)
    patients = join_purity(roster.merge(tnm_table(), on="patient_id", how="left", validate="one_to_one"))
    keep = patients.cpe.notna().to_numpy()
    data = patients.loc[keep].reset_index(drop=True)
    score, nulls = score[keep], {control: matrix[:, keep] for control, matrix in nulls.items()}

    # --- Effects: the primary adjusted effect, then descriptive context ---
    seed, n = settings["seed"], len(data)
    indices = np.random.default_rng([seed, STREAM, 1]).integers(0, n, size=(settings["bootstraps"], n))
    cpe, other = data.cpe.to_numpy(float)[:, None], next(name for name in OUTCOME_COLUMNS if name != outcome)
    none = np.empty((n, 0))
    specs = [  # (statistic, family, interest, controls)
        (f"{outcome} | CPE", "primary", data[OUTCOME_COLUMNS[outcome]].to_numpy(float), cpe),
        (f"{outcome} (unadjusted)", "descriptive", data[OUTCOME_COLUMNS[outcome]].to_numpy(float), none),
        (f"{other} | CPE", "descriptive", data[OUTCOME_COLUMNS[other]].to_numpy(float), cpe),
        ("CPE", "descriptive", cpe[:, 0], none),
    ]
    rows = []
    for k, (statistic, family, interest, controls) in enumerate(specs):
        row, _ = effect(score, interest, controls, nulls, indices, settings["permutations"],
                        np.random.default_rng([seed, STREAM, 10 + k]))
        rows.append({"family": family, "statistic": statistic, **row})
        LOGGER.info("%s: rho %.3f [%.3f, %.3f], P=%.3g, competitive P %.3g / %.3g", statistic, row["rho"], row["rho_low"],
                    row["rho_high"], row["p"], row["competitive_p_expression_length"],
                    row["competitive_p_expression_length_gc"])
    y = data[OUTCOME_COLUMNS[outcome]].to_numpy(float)
    low, high, used = percentile_ci(bootstrap_partial(cpe[:, 0], y, none, indices))
    rows.append({"family": "descriptive", "statistic": f"rho(CPE, {outcome})", "patients": n,
                 "rho": float(row_spearman(cpe[:, 0], y)[0]), "rho_low": low, "rho_high": high, "bootstraps_used": used})
    results = pd.DataFrame(rows)
    results.insert(results.columns.get_loc("p") + 1, "q_bh", np.where(results.family.eq("primary"), results.p, np.nan))
    LOGGER.info("rho(CPE, %s) = %.3f [%.3f, %.3f]", outcome, rows[-1]["rho"], low, high)

    write_table(results, OUT_TABLES / "composition.tsv")
    write_table(data[["sample_id", "patient_id", "stage", "n_category", "t_category", "purity_source_id", "cpe",
                      "panel_score"]], OUT_TABLES / "composition_patients.tsv")
    figure(results, rbp, outcome, color)


if __name__ == "__main__":
    main()
