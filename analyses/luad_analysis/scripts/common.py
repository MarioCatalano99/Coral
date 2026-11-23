"""Shared paths, configuration, IO and statistics for the LUAD stage analysis.

Data flow: scripts/inputs/ -> s1_cohorts -> s2_panels -> s3_overlap -> s4_stage -> follow-ups s5, s6, s8 -> figure_main.
Inputs are read from data/luad/ and hash-checked against config.json; outputs go to the analysis directory's tables/
and figures/.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]  # analyses/luad_analysis/; this file sits in its scripts/
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))
from coral.prediction_io import checkpoint_hash, read_predictions, sha256_file  # noqa: E402,F401

DATA = REPO / "data" / "luad"
SOURCE_DIR = DATA / "luad_sources"  # downloaded third-party files (scripts/inputs/download_*.py)
CONFIG_PATH = ROOT / "config.json"
OUT_TABLES, OUT_FIGURES = ROOT / "tables", ROOT / "figures"
OUT_SUPPLEMENTARY = OUT_FIGURES / "Supplementary"  # SupplementaryFigure<n>, numbered as in the paper
LOG_OFFSET = 0.001  # Xena Toil stores log2(TPM + 0.001)

LOGGER = logging.getLogger("luad_stage")


# --- Configuration, logging, IO ---

def load_config() -> dict[str, Any]:
    """Return the analysis configuration."""
    return json.loads(CONFIG_PATH.read_text())


def setup_logging() -> None:
    """Configure one log format for every step."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("matplotlib", "fontTools", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def source(key: str) -> Path:
    """Return a hash-checked input path from data/luad/."""
    entry = load_config()["sources"][key]
    path = DATA / entry["file"]
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise ValueError(f"{path} has sha256 {actual}, config.json pins {entry['sha256']}")
    return path


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a TSV or Parquet table, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False, compression="zstd")
    else:
        frame.to_csv(path, sep="\t", index=False)
    LOGGER.info("wrote %s (%d rows)", path.relative_to(REPO) if path.is_relative_to(REPO) else path, len(frame))


def ceil_fraction(fraction: float, total: int) -> int:
    """Return ceil(fraction * total) without floating-point overshoot (0.01 * 33300 -> 333)."""
    return math.ceil(round(fraction * total, 9))


def tumor_samples() -> pd.DataFrame:
    """Return primary tumors (sample_id, patient_id); the cohort has one tumor per patient."""
    samples = pd.read_csv(source("samples"), sep="\t")
    tumors = samples.loc[samples.sample_type.eq("primary_tumor"), ["sample_id", "patient_id"]]
    if tumors.patient_id.duplicated().any():
        raise ValueError("more than one primary tumor per patient")
    return tumors.reset_index(drop=True)


def load_log_expression(sample_ids: list[str], transcript_ids: list[str] | None = None) -> pd.DataFrame:
    """Return stored log2(TPM + 0.001) values, transcripts x samples, in the requested order."""
    frame = pd.read_parquet(source("expression"), columns=["transcript_id", *sample_ids]).set_index("transcript_id")
    if transcript_ids is not None:
        missing = pd.Index(transcript_ids).difference(frame.index)
        if len(missing):
            raise ValueError(f"{len(missing)} transcripts missing from the expression matrix")
        frame = frame.loc[transcript_ids]
    return frame


def to_tpm(log_values: np.ndarray) -> np.ndarray:
    """Convert stored log2(TPM + 0.001) to TPM, clipping the offset's rounding at zero."""
    return np.maximum(np.exp2(np.asarray(log_values, dtype=np.float64)) - LOG_OFFSET, 0.0)


# --- Statistics ---

def bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values (step-up, monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bins 0..n_bins-1; ties are split by input order, so callers fix the order."""
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return ranks * n_bins // len(values)


def matched_panels(strata: np.ndarray, panel: np.ndarray, n_panels: int, rng: np.random.Generator) -> np.ndarray:
    """Draw random panels (n_panels x K universe indices) with the panel's exact per-stratum counts.

    Sampling is without replacement within each stratum; candidates themselves may be drawn.
    """
    labels, counts = np.unique(strata[panel], return_counts=True)
    blocks = []
    for label, count in zip(labels, counts):
        members = np.flatnonzero(strata == label)
        keys = rng.random((n_panels, len(members)))
        chosen = np.argpartition(keys, count - 1, axis=1)[:, :count] if count < len(members) else np.tile(np.arange(len(members)), (n_panels, 1))
        blocks.append(members[chosen])
    return np.concatenate(blocks, axis=1)


def panel_scores(tpm: np.ndarray, panels: np.ndarray, batch: int = 256) -> np.ndarray:
    """log2(median panel TPM + 1) per patient; tpm is universe x patients, panels is n x K indices."""
    panels = np.atleast_2d(panels)
    out = np.empty((len(panels), tpm.shape[1]))
    for start in range(0, len(panels), batch):
        chunk = panels[start:start + batch]
        out[start:start + len(chunk)] = np.log2(np.median(tpm[chunk], axis=1) + 1.0)
    return out


def _centred_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, axis=-1)
    return ranks - ranks.mean(axis=-1, keepdims=True)


def row_spearman(scores: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Spearman rho of each row of scores with one outcome vector (average ranks for ties)."""
    x, y = _centred_ranks(np.atleast_2d(scores)), _centred_ranks(outcome)
    return (x @ y) / np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum())


def permutation_p(score: np.ndarray, outcome: np.ndarray, n: int, rng: np.random.Generator, chunk: int = 10_000) -> float:
    """Two-sided permutation P of Spearman rho, (1 + #|rho_perm| >= |rho|) / (1 + n)."""
    x, y = _centred_ranks(score), _centred_ranks(outcome)
    norm = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    observed = abs(x @ y) / norm
    hits, done = 0, 0
    while done < n:
        size = min(chunk, n - done)
        shuffled = rng.permuted(np.tile(y, (size, 1)), axis=1)
        hits += int(np.sum(np.abs(shuffled @ x) / norm >= observed - 1e-12))
        done += size
    return (1 + hits) / (1 + n)


def bootstrap_rho(score: np.ndarray, outcome: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float, int]:
    """Percentile 95% CI of Spearman rho over patient resamples; also returns the number of usable resamples."""
    idx = rng.integers(0, len(score), size=(n, len(score)))
    x, y = _centred_ranks(score[idx]), _centred_ranks(outcome[idx])
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (x * y).sum(axis=1) / np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum(axis=1))
    usable = np.isfinite(rho)
    if not usable.all():
        LOGGER.info("bootstrap: %d resamples without outcome variation were skipped", int((~usable).sum()))
    low, high = np.percentile(rho[usable], [2.5, 97.5])
    return float(low), float(high), int(usable.sum())


def competitive_p(statistic: float, null: np.ndarray) -> float:
    """Two-sided empirical P with the plus-one correction: (1 + #|null| >= |statistic|) / (1 + n)."""
    return float((1 + np.sum(np.abs(null) >= abs(statistic) - 1e-12)) / (1 + len(null)))


def percentile_ci(replicates: np.ndarray) -> tuple[float, float, int]:
    """Percentile 95% CI over bootstrap replicates, skipping undefined ones; also returns how many were usable."""
    replicates = np.asarray(replicates, dtype=float)
    usable = np.isfinite(replicates)
    if not usable.all():
        LOGGER.info("bootstrap: %d resamples with an undefined statistic were skipped", int((~usable).sum()))
    low, high = np.percentile(replicates[usable], [2.5, 97.5])
    return float(low), float(high), int(usable.sum())


# --- Partial Spearman from rank residuals, with Freedman-Lane permutation ---
#
# Every variable is replaced by its average ranks; score and interest ranks are residualized on an intercept plus
# the control ranks, and the residuals are correlated. With no controls this is exactly Spearman's rho, so one
# primitive serves univariate and adjusted effects.

def _control_basis(controls: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the span of [1, ranked controls]; rank-deficient columns are dropped."""
    controls = np.asarray(controls, dtype=float).reshape(len(controls), -1)
    design = np.column_stack([np.ones(len(controls)), rankdata(controls, axis=0) if controls.shape[1] else controls])
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    return u[:, singular > singular[0] * 1e-10]


def _residual(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the projection of each row (patients on the last axis) onto the control basis."""
    return values - (values @ basis) @ basis.T


def partial_spearman(scores: np.ndarray, interest: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Partial Spearman of each row of scores with interest, given controls (patients x c, c may be 0)."""
    basis = _control_basis(controls)
    x = _residual(rankdata(interest), basis)
    y = _residual(rankdata(np.atleast_2d(scores), axis=1), basis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (y @ x) / np.sqrt((y ** 2).sum(axis=1) * (x @ x))


def freedman_lane_p(score: np.ndarray, interest: np.ndarray, controls: np.ndarray, n: int, rng: np.random.Generator,
                    chunk: int = 10_000) -> float:
    """Two-sided Freedman-Lane P of a partial Spearman: (1 + #|rho_perm| >= |rho|) / (1 + n).

    The residuals of the reduced model rank(score) ~ controls are permuted and added back to its fitted values,
    which keeps the score-control relationship intact under the null. With no controls it is a plain permutation test.
    """
    basis = _control_basis(controls)
    x = _residual(rankdata(interest), basis)
    ranks = rankdata(score)
    residual = _residual(ranks, basis)
    fitted = ranks - residual
    observed = abs(residual @ x) / np.sqrt((residual @ residual) * (x @ x))
    hits, done = 0, 0
    while done < n:
        size = min(chunk, n - done)
        pseudo = _residual(fitted + rng.permuted(np.tile(residual, (size, 1)), axis=1), basis)
        hits += int(np.sum(np.abs(pseudo @ x) / np.sqrt((pseudo ** 2).sum(axis=1) * (x @ x)) >= observed - 1e-12))
        done += size
    return (1 + hits) / (1 + n)


def bootstrap_partial(score: np.ndarray, interest: np.ndarray, controls: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Partial Spearman in each patient resample (one row of indices each); NaN where it is undefined."""
    controls = np.asarray(controls, dtype=float).reshape(len(score), -1)
    return np.array([partial_spearman(score[i], interest[i], controls[i])[0] for i in indices])


def effect(score: np.ndarray, interest: np.ndarray, controls: np.ndarray, nulls: dict[str, np.ndarray],
           indices: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[dict[str, float], np.ndarray]:
    """One effect of the panel score: partial Spearman with interest | controls, bootstrap CI, Freedman-Lane P and,
    for every null (control name -> random-panel scores, panels x patients), the competitive P and the null's
    median and 95% range. Also returns the bootstrap replicates, so callers can form differences on shared resamples.
    """
    rho = float(partial_spearman(score, interest, controls)[0])
    replicates = bootstrap_partial(score, interest, controls, indices)
    low, high, used = percentile_ci(replicates)
    row = {"patients": len(score), "rho": rho, "rho_low": low, "rho_high": high, "bootstraps_used": used,
           "p": freedman_lane_p(score, interest, controls, permutations, rng)}
    for control, matrix in nulls.items():
        rho_null = partial_spearman(matrix, interest, controls)
        row[f"competitive_p_{control}"] = competitive_p(rho, rho_null)
        row[f"null_median_rho_{control}"] = float(np.median(rho_null))
        row[f"null_low_{control}"], row[f"null_high_{control}"] = (float(v) for v in np.percentile(rho_null, [2.5, 97.5]))
    return row, replicates


# --- Clinical variables (GDC pathological AJCC staging) ---

STAGES = {"I": 1, "II": 2, "III": 3, "IV": 4}


def stage_table() -> pd.DataFrame:
    """AJCC pathological stage I-IV per patient (substages collapsed); unknown stage is dropped."""
    gdc = pd.read_csv(source("clinical_gdc"), sep="\t", low_memory=False)
    gdc = gdc.loc[gdc.sample_type.eq("primary_tumor"), ["patient_id", "diagnosis_ajcc_pathologic_stage"]]
    roman = gdc.diagnosis_ajcc_pathologic_stage.str.extract(r"^Stage (IV|III|II|I)[ABC]?$", expand=False)
    unparsed = gdc.diagnosis_ajcc_pathologic_stage.notna() & roman.isna()
    if unparsed.any():
        raise ValueError(f"unrecognized stage values: {sorted(gdc.diagnosis_ajcc_pathologic_stage[unparsed].unique())}")
    return gdc.assign(stage=roman.map(STAGES)).dropna(subset=["stage"])[["patient_id", "stage"]]


def _tnm_level(values: pd.Series, pattern: str, unknown: str) -> pd.Series:
    """Integer level of a TNM category; the X category becomes missing and anything unrecognized raises."""
    level = values.str.extract(pattern, expand=False)
    unparsed = values.notna() & values.ne(unknown) & level.isna()
    if unparsed.any():
        raise ValueError(f"unrecognized {values.name} values: {sorted(values[unparsed].unique())}")
    return pd.to_numeric(level)


def tnm_table() -> pd.DataFrame:
    """Pathological N (0-3) and T (1-4, substages collapsed) per patient; NX and TX are missing."""
    gdc = pd.read_csv(source("clinical_gdc"), sep="\t", low_memory=False)
    gdc = gdc.loc[gdc.sample_type.eq("primary_tumor")]
    return pd.DataFrame({
        "patient_id": gdc.patient_id,
        "n_category": _tnm_level(gdc.diagnosis_ajcc_pathologic_n, r"^N([0-3])$", "NX"),
        "t_category": _tnm_level(gdc.diagnosis_ajcc_pathologic_t, r"^T([1-4])[ab]?$", "TX"),
    }).reset_index(drop=True)


# --- Matched random panels: the one null shared by s4 and every follow-up step ---

CONTROLS = ("expression_length", "expression_length_gc")


def strata_for(universe: pd.DataFrame, settings: dict, with_gc: bool) -> np.ndarray:
    """Stratum label per universe transcript: expression decile x length quintile (x GC quintile)."""
    labels = (rank_bins(universe.mean_log2_tpm1.to_numpy(), settings["expression_bins"]) * settings["length_bins"]
              + rank_bins(universe.sequence_length_nt.to_numpy(), settings["length_bins"]))
    if with_gc:
        labels = labels * settings["gc_bins"] + rank_bins(universe.gc_fraction.to_numpy(), settings["gc_bins"])
    return labels


def null_panels(universe: pd.DataFrame, panel: np.ndarray, settings: dict, rbp_index: int, control: str) -> np.ndarray:
    """s4's matched random panels for one RBP and control, from s4's seed stream [seed, rbp_index, 3 | 4]."""
    return matched_panels(strata_for(universe, settings, control.endswith("gc")), panel, settings["random_panels"],
                          np.random.default_rng([settings["seed"], rbp_index, 3 + CONTROLS.index(control)]))


def load_panel(rbp: str) -> tuple[pd.DataFrame, np.ndarray]:
    """An RBP's universe in s4's row order (by transcript ID) and the row indices of its panel."""
    rankings = pd.read_parquet(OUT_TABLES / "rankings.parquet")
    universe = rankings.loc[rankings.rbp.eq(rbp)].sort_values("transcript_id").reset_index(drop=True)
    return universe, np.flatnonzero(universe.in_panel.to_numpy())


def scores_with_nulls(rbp: str, rbp_index: int, tumors: pd.DataFrame, settings: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Panel score and s4's matched-panel scores (control -> panels x tumors) in the given tumors.

    The tumors must include every patient s4 analysed. Those patients must reproduce s4's panel scores and null stage
    correlations exactly, which guarantees that every follow-up step tests against the identical null.
    """
    universe, panel = load_panel(rbp)
    tpm = to_tpm(load_log_expression(tumors.sample_id.tolist(), universe.transcript_id.tolist()).to_numpy())
    score = panel_scores(tpm, panel)[0]
    nulls = {control: panel_scores(tpm, null_panels(universe, panel, settings, rbp_index, control)) for control in CONTROLS}

    s4 = pd.read_csv(OUT_TABLES / "patient_scores.tsv", sep="\t")
    s4 = s4.loc[s4.rbp.eq(rbp)]
    rows = pd.Index(tumors.patient_id).get_indexer(s4.patient_id)
    if (rows < 0).any() or not np.allclose(score[rows], s4.panel_score.to_numpy(), rtol=0, atol=1e-12):
        raise ValueError(f"{rbp}: panel scores do not reproduce s4's patient_scores.tsv")
    saved = pd.read_parquet(OUT_TABLES / "null_distribution.parquet")
    for control, matrix in nulls.items():
        expected = saved.loc[saved.rbp.eq(rbp) & saved.control.eq(control)].sort_values("panel").spearman_rho.to_numpy()
        if not np.allclose(row_spearman(matrix[:, rows], s4.stage.to_numpy(float)), expected, rtol=0, atol=1e-12):
            raise ValueError(f"{rbp}/{control}: rebuilt null panels do not reproduce s4's null distribution")
        LOGGER.info("%s/%s: rebuilt null panels reproduce s4 (%d panels)", rbp, control, len(expected))
    return score, nulls

