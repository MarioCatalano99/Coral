"""Synthetic-data tests for the statistics behind the stage analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import (_tnm_level, bh, bootstrap_partial, bootstrap_rho, ceil_fraction, competitive_p,  # noqa: E402
                    freedman_lane_p, matched_panels, null_panels, panel_scores, partial_spearman,
                    percentile_ci, permutation_p, rank_bins, row_spearman)


class MultiplicityTests(unittest.TestCase):
    def test_bh_matches_hand_calculation(self) -> None:
        # sorted 0.01, 0.03, 0.04 -> scaled 0.03, 0.045, 0.04 -> monotone 0.03, 0.04, 0.04
        np.testing.assert_allclose(bh(np.array([0.01, 0.04, 0.03])), [0.03, 0.04, 0.04])

    def test_competitive_p_counts_ties_and_adds_one(self) -> None:
        self.assertEqual(competitive_p(0.2, np.array([-0.2, 0.1, 0.3, 0.0])), 3 / 5)

    def test_ceil_fraction_does_not_overshoot(self) -> None:
        self.assertEqual(ceil_fraction(0.01, 33300), 333)
        self.assertEqual(ceil_fraction(0.5, 85), 43)


class PanelTests(unittest.TestCase):
    def test_rank_bins_are_equal_count(self) -> None:
        np.testing.assert_array_equal(np.bincount(rank_bins(np.random.default_rng(1).normal(size=1000), 10)), np.full(10, 100))

    def test_matched_panels_reproduce_stratum_counts_without_repeats(self) -> None:
        rng = np.random.default_rng(2)
        strata = rng.integers(0, 12, size=600)
        panel = rng.choice(600, size=40, replace=False)
        draws = matched_panels(strata, panel, 300, np.random.default_rng(3))
        target = np.bincount(strata[panel], minlength=12)
        for row in draws:
            self.assertEqual(len(np.unique(row)), 40)
            np.testing.assert_array_equal(np.bincount(strata[row], minlength=12), target)

    def test_panel_score_is_log2_median_plus_one(self) -> None:
        tpm = np.array([[0.0, 3.0], [1.0, 7.0], [5.0, 15.0]])
        np.testing.assert_allclose(panel_scores(tpm, np.array([0, 1, 2]))[0], [1.0, 3.0])


class SpearmanTests(unittest.TestCase):
    def test_row_spearman_matches_scipy_with_tied_stages(self) -> None:
        rng = np.random.default_rng(21)
        stage = rng.integers(1, 5, size=80).astype(float)
        scores = rng.normal(size=(5, 80))
        np.testing.assert_allclose(row_spearman(scores, stage), [spearmanr(row, stage).statistic for row in scores])

    def test_permutation_p_separates_trend_from_noise(self) -> None:
        rng = np.random.default_rng(22)
        stage = rng.integers(1, 5, size=100).astype(float)
        self.assertLess(permutation_p(stage + rng.normal(0, 1, 100), stage, 9_999, np.random.default_rng(1)), 0.001)
        self.assertGreater(permutation_p(rng.normal(size=100), stage, 9_999, np.random.default_rng(2)), 0.05)

    def test_bootstrap_interval_brackets_the_estimate(self) -> None:
        rng = np.random.default_rng(23)
        stage = rng.integers(1, 5, size=120).astype(float)
        score = -stage + rng.normal(0, 1.5, 120)
        rho = row_spearman(score, stage)[0]
        low, high, used = bootstrap_rho(score, stage, 2000, np.random.default_rng(3))
        self.assertTrue(low < rho < high < 0)
        self.assertEqual(used, 2000)


class PlantedStageEffectTest(unittest.TestCase):
    def test_planted_panel_beats_matched_random_panels(self) -> None:
        rng = np.random.default_rng(11)
        n_transcripts, n_patients = 400, 120
        stage = rng.integers(1, 5, size=n_patients).astype(float)
        base = rng.normal(2.0, 1.0, size=(n_transcripts, 1))
        log_tpm = base + rng.normal(0, 0.4, size=(n_transcripts, n_patients))
        panel = np.arange(20)
        log_tpm[panel] -= 0.3 * stage
        tpm = np.exp2(log_tpm)
        strata = rank_bins(base[:, 0], 4) * 2 + rank_bins(rng.normal(size=n_transcripts), 2)

        rho = row_spearman(panel_scores(tpm, panel)[0], stage)[0]
        null = row_spearman(panel_scores(tpm, matched_panels(strata, panel, 999, np.random.default_rng(12))), stage)
        self.assertLess(rho, -0.3)
        self.assertLess(competitive_p(rho, null), 0.05)


class PartialSpearmanTests(unittest.TestCase):
    def test_matches_an_independent_rank_residual_computation(self) -> None:
        rng = np.random.default_rng(31)
        control = rng.integers(1, 5, size=90).astype(float)
        interest = control + rng.integers(0, 3, size=90)
        score = rng.normal(size=90) - 0.5 * interest
        design = np.column_stack([np.ones(90), rankdata(control)])

        def residual(v: np.ndarray) -> np.ndarray:
            return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

        x, y = residual(rankdata(interest)), residual(rankdata(score))
        expected = (x @ y) / np.sqrt((x @ x) * (y @ y))
        self.assertAlmostEqual(partial_spearman(score, interest, control[:, None])[0], expected, places=12)

    def test_without_controls_it_is_spearman(self) -> None:
        rng = np.random.default_rng(32)
        stage = rng.integers(1, 5, size=70).astype(float)
        scores = rng.normal(size=(4, 70))
        np.testing.assert_allclose(partial_spearman(scores, stage, np.empty((70, 0))), row_spearman(scores, stage), atol=1e-12)

    def test_an_independent_control_leaves_the_effect_unchanged(self) -> None:
        rng = np.random.default_rng(33)
        interest = rng.normal(size=4000)
        score = interest + rng.normal(size=4000)
        unrelated = rng.normal(size=4000)
        self.assertAlmostEqual(partial_spearman(score, interest, unrelated[:, None])[0],
                               row_spearman(score, interest)[0], delta=0.01)

    def test_freedman_lane_rejects_a_planted_effect_but_not_confounding(self) -> None:
        rng = np.random.default_rng(34)
        control = rng.integers(1, 5, size=150).astype(float)
        interest = control + rng.normal(0, 0.7, size=150)
        confounded = control + rng.normal(0, 1, size=150)  # depends on the control only
        planted = confounded - interest
        self.assertGreater(row_spearman(confounded, interest)[0], 0.5)  # strong marginal effect, all via the control
        self.assertLess(permutation_p(confounded, interest, 2_000, np.random.default_rng(1)), 0.001)
        self.assertGreater(freedman_lane_p(confounded, interest, control[:, None], 2_000, np.random.default_rng(2)), 0.05)
        self.assertLess(freedman_lane_p(planted, interest, control[:, None], 2_000, np.random.default_rng(3)), 0.001)

    def test_bootstrap_partial_brackets_the_estimate(self) -> None:
        rng = np.random.default_rng(35)
        control = rng.normal(size=120)
        interest = rng.normal(size=120)
        score = -interest + control + rng.normal(size=120)
        rho = partial_spearman(score, interest, control[:, None])[0]
        indices = np.random.default_rng(4).integers(0, 120, size=(1000, 120))
        low, high, used = percentile_ci(bootstrap_partial(score, interest, control, indices))
        self.assertTrue(low < rho < high < 0)
        self.assertEqual(used, 1000)


class FollowUpHelperTests(unittest.TestCase):
    def test_null_panels_use_the_s4_seed_stream(self) -> None:
        rng = np.random.default_rng(36)
        universe = pd.DataFrame({"mean_log2_tpm1": rng.normal(size=500), "sequence_length_nt": rng.integers(200, 6000, 500),
                                 "gc_fraction": rng.uniform(0.3, 0.7, 500)})
        settings = {"expression_bins": 10, "length_bins": 5, "gc_bins": 5, "random_panels": 50, "seed": 7}
        panel = np.arange(0, 500, 25)
        strata = rank_bins(universe.mean_log2_tpm1.to_numpy(), 10) * 5 + rank_bins(universe.sequence_length_nt.to_numpy(), 5)
        expected = matched_panels(strata, panel, 50, np.random.default_rng([7, 1, 3]))
        np.testing.assert_array_equal(null_panels(universe, panel, settings, 1, "expression_length"), expected)

    def test_tnm_levels_collapse_substages_and_drop_x(self) -> None:
        values = pd.Series(["T1a", "T2b", "T4", "TX", None], name="t")
        np.testing.assert_array_equal(_tnm_level(values, r"^T([1-4])[ab]?$", "TX").to_numpy(), [1, 2, 4, np.nan, np.nan])
        with self.assertRaises(ValueError):
            _tnm_level(pd.Series(["T5"], name="t"), r"^T([1-4])[ab]?$", "TX")


if __name__ == "__main__":
    unittest.main()
