"""Tests for LUAD expression-contextualized candidate networks."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "analyses" / "luad_analysis" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str) -> object:
    """Load one analysis script module without requiring package installation."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_module("_expression_context")
TABLES = CORE
RECURRENCE = CORE
RANKS = load_module("_rankings")
BUILD = load_module("build_expression_context")


class ExpressionContextCoreTests(unittest.TestCase):
    """Cover ranking, TPM conversion, availability gates, and test helpers."""

    def test_logged_to_tpm_and_inclusive_boundary(self) -> None:
        """Invert the Xena transform and retain an exact 1-TPM boundary."""
        logged = pd.Series(np.log2(np.array([0.001, 1.001, 2.501])))
        observed = CORE.logged_to_tpm(logged)
        np.testing.assert_allclose(observed, [0.0, 1.0, 2.5], atol=1e-12)
        self.assertTrue(bool(logged.iloc[1] >= np.log2(1.001)))

    def test_rbp_expression_includes_only_coding_isoforms(self) -> None:
        """Exclude noncoding RBP isoforms and never add logged values."""
        expression = pd.DataFrame(
            {"S": np.log2([2.001, 3.001])}, index=["eligible", "ineligible"]
        )
        transcripts = pd.DataFrame(
            {
                "transcript_id": ["eligible", "ineligible"],
                "gene_id": ["G", "G"],
                "gene_name": ["RBP", "RBP"],
                "primary_eligible": [True, False],
                "transcript_biotype": ["protein_coding", "retained_intron"],
            }
        )
        result = CORE.build_rbp_expression(
            expression,
            transcripts,
            {"P": {"gene_id": "G", "gene_symbol": "RBP", "transcript_count": 2}},
            1.0,
        )["P"]
        self.assertAlmostEqual(result.loc["S", "rbp_gene_tpm"], 2.0)
        self.assertTrue(result.loc["S", "rbp_available"])

    def test_ranking_propagates_duplicates_and_breaks_ties_by_id(self) -> None:
        """Keep transcript edges distinct when two transcripts share a sequence."""
        transcripts = pd.DataFrame(
            {
                "transcript_id": ["T2", "T1", "T3"],
                "RNA_id": ["R1", "R1", "R2"],
                "gene_id": ["G1", "G1", "G2"],
                "gene_name": ["A", "A", "B"],
                "transcript_biotype": ["x", "x", "x"],
                "sequence_length_nt": [10, 10, 20],
                "primary_eligible": [True, True, True],
            }
        )
        predictions = pd.DataFrame(
            {
                "RNA_id": ["R1", "R2"],
                "Protein_id": ["P", "P"],
                "logit_margin": [2.0, 1.0],
                "interaction_probability": [0.9, 0.8],
                "prediction": [1, 1],
            }
        )
        eligibility = transcripts[["transcript_id", "primary_eligible"]].assign(Protein_id="P")
        ranked = RANKS.rank_transcripts(transcripts, eligibility, predictions, top_fraction=0.5)
        self.assertEqual(ranked["transcript_id"].tolist(), ["T1", "T2", "T3"])
        top = ranked.loc[ranked.top_candidate]
        self.assertEqual(top["transcript_id"].tolist(), ["T1", "T2"])
        self.assertEqual(int(ranked.tie_inclusive_candidate.sum()), 2)
        self.assertEqual(ranked.score_percentile.iloc[0], ranked.score_percentile.iloc[1])

    def test_active_state_requires_both_expression_gates(self) -> None:
        """Implement target AND RBP availability as an explicit truth table."""
        samples = pd.DataFrame(
            {
                "sample_id": ["S00", "S01", "S10", "S11"],
                "sample_type": ["primary_tumor"] * 4,
                "patient_id": ["P0", "P1", "P2", "P3"],
                "pair_status": ["unpaired"] * 4,
                "expression_column_index": range(4),
            }
        )
        top = pd.DataFrame({"Protein_id": ["P"], "transcript_id": ["T"]})
        expression = pd.DataFrame(
            [np.log2([0.001, 1.001, 0.001, 1.001])],
            index=["T"],
            columns=samples["sample_id"],
        )
        rbp = {
            "P": pd.DataFrame(
                {
                    "rbp_gene_tpm": [0.0, 0.0, 1.0, 1.0],
                    "rbp_available": [False, False, True, True],
                },
                index=samples["sample_id"],
            )
        }
        states, summary = CORE.build_sample_edge_states(top, expression, samples, rbp, 1.0)
        self.assertEqual(states["active"].tolist(), [False, False, False, True])
        self.assertEqual(summary["target_available_edge_count"].tolist(), [0, 1, 0, 1])
        self.assertEqual(summary["active_edge_count"].tolist(), [0, 0, 0, 1])

    def test_statistical_helpers_are_explicit_and_deterministic(self) -> None:
        """Handle all-zero Wilcoxon inputs and exact McNemar discordances."""
        self.assertEqual(
            CORE.wilcoxon_test(
                np.zeros(5),
                zero_method="pratt",
                alternative="two-sided",
                resamples=999,
                seed=7,
            ),
            (0.0, 1.0, "all_zero_differences"),
        )
        kwargs = dict(
            zero_method="pratt", alternative="two-sided", resamples=999, seed=7
        )
        first = CORE.wilcoxon_test(np.array([0, 1, 2, -1, 3]), **kwargs)
        second = CORE.wilcoxon_test(np.array([0, 1, 2, -1, 3]), **kwargs)
        self.assertEqual(first, second)
        result = CORE.exact_mcnemar([0, 1, 0, 1], [1, 0, 0, 1])
        self.assertEqual(
            [result[name] for name in (
                "matched_both_inactive",
                "matched_tumor_only",
                "matched_normal_only",
                "matched_both_active",
            )],
            [1, 1, 1, 1],
        )
        self.assertEqual(result["pvalue"], 1.0)

    def test_conditional_prevalence_is_undefined_without_rbp_samples(self) -> None:
        """Do not turn an empty conditional denominator into a biological zero."""
        observed = CORE.conditional_prevalence([0, 3], [0, 5])
        self.assertTrue(np.isnan(observed[0]))
        self.assertAlmostEqual(observed[1], 0.6)

    def test_sample_axis_and_pairing_validation(self) -> None:
        """Bind expression columns to the declared stable sample order."""
        samples = pd.DataFrame(
            {
                "sample_id": ["T", "N"],
                "sample_type": ["primary_tumor", "solid_tissue_normal"],
                "patient_id": ["P", "P"],
                "pair_status": ["matched", "matched"],
                "expression_column_index": [0, 1],
            }
        )
        config = {
            "expected_samples": {"tumors": 1, "normals": 1, "matched_pairs": 1}
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "expression.parquet"
            pq.write_table(
                pa.table({"transcript_id": ["TX"], "T": [0.0], "N": [0.0]}), path
            )
            validated = BUILD.validate_samples(path, samples, config)
            self.assertEqual(validated["sample_id"].tolist(), ["T", "N"])
            wrong = samples.copy()
            wrong["expression_column_index"] = [1, 0]
            with self.assertRaisesRegex(ValueError, "expression sample columns"):
                BUILD.validate_samples(path, wrong, config)


class ExpressionContextTableTests(unittest.TestCase):
    """Cover recurrence and literature-anchor gates at their reporting level."""

    def test_recurrent_gene_uses_any_supporting_isoform(self) -> None:
        """OR supporting isoforms within an RBP before intersecting RBPs."""
        proteins = pd.DataFrame(
            {"Protein_id": ["P1", "P2", "P3"], "gene_symbol": ["R1", "R2", "R3"]}
        )
        top = pd.DataFrame(
            {
                "Protein_id": ["P1", "P1", "P2", "P3"],
                "transcript_id": ["T1", "T2", "T3", "T4"],
                "gene_id": ["G"] * 4,
                "gene_name": ["TARGET"] * 4,
                "transcript_biotype": ["mRNA"] * 4,
                "primary_transcript_rank": [1, 2, 3, 4],
                "logit_margin": [4.0, 3.0, 2.0, 1.0],
            }
        )
        samples = pd.DataFrame(
            {
                "sample_id": ["N", "T"],
                "sample_type": ["solid_tissue_normal", "primary_tumor"],
            }
        )
        active = {("P1", "T1"): [False, False], ("P1", "T2"): [False, True]}
        active |= {("P2", "T3"): [False, True], ("P3", "T4"): [False, True]}
        rows = []
        for (protein_id, transcript_id), values in active.items():
            for sample_id, value in zip(samples["sample_id"], values, strict=True):
                rows.append(
                    {
                        "Protein_id": protein_id,
                        "transcript_id": transcript_id,
                        "sample_id": sample_id,
                        "active": value,
                    }
                )
        sample_context = pd.DataFrame([
            {"sample_id": sample_id, "Protein_id": protein_id, "rbp_available": sample_id == "T"}
            for sample_id in samples["sample_id"] for protein_id in proteins["Protein_id"]
        ])
        recurrent = RECURRENCE.build_recurrent_targets(
            top, pd.DataFrame(rows), proteins, samples, sample_context
        )
        p1 = recurrent.loc[recurrent["Protein_id"] == "P1"].iloc[0]
        self.assertEqual(p1["supporting_transcript_count"], 2)
        self.assertEqual(p1["tumor_gene_active_count"], 1)
        self.assertEqual(p1["all_linked_active_tumor_count"], 1)
        self.assertNotIn("network_selected", recurrent.columns)

    def test_non_top_anchor_is_jointly_available_but_never_active(self) -> None:
        """Keep expression context visible without bypassing the CORAL gate."""
        samples = pd.DataFrame(
            {
                "sample_id": ["T", "N"],
                "sample_type": ["primary_tumor", "solid_tissue_normal"],
            }
        )
        anchors = pd.DataFrame(
            {
                "rbp_gene": ["LIN28B", "IGF2BP1"],
                "target_gene_name": ["IGF2", "OTHER"],
                "Protein_id": ["P1", "P2"],
                "representative_transcript_id": ["T1", "T2"],
                "primary_transcript_rank": [1, 700],
                "top1pct_candidate": [True, False],
            }
        )
        expression = pd.DataFrame(
            np.log2([[2.001, 2.001], [2.001, 2.001]]),
            index=["T1", "T2"],
            columns=["T", "N"],
        )
        rbp = {
            protein_id: pd.DataFrame(
                {"rbp_gene_tpm": [2.0, 2.0], "rbp_available": [True, True]},
                index=["T", "N"],
            )
            for protein_id in ("P1", "P2")
        }
        output = TABLES.build_anchor_context(anchors, expression, rbp, samples, 1.0)
        other = output.loc[output["target_gene_name"] == "OTHER"].iloc[0]
        self.assertEqual(other["tumor_joint_available_count"], 1)
        self.assertEqual(other["tumor_active_count"], 0)

    def test_paired_outputs_keep_direction_pseudocount_and_bh_families(self) -> None:
        """Use tumor-minus-normal changes and adjust each three-RBP family separately."""
        proteins = pd.DataFrame(
            {"Protein_id": ["P1", "P2", "P3"], "gene_symbol": ["R1", "R2", "R3"]}
        )
        samples = pd.DataFrame(
            {
                "sample_id": ["T1", "N1", "T2", "N2"],
                "sample_type": ["primary_tumor", "solid_tissue_normal"] * 2,
                "patient_id": ["A", "A", "B", "B"],
                "pair_status": ["matched"] * 4,
            }
        )
        rows = []
        for protein_id in proteins["Protein_id"]:
            for sample_id, tpm, count in (
                ("T1", 2.0, 1), ("N1", 1.0, 0), ("T2", 4.0, 3), ("N2", 2.0, 1)
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "Protein_id": protein_id,
                        "rbp_gene_tpm": tpm,
                        "rbp_available": tpm >= 1,
                        "active_edge_count": count,
                        "candidate_edge_count": 640,
                    }
                )
        config = {
            "expression_pseudocount": 0.001,
            "active_count_pseudocount": 1.0,
            "expected_top_per_rbp": 640,
            "bh_alpha": 0.05,
            "wilcoxon": {
                "difference_round_decimals": 12,
                "zero_method": "pratt",
                "alternative": "two-sided",
                "permutation_resamples": 999,
                "seed": 7,
            },
        }
        changes, tests = TABLES.build_paired_outputs(
            pd.DataFrame(rows), proteins, samples, config
        )
        first = changes.loc[
            (changes["Protein_id"] == "P1") & (changes["patient_id"] == "A")
        ].iloc[0]
        self.assertEqual(first["active_edge_count_difference"], 1)
        self.assertEqual(first["active_edge_count_log2_fold_change"], 1.0)
        self.assertEqual(tests.groupby("bh_family").size().to_dict(), {
            "active_edge_count": 3,
            "rbp_expression": 3,
        })


if __name__ == "__main__":
    unittest.main()
