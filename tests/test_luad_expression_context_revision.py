"""Regression tests for RBP-conditioned denominators and paired heatmap gate states."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "analyses/luad_analysis/scripts"
sys.path.insert(0, str(SCRIPTS))

from _expression_context import build_rbp_subsets, build_recurrent_targets
from plot_expression_context import plot_paired_heatmaps, save_figure, validate_paired_values


class ConditionalRecurrenceTests(unittest.TestCase):
    """Check denominators with different RBP-positive subsets and supporting isoforms."""

    def test_own_subset_and_common_subset_are_distinct(self) -> None:
        proteins = pd.DataFrame({"Protein_id": ["P1", "P2", "P3"], "gene_symbol": ["R1", "R2", "R3"]})
        samples = pd.DataFrame({
            "sample_id": ["D", "N", "B", "A", "C"],
            "sample_type": ["primary_tumor", "solid_tissue_normal"] + ["primary_tumor"] * 3,
        })
        positives = {"P1": {"A", "B", "C"}, "P2": {"B", "C", "D"}, "P3": {"A", "C", "D"}}
        support = [
            ("G", "P1", "T1", {"A"}), ("G", "P1", "T2", {"B"}),
            ("G", "P2", "T3", {"B", "C", "D"}), ("G", "P3", "T4", {"C", "D"}),
            ("H", "P1", "X", {"B", "C"}), ("H", "P2", "Y", {"C", "D"}),
        ]
        top = pd.DataFrame([
            {"gene_id": gene, "gene_name": gene, "Protein_id": protein, "transcript_id": transcript,
             "primary_transcript_rank": rank, "transcript_biotype": "mRNA", "logit_margin": 10 - rank}
            for rank, (gene, protein, transcript, active) in enumerate(support, 1)
        ])
        states = pd.DataFrame([
            {"Protein_id": protein, "transcript_id": transcript, "sample_id": sample, "active": sample in active}
            for gene, protein, transcript, active in support for sample in samples.sample_id
        ])
        sample_context = pd.DataFrame([
            {"sample_id": sample.sample_id, "sample_type": sample.sample_type,
             "Protein_id": protein, "rbp_available": sample.sample_id in positive}
            for sample in samples.itertuples() for protein, positive in positives.items()
        ])
        result = build_recurrent_targets(top, states, proteins, samples, sample_context)
        g = result[(result.gene_id == "G") & (result.Protein_id == "P1")].iloc[0]
        self.assertEqual(g.supporting_transcript_count, 2)
        self.assertEqual(g.rbp_degree, 3)
        self.assertEqual(g.tumor_cohort_samples, 4)
        self.assertEqual(g.tumor_gene_active_count, 2)
        self.assertEqual(g.tumor_gene_active_fraction, 0.5)
        self.assertAlmostEqual(g.tumor_gene_active_given_rbp_available, 2 / 3)
        self.assertEqual(g.all_linked_rbp_available_tumor_count, 1)
        self.assertEqual(g.tumor_gene_active_given_all_rbps_available, 0)
        self.assertEqual(g.all_linked_active_tumor_given_all_rbps_available, 0)
        self.assertTrue(np.isnan(g.normal_gene_active_given_rbp_available))
        self.assertTrue(np.isnan(g.all_linked_active_normal_given_all_rbps_available))
        h = result[(result.gene_id == "H") & (result.Protein_id == "P1")].iloc[0]
        self.assertEqual(h.rbp_degree, 2)
        self.assertEqual(h.all_linked_rbp_available_tumor_count, 2)
        self.assertEqual(h.all_linked_active_tumor_given_all_rbps_available, 0.5)
        self.assertEqual(h.tumor_gene_active_given_all_rbps_available, 1)
        subsets = build_rbp_subsets(sample_context, proteins)
        self.assertEqual(len(subsets), 14)
        triple = subsets[(subsets.rbp_degree == 3) & (subsets.sample_type == "primary_tumor")].iloc[0]
        self.assertEqual(triple.rbp_available_count, 1)
        self.assertEqual(triple.rbp_available_sample_ids, "C")
        pair = subsets[(subsets.rbp_genes == "R1;R2") & (subsets.sample_type == "primary_tumor")].iloc[0]
        self.assertEqual(pair.rbp_available_sample_ids, "B;C")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_rbp_subsets(sample_context.iloc[1:], proteins)


class PairedHeatmapTests(unittest.TestCase):
    """Keep RBP-negative cells visually distinct from actual target-negative cells."""

    def setUp(self) -> None:
        self.config = json.loads((SCRIPTS.parent / "config/expression_context.json").read_text())
        self.config["expected_samples"]["matched_pairs"] = 2
        self.order = ["IGF2BP1", "IGF2BP3", "LIN28B"]
        rows = []
        for rbp in self.order:
            for patient, normal_tpm, tumor_tpm, tumor_count in (("B", 0.5, 1.0, 0), ("A", 0.0, 2.0, 320)):
                row = {"patient_id": patient, "rbp_gene": rbp, "candidate_edge_count": 640}
                for cohort, tpm, count in (("normal", normal_tpm, 0), ("tumor", tumor_tpm, tumor_count)):
                    row.update({
                        f"{cohort}_rbp_gene_tpm": tpm,
                        f"{cohort}_rbp_log2_tpm": np.log2(tpm + 0.001),
                        f"{cohort}_rbp_available": tpm >= 1,
                        f"{cohort}_active_edge_count": count,
                        f"{cohort}_active_edge_fraction": count / 640,
                    })
                rows.append(row)
        self.changes = pd.DataFrame(rows)

    def test_gate_closed_mask_preserves_inclusive_boundary_and_true_zero(self) -> None:
        import matplotlib.pyplot as plt

        validate_paired_values(self.changes, self.config)
        figure = plot_paired_heatmaps(self.changes, self.config, self.order)
        active = figure.axes[1].images[0].get_array()
        self.assertTrue(active.mask[:, [0, 2, 4]].all())
        self.assertFalse(active.mask[:, [1, 3, 5]].any())
        self.assertEqual(active[0, 1], 0.5)
        self.assertEqual(active[1, 1], 0)
        self.assertAlmostEqual(figure.axes[0].images[0].norm.vcenter, np.log2(1.001))
        self.assertFalse(np.array_equal(
            figure.axes[1].images[0].cmap(np.ma.masked), figure.axes[1].images[0].cmap(0.0),
        ))
        plt.close(figure)

    def test_invalid_flags_or_fraction_fail_before_plotting(self) -> None:
        invalid = self.changes.copy()
        invalid["normal_rbp_available"] = "False"
        with self.assertRaisesRegex(ValueError, "booleans"):
            validate_paired_values(invalid, self.config)
        invalid = self.changes.copy()
        invalid.loc[0, "tumor_active_edge_fraction"] = 0.3
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_paired_values(invalid, self.config)

    def test_failed_pdf_render_does_not_publish_png(self) -> None:
        import matplotlib.pyplot as plt

        figure = plt.figure()
        original = figure.savefig

        def save_or_fail(path: Path, dpi: int | None, bbox_inches: str) -> None:
            if path.suffix == ".pdf":
                raise OSError("simulated PDF failure")
            original(path, dpi=dpi, bbox_inches=bbox_inches)

        with tempfile.TemporaryDirectory() as name:
            base = Path(name) / "paired"
            with patch.object(figure, "savefig", side_effect=save_or_fail):
                with self.assertRaisesRegex(OSError, "simulated"):
                    save_figure(figure, base)
            self.assertFalse(base.with_suffix(".png").exists())
            self.assertFalse(base.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
