"""Tests for the reproducible LUAD training-exposure audit."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analyses" / "luad_analysis" / "scripts" / "audit_training_exposure.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("audit_training_exposure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class TrainingExposureTests(unittest.TestCase):
    """Cover exact, near, component-only, unseen, and tie behavior."""

    def setUp(self) -> None:
        """Build a small training-row fixture."""
        self.train = pd.DataFrame(
            {
                "training_row_index": [0, 1],
                "RNA_id": ["train-rna-a", "train-rna-b"],
                "Protein_id": ["train-protein-a", "train-protein-b"],
                "labels": [1, 0],
                "rna_near_target": ["trna-a", "trna-b"],
                "protein_near_target": ["tprotein-a", "tprotein-b"],
            }
        )
        self.rna_rows = {"trna-a": {0}, "trna-b": {1}}
        self.protein_rows = {"tprotein-a": {0}, "tprotein-b": {1}}
        self.near = AUDIT.MatchMetrics(0.8, 0.8, False)
        self.thresholds = (0.8, 0.6, 0.8)

    def classify(
        self,
        rna_matches: dict,
        protein_matches: dict,
        rna_exact: dict | None = None,
        protein_exact: dict | None = None,
    ) -> dict:
        """Classify the fixture query pair."""
        return AUDIT.classify_pair(
            "query-rna",
            "query-protein",
            self.train,
            rna_matches,
            protein_matches,
            rna_exact or {},
            protein_exact or {},
            self.rna_rows,
            self.protein_rows,
            self.thresholds,
        )

    def test_normalization_is_near_only_and_conservative(self) -> None:
        """Normalize uracil and ambiguous symbols without changing raw hashes."""
        self.assertEqual(AUDIT.normalize_rna("aUgX"), "ATGN")
        self.assertEqual(AUDIT.normalize_protein("acuZ"), "ACXX")
        self.assertNotEqual(AUDIT.sha256_text("AUT"), AUDIT.sha256_text("ATT"))

    def test_alignment_metrics_accept_inclusive_boundaries(self) -> None:
        """Calculate unrounded identity and shorter-sequence coverage."""
        Row = namedtuple(
            "Row", "nident alnlen qstart qend qlen tstart tend tlen"
        )
        metrics = AUDIT.alignment_metrics(Row(8, 10, 1, 8, 10, 3, 10, 20))
        self.assertEqual(metrics.identity, 0.8)
        self.assertEqual(metrics.coverage, 0.8)

    def test_exact_pair_uses_raw_row_matches(self) -> None:
        """Exact row maps take precedence over near target matches."""
        result = self.classify({}, {}, {"query-rna": {0}}, {"query-protein": {0}})
        self.assertEqual(result["exposure_class"], "exact_pair")
        self.assertEqual(result["training_row_index"], 0)
        self.assertEqual(result["training_label"], 1)

    def test_near_pair_retains_label(self) -> None:
        """Two component matches resolving to one row form a near pair."""
        result = self.classify(
            {"query-rna": {"trna-a": self.near}},
            {"query-protein": {"tprotein-a": self.near}},
        )
        self.assertEqual(result["exposure_class"], "near_pair")
        self.assertEqual(result["training_label"], 1)

    def test_component_only_subtypes_and_unseen(self) -> None:
        """Distinguish one-sided, separately seen, and wholly unseen pairs."""
        rna_a = {"query-rna": {"trna-a": self.near}}
        protein_a = {"query-protein": {"tprotein-a": self.near}}
        protein_b = {"query-protein": {"tprotein-b": self.near}}
        cases = [
            (rna_a, {}, "component_only", "rna_only"),
            ({}, protein_a, "component_only", "protein_only"),
            (rna_a, protein_b, "component_only", "both_different_rows"),
            ({}, {}, "unseen_pair", None),
        ]
        for rna_matches, protein_matches, expected_class, expected_subtype in cases:
            with self.subTest(expected_subtype=expected_subtype):
                result = self.classify(rna_matches, protein_matches)
                self.assertEqual(result["exposure_class"], expected_class)
                self.assertEqual(result["component_subtype"], expected_subtype)

    def test_best_row_is_deterministic_and_label_ambiguity_is_retained(self) -> None:
        """Choose the stronger row while counting both source labels."""
        self.rna_rows = {"shared": {0, 1}}
        self.protein_rows = {"shared-protein": {0, 1}}
        self.train["rna_near_target"] = "shared"
        self.train["protein_near_target"] = "shared-protein"
        stronger = AUDIT.MatchMetrics(0.9, 0.9, False)
        result = self.classify(
            {"query-rna": {"shared": stronger}},
            {"query-protein": {"shared-protein": stronger}},
        )
        self.assertEqual(result["training_row_index"], 0)
        self.assertEqual(result["matching_positive_rows"], 1)
        self.assertEqual(result["matching_generated_negative_rows"], 1)
        self.assertTrue(result["training_label_ambiguous"])

    def test_summary_contains_all_classes_and_exact_percentages(self) -> None:
        """Emit mutually exclusive class counts with unrounded proportions."""
        exposure = pd.DataFrame(
            {
                "Protein_id": ["p"] * 4,
                "primary_eligible": [True] * 4,
                "exposure_class": list(AUDIT.EXPOSURE_CLASSES),
            }
        )
        summary = AUDIT.make_summary(exposure)
        overall = summary.loc[summary["scope"] == "all"]
        self.assertEqual(overall["rows"].tolist(), [1, 1, 1, 1])
        self.assertEqual(overall["percentage"].tolist(), [25.0] * 4)

    @unittest.skipUnless(shutil.which("mmseqs"), "MMseqs2 is not installed")
    def test_tiny_forward_only_mmseqs_search(self) -> None:
        """Exercise the pinned command shape on a tiny nucleotide search."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            query = root / "query.fa"
            target = root / "target.fa"
            output = root / "matches.tsv"
            sequence = "AAAACCCCGGGTTTAC"
            AUDIT.write_fasta(query, [("query", sequence)])
            AUDIT.write_fasta(
                target,
                [("target", sequence), ("reverse", "GTAAACCCGGGGTTTT")],
            )
            AUDIT.run_mmseqs(
                "mmseqs",
                query,
                target,
                output,
                root / "tmp",
                search_type=3,
                min_identity=0.8,
                min_coverage=0.8,
                coverage_mode=2,
                sensitivity=7.5,
                evalue=1e6,
                max_sequences=2,
                threads=1,
                forward_only=True,
            )
            self.assertTrue(output.read_text(), "MMseqs produced no alignment rows")
            matches = AUDIT.load_component_matches(output, 0.8, 0.8)
            self.assertIn("query", matches, output.read_text())
            self.assertIn("target", matches["query"])
            self.assertNotIn("reverse", matches["query"])


if __name__ == "__main__":
    unittest.main()
