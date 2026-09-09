"""Tests for literature-anchor isoform selection and CORAL ranking."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "analyses" / "luad_analysis" / "scripts"
SCRIPT = SCRIPTS / "build_literature_anchors.py"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("build_literature_anchors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANCHORS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANCHORS
SPEC.loader.exec_module(ANCHORS)


class LiteratureAnchorTests(unittest.TestCase):
    """Cover outcome-independent representative selection and rank definitions."""

    def test_representative_uses_median_tpm_then_stable_id(self) -> None:
        """Ignore ineligible sequences and break expression ties by transcript ID."""
        anchors = pd.DataFrame({"target_gene_name": ["TARGET"]})
        transcripts = pd.DataFrame(
            {
                "gene_name": ["TARGET"] * 4,
                "gene_id": ["G"] * 4,
                "transcript_id": ["T3", "T2", "T1", "T0"],
                "RNA_id": ["r3", "r2", "r1", "r0"],
                "sequence_length_nt": [100] * 4,
                "tumor_median_tpm": [20.0, 10.0, 10.0, 100.0],
                "tumor_tpm_ge_1_count": [10] * 4,
                "normal_tpm_ge_1_count": [2] * 4,
                "sequence_eligible": [True, True, True, False],
                "primary_eligible": [True] * 4,
                "sensitivity_eligible": [True] * 4,
            }
        )
        selected = ANCHORS.select_representative_transcripts(anchors, transcripts)
        self.assertEqual(selected.iloc[0]["representative_transcript_id"], "T3")

        transcripts.loc[0, "tumor_median_tpm"] = 10.0
        selected = ANCHORS.select_representative_transcripts(anchors, transcripts)
        self.assertEqual(selected.iloc[0]["representative_transcript_id"], "T1")

    def test_gene_rank_and_representative_rank_remain_distinct(self) -> None:
        """Rank the expression representative and best-scoring gene isoform separately."""
        anchors = pd.DataFrame(
            {
                "anchor_order": [1],
                "rbp_gene": ["RBP"],
                "Protein_id": ["P"],
                "target_gene_name": ["TARGET"],
                "pmid": ["1"],
                "doi": ["doi"],
                "citation_url": ["https://example.test"],
                "evidence_context": ["human cells"],
                "evidence_type": ["RIP-qPCR"],
                "evidence_summary": ["binding"],
            }
        )
        transcripts = pd.DataFrame(
            {
                "gene_name": ["TARGET", "TARGET", "OTHER"],
                "gene_id": ["G1", "G1", "G2"],
                "transcript_id": ["T1", "T2", "T3"],
                "RNA_id": ["r1", "r2", "r3"],
                "sequence_length_nt": [100, 200, 300],
                "transcript_biotype": ["protein_coding"] * 3,
                "tumor_median_tpm": [10.0, 5.0, 1.0],
                "tumor_tpm_ge_1_count": [10, 9, 8],
                "normal_tpm_ge_1_count": [2, 2, 2],
                "sequence_eligible": [True] * 3,
                "primary_eligible": [True] * 3,
                "sensitivity_eligible": [True] * 3,
            }
        )
        predictions = pd.DataFrame(
            {
                "RNA_id": ["r1", "r2", "r3"],
                "Protein_id": ["P"] * 3,
                "logit_margin": [1.0, 3.0, 2.0],
                "interaction_probability": [0.7, 0.9, 0.8],
                "prediction": [1, 1, 1],
            }
        )
        exposure = pd.DataFrame(
            {
                "RNA_id": ["r1", "r2", "r3"],
                "Protein_id": ["P"] * 3,
                "exposure_class": ["component_only"] * 3,
                "training_row_index": pd.array([None] * 3, dtype="Int64"),
                "training_label": pd.array([None] * 3, dtype="Int8"),
                "rna_identity": [None] * 3,
                "protein_identity": [None] * 3,
            }
        )
        proteins = pd.DataFrame({"gene_symbol": ["RBP"], "Protein_id": ["P"]})
        output = ANCHORS.build_output(
            anchors, transcripts, proteins, predictions, exposure,
            transcripts[["transcript_id", "primary_eligible", "sensitivity_eligible"]].assign(Protein_id="P")
        ).iloc[0]
        self.assertEqual(output["representative_transcript_id"], "T1")
        self.assertEqual(output["primary_transcript_rank"], 3)
        self.assertEqual(output["gene_score_transcript_id"], "T2")
        self.assertEqual(output["primary_gene_rank"], 1)
        self.assertFalse(output["representative_is_gene_score_isoform"])
        eligibility = transcripts[["transcript_id", "primary_eligible", "sensitivity_eligible"]].assign(Protein_id="P")
        with self.assertRaisesRegex(ValueError, "missing classifications"):
            ANCHORS.build_output(anchors, transcripts, proteins, predictions, exposure.iloc[1:], eligibility)
        eligibility.loc[eligibility.transcript_id == "T1", "primary_eligible"] = False
        outside = ANCHORS.build_output(anchors, transcripts, proteins, predictions, exposure, eligibility).iloc[0]
        self.assertFalse(outside["top1pct_candidate"])
        self.assertEqual(outside["virtual_insertion_rank"], 3)
        self.assertTrue(pd.isna(outside["primary_transcript_rank"]))



if __name__ == "__main__":
    unittest.main()
