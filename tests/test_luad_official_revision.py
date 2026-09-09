"""Regression checks for the revised official LUAD design and artifact contracts."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analyses/luad_analysis/scripts"), str(ROOT / "src"), str(ROOT)]
from _analysis_io import load_config, staged_outputs, verify_manifest, write_table
from _rankings import rank_transcripts, compare_rankings
from _training_exposure import MatchMetrics, best_row_key
from prepare_luad_inputs import build_eligibility
from coral.prediction_io import read_predictions, sha256_file
from scripts.predict_all_vs_all import write_part_atomic, write_rank_manifest


class OfficialDesignTests(unittest.TestCase):
    """Protect RBP coding proxies and independent eligible isoform universes."""

    def test_coding_proxy_and_positive_tumor_prevalence(self) -> None:
        transcripts = pd.DataFrame({
            "transcript_id": ["coding", "noncoding", "target", "normal_only"],
            "RNA_id": ["r1", "r2", "r3", "r4"],
            "gene_id": ["RBP", "RBP", "TARGET", "OTHER"],
            "gene_name": ["R", "R", "T", "O"],
            "transcript_biotype": ["protein_coding", "retained_intron", "lncRNA", "pseudogene"],
            "sequence_eligible": [False, True, True, True],
        })
        samples = pd.DataFrame({"sample_id": ["T1", "T2", "T3", "N"],
                                "sample_type": ["primary_tumor"] * 3 + ["solid_tissue_normal"]})
        # Coding RNA need not qualify as a target to contribute to the proxy.
        tpm = np.array([[2, 0, 1, 0], [0, 100, 0, 0], [0, 100, 0, 0], [2, 100, 0, 10]])
        config = {"rbp_genes": {"P": {"gene_id": "RBP", "gene_symbol": "R", "transcript_count": 2}},
                  "rbp_tpm_cutoff": 1, "prevalence_fraction": 0.5, "target_tpm_cutoff": 1, "sensitivity_tpm_cutoff": 0.5}
        eligibility, proxy = build_eligibility(transcripts, np.log2(tpm + 0.001), samples, config)
        self.assertEqual(proxy.rbp_available.tolist(), [True, False, False, True])
        self.assertEqual(eligibility.rbp_positive_tumors.unique().tolist(), [1])
        flags = eligibility.set_index("transcript_id").primary_eligible
        self.assertTrue(flags["target"])
        self.assertFalse(flags["normal_only"])
        self.assertFalse(flags["coding"])

    def test_rbp_specific_rankings_ties_and_overlap_units(self) -> None:
        transcripts = pd.DataFrame({"transcript_id": ["t1", "t2", "t3", "t4"], "RNA_id": ["r1", "r1", "r2", "r3"],
            "gene_id": ["g1", "g1", "g2", "g3"], "gene_name": ["G1", "G1", "G2", "G3"],
            "transcript_biotype": ["x"] * 4, "sequence_length_nt": [10] * 4})
        eligibility = pd.DataFrame([(tid, protein, flag) for protein, flags in (("P1", [True]*4), ("P2", [True, True, True, False])) for tid, flag in zip(transcripts.transcript_id, flags)], columns=["transcript_id", "Protein_id", "primary_eligible"])
        scores = pd.DataFrame([(rna, protein, margin) for protein, values in (("P1", [4., 2., 0.]), ("P2", [0., 2., 4.])) for rna, margin in zip(["r1", "r2", "r3"], values)], columns=["RNA_id", "Protein_id", "logit_margin"])
        scores["interaction_probability"] = 1 / (1 + np.exp(-scores.logit_margin))
        scores["prediction"] = (scores.logit_margin > 0).astype(int)
        ranks = rank_transcripts(transcripts, eligibility, scores, top_fraction=0.25)
        self.assertEqual(ranks.groupby("Protein_id").size().to_dict(), {"P1": 4, "P2": 3})
        self.assertEqual(ranks.groupby("Protein_id").top_candidate.sum().to_dict(), {"P1": 1, "P2": 1})
        self.assertEqual(ranks.loc[ranks.Protein_id == "P1"].tie_inclusive_candidate.sum(), 2)
        corr, overlap, contrasts = compare_rankings(ranks)
        self.assertEqual(corr.set_index("unit").common_count.to_dict(), {"transcript": 3, "sequence": 2})
        np.testing.assert_allclose(corr.spearman_rho, -1)
        self.assertEqual(set(overlap.unit), {"transcript", "sequence", "gene"})
        self.assertEqual(contrasts.transcript_id.tolist(), ["t1", "t2", "t3"])
        self.assertEqual(contrasts.percentile_a.iloc[0], contrasts.percentile_a.iloc[1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            rank_transcripts(transcripts, eligibility, scores.iloc[1:])

    def test_zero_similarity_threshold_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "thresholds"):
            best_row_key(0, MatchMetrics(1, 1, True), MatchMetrics(1, 1, True), 0, 0.6, 0.8)

    def test_invalid_config_fraction_fails(self) -> None:
        config = load_config()
        config["top_fraction"] = 1.1
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "top_fraction"):
                load_config(path)


class ArtifactContractTests(unittest.TestCase):
    """Do not consume partial runs or publish falsely complete stages."""

    def test_failed_stage_preserves_previous_generation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = root / "stage.json"
            with staged_outputs(root, manifest, {}) as stage:
                write_table(pd.DataFrame({"value": [1]}), stage / "table.tsv")
            original = manifest.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with staged_outputs(root, manifest, {}) as stage:
                    write_table(pd.DataFrame({"value": [2]}), stage / "table.tsv")
                    raise RuntimeError("failed calculation")
            self.assertEqual(original, manifest.read_bytes())
            verify_manifest(manifest, root)
            self.assertEqual(pd.read_csv(root / "table.tsv", sep="\t").value.tolist(), [1])

    def test_changed_upstream_input_invalidates_completed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "input.txt"
            source.write_text("first")
            manifest = root / "stage.json"
            with staged_outputs(root, manifest, {"inputs": {str(source): sha256_file(source)}}) as stage:
                write_table(pd.DataFrame({"value": [1]}), stage / "table.tsv")
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "input changed"):
                verify_manifest(manifest, root)

    def test_completed_prediction_run_requires_exact_cartesian_product(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rnas, proteins = root / "rnas.parquet", root / "proteins.tsv"
            pd.DataFrame({"RNA_id": ["r1", "r2"]}).to_parquet(rnas, index=False)
            pd.DataFrame({"Protein_id": ["P1", "P2"]}).to_csv(proteins, sep="\t", index=False)
            frame = pd.DataFrame({"RNA_id": ["r1", "r1", "r2", "r2"], "Protein_id": ["P1", "P2"]*2, "logit_margin": [0.]*4, "interaction_probability": [0.5]*4, "prediction": [0]*4})
            config = {"rank": 0, "world_size": 1, "precision": "float32", "run_hash": "test",
                "rna_manifest_sha256": sha256_file(rnas), "protein_manifest_sha256": sha256_file(proteins)}
            part = root / "rank-00000-part-000000.parquet"
            write_part_atomic(frame, part, "test", 0)
            write_rank_manifest(root, 0, config, 4)
            self.assertEqual(len(read_predictions(root, rnas, proteins)[0]), 4)
            # Even a newly hashed manifest cannot legitimize a missing Cartesian pair.
            write_part_atomic(frame.iloc[:-1], part, "test", 0)
            write_rank_manifest(root, 0, config, 3)
            with self.assertRaisesRegex(ValueError, "Cartesian"):
                read_predictions(root, rnas, proteins)
            # Untracked parts must also be rejected.
            write_part_atomic(frame, part, "test", 0)
            write_rank_manifest(root, 0, config, 4)
            write_part_atomic(frame, root / "rank-00000-part-000001.parquet", "test", 0)
            with self.assertRaisesRegex(ValueError, "unlisted"):
                read_predictions(root, rnas, proteins)


if __name__ == "__main__":
    unittest.main()
