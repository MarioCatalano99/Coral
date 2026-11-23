"""Tests for lookup-table all-vs-all inference without external checkpoints."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coral.inference import iter_all_vs_all_predictions, predict_all_vs_all_interactions
from coral.data import DataCollatorForInference, validate_sequence_frame
from coral.model import CoralPretraining
from coral.prediction_io import checkpoint_hash, read_predictions, sha256_file
from scripts.predict_all_vs_all import (
    read_completed_rnas,
    write_part_atomic,
    write_rank_manifest,
)


class FakeTokenizer:
    """Minimal tokenizer interface needed by the inference lookup and collator."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        sequences: list[str],
        *,
        padding: bool,
        truncation: bool,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, list[list[int]]] | dict[str, torch.Tensor]:
        del truncation, return_token_type_ids
        self.calls.append(sequences)
        ids = [[1, *[ord(symbol) % 23 + 3 for symbol in sequence], 2]
               for sequence in sequences]
        if return_tensors == "pt":
            return self.pad(
                [{"input_ids": values} for values in ids], padding=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask, return_tensors=return_tensors,
            )
        return {"input_ids": ids}

    def pad(
        self,
        features: list[dict[str, list[int]]],
        *,
        padding: bool,
        pad_to_multiple_of: int | None,
        return_attention_mask: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        if not padding or not return_attention_mask or return_tensors != "pt":
            raise AssertionError("unexpected padding options")
        maximum = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of:
            maximum = (
                (maximum + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of
        ids = []
        masks = []
        for feature in features:
            values = feature["input_ids"]
            missing = maximum - len(values)
            ids.append([*values, *([0] * missing)])
            masks.append([1] * len(values) + [0] * missing)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class FakeCoral(torch.nn.Module):
    """Small model exposing CORAL's interaction-classifier shape."""

    def __init__(self) -> None:
        super().__init__()
        self.cls = torch.nn.Module()
        self.cls.seq_relationship = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.cls.seq_relationship.weight.copy_(
                torch.tensor([[0.05, -0.03], [-0.02, 0.04]])
            )
            self.cls.seq_relationship.bias.copy_(torch.tensor([0.01, -0.01]))

    def forward(
        self,
        rna_batch: dict[str, torch.Tensor],
        protein_batch: dict[str, torch.Tensor],
        return_logit_margin: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        rna_mask = rna_batch["attention_mask"]
        protein_mask = protein_batch["attention_mask"]
        rna_mean = (rna_batch["input_ids"] * rna_mask).sum(1) / rna_mask.sum(1)
        protein_mean = (protein_batch["input_ids"] * protein_mask).sum(1) / protein_mask.sum(1)
        pooled = torch.stack((rna_mean, protein_mean), dim=1).float()
        logits = self.cls.seq_relationship(pooled)
        probabilities = torch.softmax(logits, dim=1)
        outputs = (probabilities.argmax(dim=1), probabilities)
        if return_logit_margin:
            outputs += (logits[:, 1] - logits[:, 0],)
        return outputs


class FakeBackbone(torch.nn.Module):
    """Return fixed shapes expected by CoralPretraining.forward."""

    def forward(
        self,
        rna_batch: dict[str, torch.Tensor],
        protein_batch: dict[str, torch.Tensor],
        output_attentions: bool = False,
    ) -> tuple:
        del protein_batch, output_attentions
        batch_size = len(rna_batch["input_ids"])
        rna_output = torch.zeros((batch_size, 2, 2))
        protein_output = torch.zeros((batch_size, 2, 2))
        pooled_output = torch.tensor([[1.0, 2.0]]).repeat(batch_size, 1)
        return (rna_output, protein_output), pooled_output


class FakePretrainingHeads(torch.nn.Module):
    """Provide token heads and a deterministic two-class classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.seq_relationship = torch.nn.Linear(2, 2)

    def forward(
        self,
        rna_output: torch.Tensor,
        protein_output: torch.Tensor,
        pooled_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.seq_relationship(pooled_output)
        return rna_output, protein_output, logits


class ForwardHarness(torch.nn.Module):
    """Exercise CoralPretraining.forward without loading either base encoder."""

    forward = CoralPretraining.forward

    def __init__(self) -> None:
        super().__init__()
        self.bert = FakeBackbone()
        self.reverse_projection = torch.nn.Identity()
        self.cls = FakePretrainingHeads()
        self.softmax = torch.nn.Softmax(dim=1)
        self.config = SimpleNamespace(rna_vocab_size=2, protein_vocab_size=2)


class LargeInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rnas = pd.DataFrame(
            {"RNA_id": ["rna-a", "rna-b"], "RNA_seqs": ["ACGT", "AAAAC"]}
        )
        self.proteins = pd.DataFrame(
            {"Protein_id": ["protein-a", "protein-b"], "Prot_seqs": ["ACD", "WY"]}
        )

    def test_collator_expands_each_rna_across_all_proteins(self) -> None:
        tokenizer = FakeTokenizer()
        collator = DataCollatorForInference(
            [[1, 5, 2], [1, 6, 7, 2]],
            [[1, 8, 2], [1, 9, 10, 2]],
            tokenizer,
            tokenizer,
        )
        batch = collator([{"rna_index": 1}, {"rna_index": 0}])
        self.assertEqual(batch["rna_indices"].tolist(), [1, 1, 0, 0])
        self.assertEqual(batch["protein_indices"].tolist(), [0, 1, 0, 1])
        self.assertEqual(tuple(batch["rna_input_ids"].shape), (4, 8))
        self.assertEqual(tuple(batch["protein_input_ids"].shape), (4, 8))

    def test_materialized_predictions_have_complete_minimal_schema(self) -> None:
        rna_tokenizer = FakeTokenizer()
        protein_tokenizer = FakeTokenizer()
        result = predict_all_vs_all_interactions(
            FakeCoral(),
            self.rnas,
            self.proteins,
            rna_tokenizer,
            protein_tokenizer,
            batch_size=1,
        )
        self.assertEqual(
            list(result.columns),
            [
                "RNA_id",
                "Protein_id",
                "logit_margin",
                "interaction_probability",
                "prediction",
            ],
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(
            set(zip(result["RNA_id"], result["Protein_id"])),
            {
                ("rna-a", "protein-a"),
                ("rna-a", "protein-b"),
                ("rna-b", "protein-a"),
                ("rna-b", "protein-b"),
            },
        )
        self.assertEqual(rna_tokenizer.calls, [["ACGT", "AAAAC"]])
        self.assertEqual(protein_tokenizer.calls, [["ACD", "WY"]])
        self.assertTrue(result["interaction_probability"].between(0.0, 1.0).all())
        self.assertTrue(
            (result["prediction"] == (result["logit_margin"] > 0).astype("int8")).all()
        )

    def test_preflight_reuses_lookups_and_checks_each_length_bucket(self) -> None:
        rnas = pd.concat([self.rnas, pd.DataFrame(
            {"RNA_id": ["rna-long"], "RNA_seqs": ["A" * 130]}
        )], ignore_index=True)
        rna_tokenizer, protein_tokenizer = FakeTokenizer(), FakeTokenizer()
        model = FakeCoral()
        diagnostics = {}
        result = pd.concat(list(iter_all_vs_all_predictions(
            model, rnas, self.proteins, rna_tokenizer, protein_tokenizer,
            batch_size=1, preflight=True, diagnostics=diagnostics,
        )), ignore_index=True)
        expected = predict_all_vs_all_interactions(
            model, rnas, self.proteins, FakeTokenizer(), FakeTokenizer(), batch_size=1,
        )
        pd.testing.assert_frame_equal(result, expected)
        # Only the representative direct checks follow the single lookup pass.
        self.assertEqual(rna_tokenizer.calls, [
            rnas.RNA_seqs.tolist(), ["AAAAC"] * 2, ["A" * 130] * 2,
        ])
        self.assertEqual(protein_tokenizer.calls, [self.proteins.Prot_seqs.tolist()] * 3)
        checks = diagnostics["preflight"]["checks"]
        self.assertEqual([check["rna_tokens"] for check in checks], [7, 132])
        self.assertTrue(all(check["max_padding_margin_error"] == 0 for check in checks))

    def test_preflight_numerical_failure_prevents_first_prediction_batch(self) -> None:
        for values, message in (
            ([0., 1., 0.], "repeated float32 scores differ"),
            ([0., 0., 1.], "padding-dependent float32 margin"),
        ):
            with self.subTest(message=message):
                scores = [(None, None, torch.full((2,), value)) for value in values]
                with patch("coral.inference._score_batch", side_effect=scores):
                    iterator = iter_all_vs_all_predictions(
                        FakeCoral(), self.rnas, self.proteins,
                        FakeTokenizer(), FakeTokenizer(), batch_size=1, preflight=True,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        next(iterator)

    def test_manifest_validation_rejects_duplicate_sequences(self) -> None:
        duplicated = pd.DataFrame(
            {"RNA_id": ["one", "two"], "RNA_seqs": ["ACGT", "ACGT"]}
        )
        with self.assertRaisesRegex(ValueError, "RNA_seqs values must be unique"):
            validate_sequence_frame(
                duplicated,
                id_column="RNA_id",
                sequence_column="RNA_seqs",
                alphabet=frozenset("ACGT"),
                max_length=10,
            )

    def test_model_logit_margin_is_opt_in_and_backward_compatible(self) -> None:
        model = ForwardHarness()
        batch = {
            "input_ids": torch.tensor([[1, 10, 2]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }
        default_outputs = model(batch, batch)
        margin_outputs = model(batch, batch, return_logit_margin=True)
        self.assertEqual(len(default_outputs), 2)
        self.assertEqual(len(margin_outputs), 3)
        logits = model.cls.seq_relationship(torch.tensor([[1.0, 2.0]]))
        expected_margin = logits[:, 1] - logits[:, 0]
        torch.testing.assert_close(margin_outputs[2], expected_margin)
        attention_outputs = model(
            batch,
            batch,
            output_attentions=True,
            return_logit_margin=True,
        )
        self.assertEqual(len(attention_outputs), 5)
        torch.testing.assert_close(attention_outputs[2], expected_margin)

    def test_rna_shards_are_disjoint_and_complete(self) -> None:
        frames = []
        for shard_index in range(2):
            frames.append(
                predict_all_vs_all_interactions(
                    FakeCoral(),
                    self.rnas,
                    self.proteins,
                    FakeTokenizer(),
                    FakeTokenizer(),
                    batch_size=1,
                    shard_index=shard_index,
                    num_shards=2,
                )
            )
        pairs = [set(zip(frame["RNA_id"], frame["Protein_id"])) for frame in frames]
        self.assertFalse(pairs[0] & pairs[1])
        self.assertEqual(len(pairs[0] | pairs[1]), 4)

    def test_atomic_parts_are_validated_for_resume(self) -> None:
        frame = pd.DataFrame(
            {
                "RNA_id": ["rna-a", "rna-a"],
                "Protein_id": ["protein-a", "protein-b"],
                "logit_margin": [-1.0, 1.0],
                "interaction_probability": [0.25, 0.75],
                "prediction": [0, 1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            part = output_dir / "rank-00000-part-000000.parquet"
            write_part_atomic(frame, part, "run-123", 0)
            completed = read_completed_rnas(
                output_dir, 0, "run-123", {"protein-a", "protein-b"}
            )
            self.assertEqual(completed, {"rna-a"})
            write_rank_manifest(
                output_dir,
                0,
                {"run_hash": "run-123"},
                expected_rows=2,
            )
            self.assertTrue((output_dir / "rank-00000-manifest.json").is_file())


    def test_completed_prediction_run_requires_exact_cartesian_product(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rnas, proteins = root / "rnas.parquet", root / "proteins.tsv"
            pd.DataFrame({"RNA_id": ["r1", "r2"]}).to_parquet(rnas, index=False)
            pd.DataFrame({"Protein_id": ["P1", "P2"]}).to_csv(proteins, sep="\t", index=False)
            frame = pd.DataFrame({"RNA_id": ["r1", "r1", "r2", "r2"], "Protein_id": ["P1", "P2"] * 2, "logit_margin": [0.] * 4,
                                  "interaction_probability": [0.5] * 4, "prediction": [0] * 4})
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

    def test_checkpoint_hash_tracks_names_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "adapter_config.json").write_text("{}")
            first = checkpoint_hash(root)
            self.assertEqual(first, checkpoint_hash(root))
            (root / "adapter_model.safetensors").write_bytes(b"other weights")
            self.assertNotEqual(first, checkpoint_hash(root))


if __name__ == "__main__":
    unittest.main()
