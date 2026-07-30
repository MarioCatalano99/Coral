"""Assert the renamed CORAL model is bit-identical to the architecture it replaced.

`src/coral/model.py` carries the semantics of what used to be
`mechanistic_interpretability/coral/splitProteinCrossmodel_new.py` under the public
`Coral*` class names. `tests/reference/crossmodel_reference.py` is a verbatim frozen
copy of that original module and is the ground truth here: it must never be edited.

Two tiers:

- default — the custom stack only (cross-modality encoder, pooler, pre-training
  heads, config). No encoder downloads, no GPU, runs in seconds.
- `slow`  — the full `CoralPretraining` against the real DNABERT-2 / ESM-2 backbones
  and the released checkpoint. Run with `pytest -m slow`.

Tolerance is exact (`rtol=0, atol=0`). A rename must not move a single bit; if this
fails, an expression was changed, not just a name.
"""

import inspect
from pathlib import Path

import pytest
import torch

from tests.reference import crossmodel_reference as ref

from coral import model as new

BATCH, RNA_LEN, PROT_LEN = 3, 11, 17
SEED = 42


def build_config(module):
    """Same config on both sides; dropout > 0 is fine because we only run in eval()."""
    cls = module.CoralConfig if hasattr(module, "CoralConfig") else module.CrossModelConfig
    return cls(
        rna_encoder_checkpoint=None,
        protein_encoder_checkpoint=None,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2,
    )


def seeded(fn):
    """Build a module under a fixed seed so both sides get identical init."""
    torch.manual_seed(SEED)
    return fn()


def make_inputs():
    """Random feats plus masks that include a fully-padded row.

    The fully-padded row matters: it drives an all-`-inf` softmax row to NaN, which
    the model flattens with `nan_to_num`. That path is where a "harmless" rewrite of
    the masking or softmax call would show up.
    """
    torch.manual_seed(SEED + 1)
    cfg = build_config(new)
    rna_feats = torch.randn(BATCH, RNA_LEN, cfg.hidden_size)
    protein_feats = torch.randn(BATCH, PROT_LEN, cfg.hidden_size)

    rna_mask = torch.ones(BATCH, RNA_LEN)
    rna_mask[0, 7:] = 0.0            # ordinary right padding
    rna_mask[2, :] = 0.0             # fully padded -> NaN softmax row
    protein_mask = torch.ones(BATCH, PROT_LEN)
    protein_mask[1, 12:] = 0.0

    return rna_feats, rna_mask, protein_feats, protein_mask


def assert_exact(actual, expected, what):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                               msg=lambda m: f"{what} differs:\n{m}")


# --- Config -----------------------------------------------------------------


def test_config_fields_match():
    """Every config field carries over; only `model_type` is renamed."""
    a, b = build_config(new).to_dict(), build_config(ref).to_dict()
    assert a.pop("model_type") == "Coral"
    assert b.pop("model_type") == "CrossModalityRPIpred"
    a.pop("transformers_version", None)
    b.pop("transformers_version", None)
    assert a == b


def test_config_signature_matches():
    """No parameter added, removed, reordered, or given a different default."""
    assert (inspect.signature(new.CoralConfig.__init__)
            == inspect.signature(ref.CrossModelConfig.__init__))


def test_activation_table_matches():
    assert new.ACT2FN.keys() == ref.ACT2FN.keys()
    for key in new.ACT2FN:
        # `swish` is redefined per-module, so compare behaviour not identity.
        x = torch.randn(64, dtype=torch.float64)
        assert_exact(new.ACT2FN[key](x), ref.ACT2FN[key](x), f"ACT2FN[{key!r}]")


# --- Custom stack -----------------------------------------------------------


def test_cross_modality_encoder_is_bit_identical():
    cfg_new, cfg_ref = build_config(new), build_config(ref)
    enc_new = seeded(lambda: new.CrossModalityEncoder(cfg_new)).eval()
    enc_ref = seeded(lambda: ref.CrossModalityEncoder(cfg_ref)).eval()

    # Identical construction under an identical seed also proves submodule
    # declaration order is unchanged.
    assert enc_new.state_dict().keys() == enc_ref.state_dict().keys()
    for key, value in enc_new.state_dict().items():
        assert_exact(value, enc_ref.state_dict()[key], f"init weight {key}")

    rna_feats, rna_mask, protein_feats, protein_mask = make_inputs()
    with torch.no_grad():
        out_new = enc_new(rna_feats, rna_mask, protein_feats, protein_mask,
                          output_attentions=True)
        out_ref = enc_ref(rna_feats, rna_mask, protein_feats, protein_mask,
                          output_attentions=True)

    assert len(out_new) == len(out_ref) == 4
    assert_exact(out_new[0], out_ref[0], "rna_feats")
    assert_exact(out_new[1], out_ref[1], "protein_feats")
    for i, name in enumerate(("rna->protein", "protein->rna")):
        assert_exact(out_new[2][i], out_ref[2][i], f"cross attention {name}")
    for i, name in enumerate(("rna self", "protein self")):
        assert_exact(out_new[3][i], out_ref[3][i], f"self attention {name}")


def test_nan_path_is_exercised_and_finite():
    """Guard the guard: the fully-padded row must actually produce zeros, not NaN."""
    cfg = build_config(new)
    enc = seeded(lambda: new.CrossModalityEncoder(cfg)).eval()
    rna_feats, rna_mask, protein_feats, protein_mask = make_inputs()
    with torch.no_grad():
        out = enc(rna_feats, rna_mask, protein_feats, protein_mask, output_attentions=True)
    assert torch.isfinite(out[0]).all(), "rna_feats contains NaN/inf"
    cross_rna = out[2][0]
    assert torch.isfinite(cross_rna).all(), "cross attention contains NaN/inf"
    assert (cross_rna[2] == 0).all(), "fully-padded row should attend nowhere"


def test_pooler_is_bit_identical():
    pool_new = seeded(lambda: new.BertPooler(build_config(new))).eval()
    pool_ref = seeded(lambda: ref.BertPooler(build_config(ref))).eval()

    rna_feats, rna_mask, protein_feats, protein_mask = make_inputs()
    with torch.no_grad():
        assert_exact(pool_new(rna_feats, rna_mask, protein_feats, protein_mask),
                     pool_ref(rna_feats, rna_mask, protein_feats, protein_mask),
                     "pooled_output")


def test_pretraining_heads_are_bit_identical():
    cfg_new, cfg_ref = build_config(new), build_config(ref)
    torch.manual_seed(SEED + 2)
    rna_emb = torch.nn.Parameter(torch.randn(cfg_new.rna_vocab_size, cfg_new.hidden_size))
    prot_emb = torch.nn.Parameter(
        torch.randn(cfg_new.protein_vocab_size, cfg_new.protein_encoder_hidden_size))

    heads_new = seeded(lambda: new.BertPreTrainingHeads(cfg_new, rna_emb, prot_emb)).eval()
    heads_ref = seeded(lambda: ref.BertPreTrainingHeads(cfg_ref, rna_emb, prot_emb)).eval()

    torch.manual_seed(SEED + 3)
    rna_seq = torch.randn(BATCH, RNA_LEN, cfg_new.hidden_size)
    prot_seq = torch.randn(BATCH, PROT_LEN, cfg_new.protein_encoder_hidden_size)
    pooled = torch.randn(BATCH, cfg_new.hidden_size)

    with torch.no_grad():
        out_new = heads_new(rna_seq, prot_seq, pooled)
        out_ref = heads_ref(rna_seq, prot_seq, pooled)
    for got, want, name in zip(out_new, out_ref,
                               ("rna_prediction_scores", "protein_prediction_scores",
                                "seq_relationship_score")):
        assert_exact(got, want, name)


def test_soft_cross_entropy_is_bit_identical():
    torch.manual_seed(SEED + 4)
    logits = torch.randn(BATCH, 2)
    targets = torch.softmax(torch.randn(BATCH, 2), dim=-1)
    assert_exact(new.soft_cross_entropy(logits, targets),
                 ref.soft_cross_entropy(logits, targets), "soft_cross_entropy")


def test_reverse_projection_is_declared():
    """The load-bearing architectural difference from the old public model.py."""
    src = inspect.getsource(new.CoralPretraining.__init__)
    assert "self.reverse_projection = nn.Linear(config.hidden_size," in src
    assert "self.reverse_projection(protein_output)" in inspect.getsource(
        new.CoralPretraining.forward)
    assert new.CoralPretraining._supports_param_buffer_assignment is False


def test_forward_signature_matches():
    """`matched_labels` / `soft_matched_labels` must survive the rename verbatim."""
    assert (inspect.signature(new.CoralPretraining.forward)
            == inspect.signature(ref.CrossModalityRPIpredPretraining.forward))


# --- Full model (needs the real encoders) -----------------------------------


@pytest.fixture(scope="module")
def encoder_checkpoints():
    dnabert = Path(__file__).resolve().parent.parent / "DNABERT-2-117M"
    if not (dnabert / "pytorch_model.bin").exists():
        pytest.skip(f"DNABERT-2 weights missing at {dnabert}"
                    " — run scripts/download_dnabert_weights.sh")
    return str(dnabert), "facebook/esm2_t30_150M_UR50D"


@pytest.mark.slow
def test_full_model_state_dict_keys_match(encoder_checkpoints):
    """The rename must not move a single state-dict key."""
    rna_ckpt, protein_ckpt = encoder_checkpoints
    model_new = seeded(lambda: new.CoralPretraining(
        config=new.CoralConfig(rna_encoder_checkpoint=rna_ckpt,
                               protein_encoder_checkpoint=protein_ckpt)))
    model_ref = seeded(lambda: ref.CrossModalityRPIpredPretraining(
        config=ref.CrossModelConfig(rna_encoder_checkpoint=rna_ckpt,
                                    protein_encoder_checkpoint=protein_ckpt)))

    keys_new = set(model_new.state_dict())
    keys_ref = set(model_ref.state_dict())
    assert keys_new == keys_ref, (
        f"missing: {sorted(keys_ref - keys_new)[:10]} "
        f"unexpected: {sorted(keys_new - keys_ref)[:10]}")
    assert "reverse_projection.weight" in keys_new


@pytest.mark.slow
def test_full_model_forward_is_bit_identical(encoder_checkpoints):
    rna_ckpt, protein_ckpt = encoder_checkpoints
    model_new = seeded(lambda: new.CoralPretraining(
        config=new.CoralConfig(rna_encoder_checkpoint=rna_ckpt,
                               protein_encoder_checkpoint=protein_ckpt))).eval()
    model_ref = seeded(lambda: ref.CrossModalityRPIpredPretraining(
        config=ref.CrossModelConfig(rna_encoder_checkpoint=rna_ckpt,
                                    protein_encoder_checkpoint=protein_ckpt))).eval()
    model_ref.load_state_dict(model_new.state_dict())

    torch.manual_seed(SEED + 5)
    rna_batch = {
        "input_ids": torch.randint(5, 4000, (2, 9)),
        "attention_mask": torch.ones(2, 9, dtype=torch.long),
    }
    protein_batch = {
        "input_ids": torch.randint(4, 30, (2, 13)),
        "attention_mask": torch.ones(2, 13, dtype=torch.long),
    }
    protein_batch["attention_mask"][1, 10:] = 0

    with torch.no_grad():
        out_new = model_new(rna_batch, protein_batch, output_attentions=True)
        out_ref = model_ref(rna_batch, protein_batch, output_attentions=True)

    assert_exact(out_new[0], out_ref[0], "predictions")
    assert_exact(out_new[1], out_ref[1], "match_probabilities")
    for i in range(2):
        assert_exact(out_new[2][i], out_ref[2][i], f"cross attention [{i}]")
        assert_exact(out_new[3][i], out_ref[3][i], f"self attention [{i}]")
