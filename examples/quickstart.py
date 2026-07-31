"""Minimal end-to-end inference example with the released CORAL weights."""

import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from coral import paths
from coral.inference import predict_interactions
from coral.model import CoralConfig, CoralPretraining


RNA_ENCODER = str(paths.DNABERT_DIR)
PROTEIN_ENCODER = paths.PROTEIN_ENCODER
WEIGHTS = paths.checkpoint(9)

pairs = pd.read_csv(ROOT / "examples" / "sample_pairs.csv")

rna_tokenizer = AutoTokenizer.from_pretrained(RNA_ENCODER, trust_remote_code=True)
protein_tokenizer = AutoTokenizer.from_pretrained(PROTEIN_ENCODER, trust_remote_code=True)

config = CoralConfig(rna_encoder_checkpoint=RNA_ENCODER, protein_encoder_checkpoint=PROTEIN_ENCODER)
model = PeftModel.from_pretrained(CoralPretraining(config=config), str(WEIGHTS)).merge_and_unload()

device = "cuda" if torch.cuda.is_available() else "cpu"
results = predict_interactions(model, pairs, rna_tokenizer, protein_tokenizer, device=device, batch_size=2)
print(results[["RNA_id", "Protein_id", "prediction", "interaction_probability"]])
