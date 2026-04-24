# CORAL — Cross-Attention for RNA-protein Association Learning

![CORAL architecture overview](docs/assets/hero.png)

*Bidirectional cross-attention over ESM-2 and DNABERT2 representations, fine-tuned with Low-Rank Adaptation, predicts RNA-protein interactions while learning attention patterns that align with structural contact interfaces.*

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-TODO-red)](https://www.biorxiv.org/)

## Summary

CORAL (Cross-attention for RNA-protein Association Learning) is a deep-learning framework for predicting RNA-protein interactions from sequence alone. Unlike methods that encode each molecule in isolation and combine the resulting vectors late, CORAL couples a pretrained protein language model (ESM-2) and a pretrained nucleotide language model (DNABERT2) through a bidirectional cross-attention encoder, then fine-tunes both backbones jointly with Low-Rank Adaptation. Alongside the model we release a benchmarking framework that partitions RNA-protein pairs under three levels of train/test redundancy — conventional random splits, pairwise non-redundant splits, and the more stringent component-wise non-redundant splits — exposing the gap between benchmark accuracy and true generalization that has been masked in prior work. Under the most stringent component-wise evaluation CORAL holds an F1 of 0.65 where the next-best method drops to 0.55, and specific cross-attention heads attend systematically to structurally resolved contact residues (27% elevated attention at interface positions across 309 experimentally resolved complexes, p < 0.01).

## Key features

- **Bidirectional cross-attention encoder** that lets RNA and protein representations condition on each other during encoding rather than only at classification time.
- **LoRA fine-tuning** of ESM-2 and DNABERT2 so pretrained sequence knowledge is preserved while task-specific features are learned.
- **Three-tier non-redundant evaluation framework** (random, pairwise, component-wise) that surfaces real-world generalization behavior.
- **Interpretable attention heads** — heads 0 and 7 systematically prioritise structural contact positions without explicit structural supervision.
- **Released trained checkpoint** for direct inference; training from scratch is provided as a reference implementation.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MarioCatalano99/Coral.git
cd Coral

# 2. Create the conda environment
conda env create -f environment.yml
conda activate coral

# 3. Fetch the DNABERT-2 weights (~447 MB)
#    The modified tokenizer / model source code ships inside DNABERT-2-117M/
#    in this repo — the script only downloads pytorch_model.bin from the
#    upstream HuggingFace release.
bash scripts/download_dnabert_weights.sh

# 4. Download the trained CORAL weights from Zenodo
bash scripts/download_weights.sh
```

Pip-only installation is supported via `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

With the conda environment active and the weights downloaded, the bundled
example runs end-to-end in under twenty lines:

```bash
python examples/quickstart.py
```

This loads three RNA-protein pairs from `examples/sample_pairs.csv`, merges the
LoRA adapters into the encoders, and prints per-pair interaction probabilities.

## Reproducing paper results

We distinguish three reproduction tiers. Only the first two are expected to
work on commodity hardware; the third is a reference implementation.

### (a) Inference with the trained model

Works on CPU (slow) or any CUDA GPU. This is what most readers will want.

```bash
python scripts/predict.py \
  --input-csv examples/sample_pairs.csv \
  --weights weights/coral-epoch-9 \
  --output predictions.csv
```

### (b) Reproduce Table 2 from the trained model + released splits

Reasonable on a single GPU with ≥ 8 GB VRAM; each fold scores roughly
10,000 RNA-protein pairs.

```bash
bash scripts/download_data.sh  # one-off; pulls the 25 folds from Zenodo

python scripts/predict.py \
  --input-csv Data/datasets/componentwise_splits_60_80/fold_2161/val.csv \
  --weights weights/coral-epoch-9 \
  --output results/componentwise_60_80_fold_2161.csv
```

To recompute the attention-interpretability tables (paper Fig. 3 and the
population-level statistics):

```bash
python scripts/compute_attention_stats.py   # writes attention_heads_stats.csv
python scripts/select_attention_heads.py    # Friedman + Wilcoxon-BH on the CSV
python scripts/analyze_attention.py         # t-test + bootstrap + Cohen's d
```

### (c) Retrain from scratch (reference implementation)

Hardware-dependent. Training uses the full non-truncated sequences
(proteins up to 1,250 aa, RNAs up to 6,000 nt) and bfloat16 autocast; in
practice this requires a GPU with 24-40 GB VRAM. We provide the script but
do not guarantee out-of-the-box reproduction on smaller accelerators.

```bash
python scripts/train.py \
  --split componentwise_splits_60_80 \
  --fold 2161 \
  --epochs 10
```

## Repository structure

```
Coral/
├── src/coral/               Installable Python package
│   ├── model.py             CoralConfig and the cross-modality transformer
│   ├── data.py              Tokenization, MLM collator, contact-label collator
│   ├── attention.py         AttentionAnalyzer for cross-attention extraction
│   └── inference.py         predict_interactions / predict_allVsall_interactions
├── scripts/                 CLI drivers — one per paper experiment
│   ├── train.py             Fine-tune from scratch on a single fold
│   ├── predict.py           Score a CSV of RNA-protein pairs
│   ├── compute_attention_stats.py   Per-head contact/non-contact summary
│   ├── select_attention_heads.py    Friedman + BH-FDR post-hoc
│   ├── analyze_attention.py         Population-level statistics for selected heads
│   ├── download_weights.sh          Fetch weights from Zenodo
│   └── download_data.sh             Fetch benchmark splits from Zenodo
├── examples/
│   ├── sample_pairs.csv     Three-row input for the quick start
│   └── quickstart.py        Minimal load-and-predict demo
├── weights/coral-epoch-9/   Zenodo-hosted LoRA + modules_to_save checkpoint
├── data/pdb/                PDB-derived pairs and contact annotations
├── docs/assets/             README figures (hero image)
├── environment.yml          Conda environment, pinned
├── requirements.txt         Pip mirror of runtime dependencies
├── CITATION.cff             Citation metadata
└── LICENSE                  Apache-2.0
```

## Data availability

All training and benchmarking splits are archived on Zenodo and fetched with
`scripts/download_data.sh`.

- **Positive pairs** were compiled from RNAInter, RPI488, RPI369, RPI2241, and
  RPI1807. We retained RNAInter interactions tagged as strong experimental
  evidence; from the RPI datasets we kept only positives. Proteins longer
  than 1,250 aa and RNAs longer than 6,000 nt were excluded, yielding 24,012
  high-confidence positive interactions.
- **Splits** come in five folds per strategy: redundant, pairwise at
  80/60% and 50/30% RNA/protein identity, component-wise at the same two
  thresholds (25 folds total). Negative pairs are random RNA-protein
  re-pairings within the same split, balanced 1:1 with positives.
- **PDB contacts** (`data/pdb/`) define structurally resolved interface
  residues at ≤3.4 Å inter-atomic distance. These are shipped in-repo and
  are needed for the attention-interpretability analyses.

Licensing of the redistributed data follows the licenses of the source
databases; consult each source for details.

[![Weights DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19729893.svg)](https://doi.org/10.5281/zenodo.19729893)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19730810.svg)](https://doi.org/10.5281/zenodo.19730810)

## Model weights

The final checkpoint (epoch 9 of the redundant-split training described in the
paper) is stored on Zenodo alongside the benchmark splits. Load it via PEFT:

```python
from peft import PeftModel
from coral.model import CoralConfig, CoralPretraining

config = CoralConfig(
    rna_encoder_checkpoint="./DNABERT-2-117M",
    protein_encoder_checkpoint="facebook/esm2_t30_150M_UR50D",
)
model = PeftModel.from_pretrained(CoralPretraining(config=config),
                                  "weights/coral-epoch-9").merge_and_unload()
```

**Future work.** A Hugging Face Hub release that lets users skip the
local DNABERT2 clone and load CORAL with a single `from_pretrained` call is
on the roadmap.

## Citation

If you use CORAL in your research, please cite the preprint:

```bibtex
@article{catalano2026coral,
  title   = {Cross-Attention Over RNA And Protein Sequences Enables Generalizable Interaction Prediction},
  author  = {Catalano, Mario and Pepe, Gerardo and Ausiello, Gabriele and McWhite, Claire and Gambosi, Giorgio and Helmer Citterich, Manuela and Gherardini, PierFederico},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {TODO:10.1101/2026.04.22.000000},
  url     = {TODO:https://www.biorxiv.org/content/10.1101/2026.04.22.000000v1}
}
```

## License

- **Code:** Apache-2.0. See [`LICENSE`](LICENSE).
- **Paper and figures:** CC-BY 4.0.

## Contact

PierFederico Gherardini — <pier.federico.gherardini@uniroma2.it> (lead contact).
