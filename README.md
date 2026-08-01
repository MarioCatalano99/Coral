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
- **Interpretable attention heads** — cross-attention heads systematically prioritise structural contact positions and RNA-binding domains without explicit structural supervision.
- **Released trained checkpoint** for direct inference; training from scratch is provided as a reference implementation.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MarioCatalano99/Coral.git
cd Coral

# 2. Create the conda environment
conda env create -f environment.yml
conda activate coral

# 3. Install the package (editable, so `import coral` works everywhere)
pip install -e .

# 4. Fetch the DNABERT-2 weights (~447 MB)
#    The modified tokenizer / model source code ships inside DNABERT-2-117M/
#    in this repo — the script only downloads pytorch_model.bin from the
#    upstream HuggingFace release.
bash scripts/download_dnabert_weights.sh

# 5. Download the trained CORAL weights from Zenodo
bash scripts/download_weights.sh
```

Pip-only installation is supported via `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Step 3 is a convenience, not a requirement: every script and notebook puts
`src/` on `sys.path` itself, so a fresh clone runs without installing anything.

## Quick start

With the environment active and the weights downloaded, the bundled
example runs end-to-end in under twenty lines:

```bash
python examples/quickstart.py
```

This loads three RNA-protein pairs from `examples/sample_pairs.csv`, merges the
LoRA adapters into the encoders, and prints per-pair interaction probabilities.

## Repository structure

```
Coral/
├── src/coral/               Installable Python package
│   ├── model.py             CoralConfig / Coral / CoralPretraining
│   ├── data.py              Tokenization + padding, contact and RBD collators
│   ├── attention.py         AttentionAnalyzer for cross-attention extraction
│   ├── inference.py         predict_interactions / predict_allVsall_interactions
│   ├── paths.py             Single source of truth for every filesystem path
│   ├── stats.py             Hierarchical bootstrap, per-unit Cohen's d
│   ├── head_selection.py    Friedman + Wilcoxon best-vs-rest with BH FDR
│   └── rbd.py               RBD interval loader and residue-mask helper
├── scripts/                 General entry points
│   ├── train.py             Fine-tune from scratch on a single fold
│   ├── predict.py           Score a CSV of RNA-protein pairs
│   ├── download_weights.sh  Fetch the trained checkpoint from Zenodo
│   ├── download_data.sh     Fetch the benchmark splits from Zenodo
│   └── download_dnabert_weights.sh
├── analyses/                One directory per interpretability analysis
│   ├── contact_analysis/    {scripts,notebooks,results}/  — paper Figure 5
│   └── RBD_analysis/        {scripts,notebooks,results}/  — paper Figure 6
├── data/                    All datasets (see "Data availability")
│   ├── datasets/            Benchmark splits, from Zenodo
│   ├── pdb/                 PDB pairs, contact annotations, structures
│   ├── rbd/                 RBD census and Pfam scan inputs
│   ├── predictions/         CORAL inference outputs (regenerable)
│   ├── sequences/           Derived FASTA (regenerable)
│   └── external/Pfam/       Third-party reference DBs
├── checkpoints/             coral-epoch-{0..9}/ — LoRA + modules_to_save
├── DNABERT-2-117M/          RNA encoder source (weights fetched separately)
├── examples/                quickstart.py + sample_pairs.csv
├── docs/assets/             README figures
├── environment.yml          Conda environment, pinned
├── requirements.txt         Pip mirror of runtime dependencies
├── pyproject.toml           Packaging
├── CITATION.cff             Citation metadata
└── LICENSE                  Apache-2.0
```

A clone carries only the small analysis inputs under `data/` — the PDB contact
annotations and the RBD census, about 3 MB. The bulk (splits, Pfam, PDB
structures, predictions, derived FASTA) is gitignored: it is fetched from Zenodo
and RCSB or regenerated by the commands below. `checkpoints/` and every
`results/` directory are gitignored for the same reason. Scripts create their
output directories if absent, so a fresh clone works.

To send results somewhere other than `analyses/<name>/results/`, set
`CORAL_RESULTS_DIR`. `src/coral/paths.py` also honours `CORAL_REPO_ROOT`, which
is only needed if the package is installed non-editably.

## Model weights

The final checkpoint (epoch 9 of the redundant-split training described in the
paper) is on Zenodo. `bash scripts/download_weights.sh` puts it in
`checkpoints/coral-epoch-9/`. Load it via PEFT:

```python
from peft import PeftModel
from coral import paths
from coral.model import CoralConfig, CoralPretraining

config = CoralConfig(
    rna_encoder_checkpoint=str(paths.DNABERT_DIR),
    protein_encoder_checkpoint=paths.PROTEIN_ENCODER,
)
model = PeftModel.from_pretrained(CoralPretraining(config=config),
                                  str(paths.checkpoint(9))).merge_and_unload()
```

> **Weights version.** Versions of this record before v2 contained a checkpoint
> that does not match the published architecture — it was trained at LoRA rank 8
> and without the `reverse_projection` module. Use v2 or later. The download
> script and the badge below both resolve through the concept DOI, which always
> points at the newest version.

**Future work.** A Hugging Face Hub release that lets users skip the
local DNABERT2 clone and load CORAL with a single `from_pretrained` call is
on the roadmap.

## Reproducing paper results

We distinguish three reproduction tiers. Only the first two are expected to
work on commodity hardware; the third is a reference implementation.

### (a) Inference with the trained model

Works on CPU (slow) or any CUDA GPU. This is what most readers will want.

```bash
python scripts/predict.py \
  --input-csv examples/sample_pairs.csv \
  --output predictions.csv
```

### (b) Reproduce Table 2 from the trained model + released splits

Reasonable on a single GPU with ≥ 8 GB VRAM; each fold scores roughly
10,000 RNA-protein pairs.

```bash
bash scripts/download_data.sh  # one-off; pulls the 25 folds from Zenodo

python scripts/predict.py \
  --input-csv data/datasets/componentwise_splits_60_80/fold_2161/val.csv \
  --output results/componentwise_60_80_fold_2161.csv
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

## Regenerating the interpretability analyses

Both analyses run from the repo root and read `checkpoints/coral-epoch-9/`.
Each screens its own attention heads: do not assume a head subset carries from
one analysis to the other. Results land in `analyses/<name>/results/`, with
`heads_all/` averaging all 12 heads and `heads_best/` averaging the screened
subset.

### Contact analysis (Figure 5)

Tests whether cross-attention is elevated at residue-level contacts in PDB
complexes. The score is the geometric mean of the RNA→protein and protein→RNA
directions, taken from the last cross-attention layer. Head screening selects
`[0, 7]`.

```bash
python analyses/contact_analysis/scripts/screen_attention_heads.py
python analyses/contact_analysis/scripts/attention_analysis.py            # heads_all
python analyses/contact_analysis/scripts/attention_analysis.py \
    --selected-heads analyses/contact_analysis/results/selected_heads.json  # heads_best
python analyses/contact_analysis/scripts/plot_figure5.py
```

Panel b (3D structures) additionally needs `pymol-open-source`:

```bash
python analyses/contact_analysis/scripts/build_panelb_sessions.py
# then pick camera angles in analyses/contact_analysis/notebooks/panel_b_camera.ipynb
```

Expected (`heads_best`): 309 complexes, mean Δ = 5.12e-03, one-sample
t = 9.79, p = 3.5e-20; 244/309 complexes have Δ > 0; bootstrap 95% CI
[4.15e-03, 6.19e-03]; per-complex Cohen's d mean 0.35, median 0.31.
The whole run takes under a minute on a single GPU.

### RBD analysis (Figure 6)

Tests whether protein-side attention, marginalized over the RNA axis, is
elevated at annotated RNA-binding domains. The unit of analysis is the protein.
Head screening selects head `[2]` — notably *not* the pair contact analysis
selects.

Steps 1–3 build the RBD annotations and need `hmmer` plus a local Pfam-A copy
in `data/external/Pfam/`:

```bash
jupyter execute analyses/RBD_analysis/notebooks/01_build_rbd_whitelist.ipynb
jupyter execute analyses/RBD_analysis/notebooks/02_build_protein_fasta.ipynb
bash analyses/RBD_analysis/scripts/run_hmmscan.sh
python analyses/RBD_analysis/scripts/parse_rbd_domtbl.py \
    --domtbl data/rbd/predicted_interacting_proteins.domtbl \
    --fasta data/sequences/predicted_interacting_proteins.fa \
    --rbd-json data/rbd/semiconservative_rbd2pfam.json \
    --out analyses/RBD_analysis/results/rbd_annotations
```

Then the test, controls and figures:

```bash
python analyses/RBD_analysis/scripts/screen_attention_heads.py \
    --predictions data/predictions/val_1596_predicted.csv
python analyses/RBD_analysis/scripts/attention_analysis.py            # heads_all
python analyses/RBD_analysis/scripts/attention_analysis.py \
    --selected-heads analyses/RBD_analysis/results/selected_heads.json  # heads_best
python analyses/RBD_analysis/scripts/controls.py --heads best
python analyses/RBD_analysis/scripts/plot_figure6.py
```

Expected (`heads_best`): 462 proteins, mean Δ = 2.95e-03, one-sample
t = 27.08, p = 1.2e-97; 435/462 proteins have Δ > 0; bootstrap 95% CI
[2.74e-03, 3.17e-03]; per-protein Cohen's d mean 0.88, median 0.94. The effect
survives the position-shuffle null (p < 0.001) and holds across domain families.

## Data availability

All training and benchmarking splits are archived on Zenodo and fetched with
`scripts/download_data.sh` into `data/datasets/`.

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
  residues at ≤3.4 Å inter-atomic distance. Needed for the contact analysis.
- **RBD inputs** (`data/rbd/`) are the Gerstberger RBP/RBD census plus the
  Pfam scan outputs used to annotate RNA-binding domains.

Licensing of the redistributed data follows the licenses of the source
databases; consult each source for details.

[![Weights DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19729892.svg)](https://doi.org/10.5281/zenodo.19729892)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19730809.svg)](https://doi.org/10.5281/zenodo.19730809)

Both badges point at **concept DOIs**, which always resolve to the latest
version of their record.

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
