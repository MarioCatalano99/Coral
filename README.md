# CORAL — Cross-Attention for RNA-protein Association Learning

![CORAL architecture overview](docs/assets/hero.png)

*Bidirectional cross-attention links DNABERT-2 and ESM-2 so that each molecule is encoded in the context of its partner; both encoders are adapted to the task by low-rank adaptation.*

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.04.22.720174-red)](https://doi.org/10.64898/2026.04.22.720174)

## Summary

CORAL (Cross-attention for RNA-protein Association Learning) predicts whether an RNA
and a protein interact from their sequences alone. It links two pretrained language
models, DNABERT-2 for the RNA and the 150M-parameter ESM-2 for the protein, through a
bidirectional cross-attention encoder. Each molecule is therefore represented in the
light of its partner. Both encoders are adapted to the task by low-rank adaptation
(LoRA).

Reported accuracies for this task depend on how the benchmark is built, so we rebuilt
the evaluation to close two shortcuts:

- **Sequence similarity** between training and test molecules. It is controlled
  separately for the RNA and the protein, up to a component-wise split in which neither
  molecule of a test pair has a relative in training.
- **Concentration of known interactions** on heavily studied molecules. It is halved
  by degree-corrected sampling of non-interacting pairs.

When neither molecule had a relative in training, four published predictors ranked
pairs close to chance (AUC 0.48–0.59), while CORAL retained an AUC of 0.72. CORAL was
never shown a structure, yet its protein-side cross-attention was higher inside
annotated RNA-binding domains than outside them in 94% of proteins. In TCGA lung
adenocarcinoma, CORAL's top-ranked candidate targets of the oncofetal protein IGF2BP1
were associated with pathological stage in the 86 tumours expressing it. The part of
that association specific to CORAL's candidates followed lymph-node involvement.

## Key results

**Benchmark (Figure 3).** AUC, mean over five folds per split, from the same training
and test pairs for every method. Clusters were built at 60% protein and 80% RNA sequence
identity. Every set is balanced, with degree-corrected negatives.

| Split | CORAL | ZHMolGraph | IPMiner | NPI-GNN | SeqMG |
| --- | --- | --- | --- | --- | --- |
| Redundant (random) | **0.945 ± 0.002** | 0.92 ± 0.02 | 0.88 | 0.84 | 0.69 |
| Pairwise | **0.939 ± 0.002** | 0.87 | 0.86 | 0.87 | 0.68 |
| Component-wise | **0.72 ± 0.03** | 0.59 ± 0.08 | 0.53 | 0.48 ± 0.02 | 0.57 |

- **Component-wise split:** CORAL's accuracy was 0.66 ± 0.04 and its F1 0.66 ± 0.06.
  The published methods reached accuracies of 0.50–0.55. ZHMolGraph's F1 of 0.62 came
  from calling most pairs interacting (recall 0.76, specificity 0.35).
- **Uniformly sampled negatives:** the five methods were nearly indistinguishable on
  the redundant and pairwise splits. On the component-wise split, CORAL's accuracy
  barely moved between negative sets (0.68 uniform, 0.66 degree-corrected), while
  ZHMolGraph's fell from 0.63 to 0.55.

**Attention at RNA-binding domains (Figure 4).** 496 proteins carry Pfam annotations
from 85 RNA-binding families. The final model's RNA→protein attention, from heads 5, 10
and 11 of the last cross-attention layer, was higher inside the annotated domains than
outside in 467 of them (94.2%).

- Mean Δ was 2.65 × 10⁻³ (95% CI 2.46–2.85 × 10⁻³), with t(495) = 26.2. The median
  per-protein Cohen's d was 0.88.
- The observed Δ exceeded all 1,000 rotations of the domain annotation along each
  protein.
- The effect was significant in 13 of the 14 domain families with at least ten
  proteins. PUA was the exception.

**Lung adenocarcinoma (Figure 5).** For IGF2BP1, the panel is the 334 transcripts
CORAL ranks in the top 1% of those expressed in TCGA LUAD.

- **Stage:** panel expression fell with pathological stage in the 86 IGF2BP1-positive
  tumours (Spearman ρ = −0.27, BH q = 0.031). Fewer than 3% of random panels matched
  for expression and length were as strongly associated (competitive P = 0.027;
  0.018 with GC content also matched).
- **IGF2BP1-negative tumours:** the gradient was absent.
- **N versus T:** the part specific to CORAL's candidates followed nodal involvement
  (N given T, competitive P = 0.018), not primary-tumour extent (T given N, P = 0.77).
- **Other proteins:** the IGF2BP3 and LIN28B panels were not associated with stage.

## Key features

- **Bidirectional cross-attention encoder.** Two stacked cross-modality layers with
  separate weights for each direction. RNA and protein condition on each other during
  encoding rather than only at the classifier.
- **LoRA adaptation of both encoders** (r = 16, α = 8). The pretrained representations
  can reorganise around the task at a fraction of the cost of full fine-tuning.
- **A benchmark that closes two shortcuts.**
  - Three splits of increasing stringency: redundant, pairwise and component-wise.
  - Negatives that remove half of each molecule's degree imbalance.
- **Interpretable attention.** Every head yields an explicit RNA k-mer × protein
  residue matrix. Protein-side attention concentrates on annotated RNA-binding domains
  without structural supervision.
- **Released final model and benchmark splits** on Zenodo.

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

Step 3 is a convenience, not a requirement: every script puts `src/` on
`sys.path` itself, so a fresh clone runs without installing anything.

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
│   ├── data.py              Tokenization + training, analysis, inference collators
│   ├── attention.py         AttentionAnalyzer for cross-attention extraction
│   ├── inference.py         Pairwise and streaming all-vs-all inference
│   ├── prediction_io.py     Parquet prediction parts, checkpoint hashing, score checks
│   ├── paths.py             Single source of truth for every filesystem path
│   ├── stats.py             Hierarchical bootstrap, per-unit Cohen's d
│   ├── head_selection.py    Friedman + Wilcoxon best-vs-rest with BH FDR
│   └── rbd.py               RBD interval loader and residue-mask helper
├── scripts/                 General entry points
│   ├── train.py             Fine-tune from scratch on a single fold
│   ├── predict.py           Score a CSV of RNA-protein pairs
│   ├── predict_all_vs_all.py  Stream manifest cross-products to Parquet parts
│   ├── download_weights.sh  Fetch the trained checkpoint from Zenodo
│   ├── download_data.sh     Fetch the benchmark splits from Zenodo
│   └── download_dnabert_weights.sh
├── analyses/
│   ├── RBD_analysis/        run.sh + scripts/ — attention at RNA-binding domains, paper Figure 4
│   └── luad_analysis/       scripts/{inputs,steps}/ — lung adenocarcinoma, paper Figure 5
├── data/                    All datasets (see "Data availability")
│   ├── datasets/            Benchmark splits, from Zenodo
│   ├── rbd/                 RBD census, whitelist and Pfam scan
│   ├── predictions/         CORAL inference outputs (regenerable)
│   ├── sequences/           Derived FASTA (regenerable)
│   ├── luad/                TCGA LUAD inputs and predictions
│   └── external/Pfam/       Third-party reference DBs
├── checkpoints/             coral-epoch-9/ — released weights, from Zenodo
├── DNABERT-2-117M/          RNA encoder source (weights fetched separately)
├── examples/                quickstart.py + sample_pairs.csv
├── docs/assets/             README figures
├── environment.yml          Conda environment, pinned
├── requirements.txt         Pip mirror of runtime dependencies
├── pyproject.toml           Packaging
├── CITATION.cff             Citation metadata
└── LICENSE                  Apache-2.0
```

A clone carries only small analysis inputs under `data/`, about 6 MB. The bulk is
gitignored: splits, Pfam, TCGA inputs, predictions and derived FASTA. It is fetched
from Zenodo and UCSC Xena, or regenerated by the commands below. `checkpoints/` and every
`results/` directory are gitignored for the same reason. Scripts create their output
directories if absent, so a fresh clone works.

To send results somewhere other than `analyses/<name>/results/`, set
`CORAL_RESULTS_DIR`. `src/coral/paths.py` also honours `CORAL_REPO_ROOT`, which
is only needed if the package is installed non-editably.

## Model weights

The released checkpoint is the final model of the paper, used for the attention and
lung adenocarcinoma analyses. It was trained on the degree-corrected pairs of redundant
fold 1596, with the training and test sets combined: 25,202 interacting and 25,202
non-interacting pairs.

- **Settings:** 10 epochs, last epoch kept, learning rate 5 × 10⁻⁵, bfloat16 mixed
  precision.
- **Size:** 52.8 M trainable parameters.

`bash scripts/download_weights.sh` puts it in `checkpoints/coral-epoch-9/`. Load it
via PEFT:

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

Two tensors in the checkpoint are not loaded, and PEFT skips them silently: the LoRA
factors of `protein_encoder.embeddings.position_embeddings`. ESM-2 uses rotary position
embeddings, so these were never trained, and current `transformers` no longer builds
the module. Every other tensor in the checkpoint is loaded.

**Weights versions.** The download script and the badge below resolve through the
concept DOI, which always points at the newest version.

| Version | Contents |
| --- | --- |
| v3 (current), [10.5281/zenodo.24465806](https://doi.org/10.5281/zenodo.24465806) | Final model of the paper, degree-corrected negatives (above) |
| v2, [10.5281/zenodo.21748220](https://doi.org/10.5281/zenodo.21748220) | Preprint model: redundant fold 1596 training set, uniformly sampled negatives |
| v1 | Does not match the published architecture (LoRA r = 8, no `reverse_projection`); do not use |

**Future work.** A Hugging Face Hub release that lets users skip the
local DNABERT2 clone and load CORAL with a single `from_pretrained` call is
on the roadmap.

## Reproducing paper results

### (a) Inference with the trained model

Works on CPU (slow) or any CUDA GPU. This is what most readers will want.

```bash
python scripts/predict.py \
  --input-csv examples/sample_pairs.csv \
  --output predictions.csv
```

For a large all-vs-all job, supply deduplicated sequence manifests and stream bounded
Parquet parts. Existing compatible parts are validated and resumed automatically:

```bash
torchrun --standalone --nproc_per_node=2 scripts/predict_all_vs_all.py \
  --rnas data/luad/luad_rna_sequences.parquet \
  --proteins data/luad/luad_proteins.tsv \
  --output-dir data/luad/luad_predictions
```

Interactive runs show one progress bar per rank, measured in completed RNAs. Progress
resumes from validated parts; routine part writes do not add log lines above the bars.

### (b) Benchmark (Figure 3)

Each value in Figure 3 averages five models, one per fold. Each model was trained on its
fold's `train.csv` and scored on its `val.csv`, the test set.

- **Epoch:** for each method, the epoch with the highest five-fold mean AUC.
- **Threshold for accuracy and F1:** set per fold to the mean of the median scores
  the model gave the test sets of the other four folds.

Training uses the full sequences (proteins up to 1,250 aa, RNAs up to 6,000 nt) with
bfloat16 autocast. In practice this needs a GPU with 24–40 GB of VRAM; we provide the
script but do not guarantee out-of-the-box reproduction on smaller accelerators.

```bash
bash scripts/download_data.sh   # one-off; 15 folds into data/datasets/

python scripts/train.py \
  --split pairwise_splits_60_80 \
  --fold 628 \
  --epochs 10 \
  --lr 5e-5
```

The paper used 5 × 10⁻⁵ for every trainable parameter on the redundant and pairwise
splits. On the component-wise split it used two rates:

- 1 × 10⁻⁶ for the LoRA corrections of the encoders;
- 1 × 10⁻⁵ for the cross-modality encoder, the protein projection and the heads.

`scripts/train.py` applies one rate (`--lr`) to every trainable parameter and does not
implement this two-rate setup yet.

### (c) Attention at RNA-binding domains (Figure 4)

Tests whether protein-side attention is higher inside Pfam-annotated RNA-binding
domains than outside them. Attention is averaged over the RNA axis and over all of a
protein's partners, and the unit of analysis is the protein.

The cohort comes from the 25,202 interacting pairs of redundant fold 1596 (train and
test):

1. The released model scores 24,708 of these pairs as interacting (p > 0.5), across
   4,141 proteins.
2. Of those proteins, 669 carry an annotated domain from the 85 whitelisted families.
3. 496 of them pass the length and domain-coverage filters.

`analyses/RBD_analysis/run.sh` runs the whole pipeline: scoring, Pfam annotation,
head screen, test, controls and figure. The Pfam hits of every protein the released
model calls interacting are committed in `data/rbd/`, so neither HMMER nor Pfam-A is
needed. It takes about 30 min on two GPUs; `analyses/RBD_analysis/README.md` lists the
steps, inputs and outputs.

```bash
bash analyses/RBD_analysis/run.sh
```

Expected results:

- **Heads:** the screen selects heads `[5, 10, 11]`.
- **Main test:** 467/496 proteins have Δ > 0, and mean Δ = 2.65e-03 (bootstrap 95% CI
  [2.46e-03, 2.85e-03]). One-sample t = 26.2, p = 7.6e-96.
- **Effect size:** per-protein Cohen's d has a median of 0.88 and a mean of 0.85.
- **Position-shuffle null:** empirical p = 0.001, with the observed value 33 null SDs
  above the null mean.
- **Domain families:** 13 of 14 are positive after BH correction; PUA is the exception.

### (d) Lung adenocarcinoma (Figure 5)

The workflow has two parts:

1. **Inputs.** `analyses/luad_analysis/scripts/inputs/`
   downloads and prepares the TCGA LUAD data: expression from UCSC Xena (Toil, GENCODE
   v23), the cohorts, the transcript sequences and the tumour purity estimates.
   `run_inference.sh` then scores the transcripts against IGF2BP1, IGF2BP3 and LIN28B
   with the checkpoint named in that directory's `config.json`.
2. **Analysis.** Steps s1–s6, s8 and the main figure run in about 5 min on CPU.

The directory's README gives the commands, the two files to place by hand and the
figure each script draws.

Every step checks the SHA-256 of its inputs and of the checkpoint the predictions were
made with.

Expected results:

| Quantity | IGF2BP1 | IGF2BP3 | LIN28B |
| --- | --- | --- | --- |
| Positive tumours | 86 | 308 | 24 |
| Eligible transcripts | 33,342 | 32,811 | 32,955 |
| Panel size (top 1%) | 334 | 329 | 330 |

- **Stage, IGF2BP1:** ρ = −0.27 (95% CI −0.46 to −0.07), q = 0.031. Competitive
  P = 0.027 (0.018 with GC content also matched).
- **N given T:** ρ = −0.19, competitive P = 0.018.
- **T given N:** ρ = −0.28, competitive P = 0.77.

## Data availability

All training and benchmarking splits are archived on Zenodo (current version v3,
[10.5281/zenodo.24466337](https://doi.org/10.5281/zenodo.24466337)) and fetched with
`scripts/download_data.sh` into `data/datasets/`.

- **Positive pairs** come from five sources:
  - RNAInter, keeping only interactions tagged as strong experimental evidence;
  - RPI488, RPI369, RPI2241 and RPI1807, keeping only their positives.

  Proteins longer than 1,250 aa and RNAs longer than 6,000 nt were excluded, leaving
  25,202 interacting pairs. RNAs are written in their DNA form (U as T), the form
  DNABERT-2 was pretrained on.
- **Sequence clusters.** RNAs and proteins were clustered separately with MMseqs2: 60%
  identity for proteins, 80% for RNAs, 60% coverage.
- **Splits.** Three strategies of increasing stringency, five folds each (15 folds):

  | Split | Assignment unit | Training pairs | Test pairs |
  | --- | --- | --- | --- |
  | Redundant (`redundant_split`) | Pairs, at random | 40,322 | 10,082 |
  | Pairwise (`pairwise_splits_60_80`) | Interaction clusters | 38,154–38,202 | 9,822–9,870 |
  | Component-wise (`componentwise_splits_60_80`) | RNA and protein clusters, independently | 22,110–24,100 | 3,744–4,264 |

  - **Pairwise:** no test pair shares both its RNA cluster and its protein cluster
    with a training pair.
  - **Component-wise:** pairs with one molecule on each side are discarded, so no test
    RNA or protein has a relative in training.
  - **Fold choice:** the five folds of each non-redundant split were chosen among
    100,000 random assignments. Each has at least 11,000 training positives, and the
    choice maximises the number of test positives.
- **Negative pairs** are combinations of the set's own molecules that are not reported
  to interact. They are drawn separately for each training and test set and balanced
  1:1 with positives.
  - **Degree correction:** each molecule receives c = (1 − λ)·n + λ·d negatives, with
    λ = 0.5. Here n is its count under uniform sampling and d its number of
    interacting partners, so every molecule keeps half of the imbalance that uniform
    sampling creates.
  - **Excluded:** reported interactions, duplicate negatives, and training negatives
    that also occur as test negatives of the same fold.
- **Layout.** `data/datasets/<split>/fold_<seed>/{train,val}.csv`, with columns
  `RNA_id, Protein_id, labels, RNA_seqs, Prot_seqs`; `val.csv` is the test set.
- **Uniformly sampled negatives.** Supplementary Figure 2 uses splits whose negatives
  were sampled uniformly over molecules. They stay in v1 of the data record,
  [10.5281/zenodo.19730810](https://doi.org/10.5281/zenodo.19730810), together with the
  30/50 identity splits of the preprint. v1 and v3 hold the same interacting pairs,
  splits and folds; only the negatives differ. Do not use v2: its archive is incomplete.
- **RBD inputs** (`data/rbd/`) are the Gerstberger RBP/RBD census, the 85-family RBD
  whitelist derived from it, and the Pfam scan of every protein in the analysis.

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
  doi     = {10.64898/2026.04.22.720174},
  url     = {https://www.biorxiv.org/content/10.64898/2026.04.22.720174v2}
}
```

## License

- **Code:** Apache-2.0. See [`LICENSE`](LICENSE).
- **Paper and figures:** CC-BY 4.0.

## Contact

PierFederico Gherardini — <pier.federico.gherardini@uniroma2.it> (lead contact).
