"""Single source of truth for filesystem paths.

Everything is anchored at `REPO_ROOT`, so scripts and notebooks work regardless
of the current working directory. Nothing here touches the filesystem — these
are pure path computations; callers `mkdir` what they intend to write.

Two environment overrides:

- ``CORAL_REPO_ROOT`` — needed when the package is installed non-editably, where
  walking up from ``site-packages/coral/paths.py`` would not find the data.
- ``CORAL_RESULTS_DIR`` — redirect regenerable analysis outputs somewhere else
  (scratch, a shared volume). Defaults to ``analyses/<name>/results/``.
"""

import os
from pathlib import Path

# --- Anchors --------------------------------------------------------------- #
# src/coral/paths.py -> src/coral -> src -> repo root
REPO_ROOT = Path(os.environ.get("CORAL_REPO_ROOT",
                                Path(__file__).resolve().parents[2]))

# --- Encoder weights ------------------------------------------------------- #
DNABERT_DIR = REPO_ROOT / "DNABERT-2-117M"
PROTEIN_ENCODER = "facebook/esm2_t30_150M_UR50D"

# --- Shared inputs --------------------------------------------------------- #
DATA_DIR = REPO_ROOT / "data"
RBD_DIR = DATA_DIR / "rbd"                  # RBD-analysis inputs
DATASETS_DIR = DATA_DIR / "datasets"        # train/val splits (Zenodo)
PREDICTIONS_DIR = DATA_DIR / "predictions"  # CORAL inference outputs
SEQUENCES_DIR = DATA_DIR / "sequences"      # derived FASTA
EXTERNAL_DIR = DATA_DIR / "external"        # large third-party reference DBs
PFAM_DIR = EXTERNAL_DIR / "Pfam"

# --- Trained model checkpoints --------------------------------------------- #
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# --- Per-analysis code ----------------------------------------------------- #
ANALYSES_DIR = REPO_ROOT / "analyses"


def checkpoint(epoch: int) -> Path:
    """Checkpoint directory for a training epoch.

    Example: checkpoint(9) -> <repo>/checkpoints/coral-epoch-9
    """
    return CHECKPOINTS_DIR / f"coral-epoch-{epoch}"


def analysis_dir(analysis: str) -> Path:
    """Root directory for a named analysis (e.g. 'RBD_analysis')."""
    return ANALYSES_DIR / analysis


def analysis_results(analysis: str) -> Path:
    """Regenerable output directory for an analysis.

    Gitignored build artifacts, not source. Honours ``CORAL_RESULTS_DIR``.
    """
    override = os.environ.get("CORAL_RESULTS_DIR")
    if override:
        return Path(override) / analysis
    return ANALYSES_DIR / analysis / "results"
