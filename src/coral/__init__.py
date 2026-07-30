"""CORAL — Cross-attention for RNA-protein Association Learning."""

from . import paths
from .model import Coral, CoralConfig, CoralPretraining
from .data import (
    CoralDataCollator,
    DataCollatorForContactPrediction,
    DataCollatorForRBDPrediction,
)
from .attention import AttentionAnalyzer
from .head_selection import select_heads_by_friedman_wilcoxon
from .rbd import intervals_to_residue_mask, load_rbd_intervals
from .stats import (
    bootstrap_over_units,
    cohens_d_per_unit,
    hierarchical_bootstrap,
)
from .inference import (
    predict_allVsall_interactions,
    predict_interactions,
    prepare_interactions_dataframe,
)

__all__ = [
    "paths",
    # model
    "Coral",
    "CoralConfig",
    "CoralPretraining",
    # collators
    "CoralDataCollator",
    "DataCollatorForContactPrediction",
    "DataCollatorForRBDPrediction",
    # attention interpretability
    "AttentionAnalyzer",
    "select_heads_by_friedman_wilcoxon",
    "load_rbd_intervals",
    "intervals_to_residue_mask",
    # statistics
    "hierarchical_bootstrap",
    "bootstrap_over_units",
    "cohens_d_per_unit",
    # inference
    "predict_interactions",
    "predict_allVsall_interactions",
    "prepare_interactions_dataframe",
]
