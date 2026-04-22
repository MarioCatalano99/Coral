"""CORAL — Cross-attention for RNA-protein Association Learning."""

from .model import Coral, CoralConfig, CoralPretraining
from .data import CoralDataCollator, DataCollatorForContactPrediction
from .attention import AttentionAnalyzer
from .inference import (
    predict_allVsall_interactions,
    predict_interactions,
    prepare_interactions_dataframe,
)

__all__ = [
    "Coral",
    "CoralConfig",
    "CoralPretraining",
    "CoralDataCollator",
    "DataCollatorForContactPrediction",
    "AttentionAnalyzer",
    "predict_interactions",
    "predict_allVsall_interactions",
    "prepare_interactions_dataframe",
]
