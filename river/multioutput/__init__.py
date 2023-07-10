"""Multi-output models."""
from __future__ import annotations

from .chain import (
    ClassifierChain,
    MonteCarloClassifierChain,
    ProbabilisticClassifierChain,
    RegressorChain,
    RegressorFullChain,
)
from .parallel import RegressorParallel, RegressorFullParallel
from .encoder import MultiClassEncoder

__all__ = [
    "ClassifierChain",
    "MonteCarloClassifierChain",
    "MultiClassEncoder",
    "ProbabilisticClassifierChain",
    "RegressorChain",
    "RegressorFullChain",
    "RegressorFullParallel",
    "RegressorParallel",
]
