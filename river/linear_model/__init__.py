"""Linear models."""
from __future__ import annotations

from . import base
from .alma import ALMAClassifier
from .bayesian_lin_reg import BayesianLinearRegression
from .lin_reg import LinearRegression
from .log_reg import LogisticRegression
from .pa import PAClassifier, PARegressor
from .perceptron import Perceptron
from .softmax import SoftmaxRegression
from .multipa import MultiOutputPARegressor

__all__ = [
    "base",
    "ALMAClassifier",
    "BayesianLinearRegression",
    "LinearRegression",
    "LogisticRegression",
    "PAClassifier",
    "PARegressor",
    "Perceptron",
    "SoftmaxRegression",
    "MultiOutputPARegressor",
]
