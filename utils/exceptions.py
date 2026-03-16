"""Project-specific exception types for analysis and runtime failures."""
from __future__ import annotations


class VibeSpinError(Exception):
    """Base class for project-specific exceptions."""


class NumericalAnalysisError(VibeSpinError, RuntimeError):
    """Base class for mathematically undefined or failed analysis results."""


class ZeroVarianceAutocorrelationError(NumericalAnalysisError):
    """Raised when autocorrelation is undefined because the series variance is zero."""
