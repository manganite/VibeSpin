"""Public API for VibeSpin utility and analysis functions."""
from __future__ import annotations

from utils.exceptions import NumericalAnalysisError, VibeSpinError, ZeroVarianceAutocorrelationError
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    UNCERTAINTY_METHOD_BOOTSTRAP,
    blocking_error,
    summarize_derived_observable,
    summarize_primary_observable,
    summarize_seed_ensemble,
)
from utils.system import parse_args_compat

__all__ = [
    'VibeSpinError',
    'NumericalAnalysisError',
    'ZeroVarianceAutocorrelationError',
    'DEFAULT_CONFIDENCE_LEVEL',
    'UNCERTAINTY_METHOD_BLOCKING',
    'UNCERTAINTY_METHOD_BOOTSTRAP',
    'blocking_error',
    'summarize_primary_observable',
    'summarize_derived_observable',
    'summarize_seed_ensemble',
    'parse_args_compat',
]
