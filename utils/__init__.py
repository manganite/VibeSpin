"""Public API for VibeSpin utility and analysis functions."""
from __future__ import annotations

from utils.cli_helpers import parse_args_compat
from utils.exceptions import NumericalAnalysisError, VibeSpinError, ZeroVarianceAutocorrelationError
from utils.physics_helpers import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    UNCERTAINTY_METHOD_BOOTSTRAP,
    blocking_error,
    summarize_derived_observable,
    summarize_primary_observable,
    summarize_seed_ensemble,
)

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
