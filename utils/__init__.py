"""Public API for VibeSpin utility and analysis functions.

The export list mirrors the utilities that scripts and notebooks actually
consume, so it can serve as an accurate overview of the public surface.
Submodule imports (``utils.statistics``, ``utils.observables``, ...) remain
the canonical access path.
"""
from __future__ import annotations

from utils.equilibration import (
    adaptive_equilibrate,
    convergence_equilibrate,
    convergence_equilibrate_with_status,
    estimate_relaxation_time_two_start,
)
from utils.exceptions import NumericalAnalysisError, VibeSpinError, ZeroVarianceAutocorrelationError
from utils.observables import (
    calculate_entropy,
    calculate_thermodynamics,
    correlation_length_1e,
    derived_thermo_estimate,
    get_averaged_correlation,
    pair_correlation_x,
    radial_average_sk,
    simulate_equilibrium_correlation,
)
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    UNCERTAINTY_METHOD_BOOTSTRAP,
    UNCERTAINTY_METHOD_REPLICATE,
    blocking_error,
    calculate_autocorr,
    estimate_effective_sample_size,
    estimate_tau_int_or_nan,
    power_fit,
    summarize_asymmetric_replicate_uncertainty,
    summarize_derived_observable,
    summarize_entropy_observable,
    summarize_primary_observable,
    summarize_replicate_samples,
    summarize_seed_ensemble,
)
from utils.sweep_helpers import derive_point_seed, validate_sweep_uncertainty_args
from utils.system import parallel_sweep, parse_args_compat, setup_logging

__all__ = [
    'VibeSpinError',
    'NumericalAnalysisError',
    'ZeroVarianceAutocorrelationError',
    'DEFAULT_CONFIDENCE_LEVEL',
    'UNCERTAINTY_METHOD_BLOCKING',
    'UNCERTAINTY_METHOD_BOOTSTRAP',
    'UNCERTAINTY_METHOD_REPLICATE',
    'adaptive_equilibrate',
    'blocking_error',
    'calculate_autocorr',
    'calculate_entropy',
    'calculate_thermodynamics',
    'convergence_equilibrate',
    'convergence_equilibrate_with_status',
    'correlation_length_1e',
    'derive_point_seed',
    'derived_thermo_estimate',
    'estimate_effective_sample_size',
    'estimate_relaxation_time_two_start',
    'estimate_tau_int_or_nan',
    'get_averaged_correlation',
    'pair_correlation_x',
    'parallel_sweep',
    'parse_args_compat',
    'power_fit',
    'radial_average_sk',
    'setup_logging',
    'simulate_equilibrium_correlation',
    'summarize_asymmetric_replicate_uncertainty',
    'summarize_derived_observable',
    'summarize_entropy_observable',
    'summarize_primary_observable',
    'summarize_replicate_samples',
    'summarize_seed_ensemble',
    'validate_sweep_uncertainty_args',
]
