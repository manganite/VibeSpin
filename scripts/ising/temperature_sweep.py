"""
Standardized temperature sweep for the 2D Ising model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from models.ising_model import IsingSimulation
from utils.cli_helpers import parse_args_compat
from utils.physics_helpers import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    UNCERTAINTY_METHOD_BOOTSTRAP,
    summarize_asymmetric_replicate_uncertainty,
    summarize_derived_observable,
    summarize_entropy_observable,
    summarize_primary_observable,
    summarize_seed_ensemble,
)
from utils.system_helpers import (
    convergence_equilibrate,
    convergence_equilibrate_with_status,
    parallel_sweep,
    plot_temperature_sweep,
    setup_logging,
)

_WORKER_CONFIDENCE_LEVEL = DEFAULT_CONFIDENCE_LEVEL
_WORKER_DERIVED_METHOD = UNCERTAINTY_METHOD_BLOCKING
_WORKER_DERIVED_BOOTSTRAP_RESAMPLES = 0
_TC_ISING_THEORY = 2.26918531421


def simulate_temperature(
    params: _SweepPoint,
) -> tuple[float, float, float, float, float]:
    """
    Worker function to simulate a single temperature point for the Ising model.
    """
    T = params.temperature
    L = params.size
    meas_steps = params.meas_steps
    eq_probe_steps = params.eq_probe_steps
    eq_max_steps = params.eq_max_steps

    # Initialize two simulations for the two-start convergence test
    sim_r = IsingSimulation(size=L, temp=T, init_state='random')
    sim_o = IsingSimulation(size=L, temp=T, init_state='ordered')

    # Robust equilibration via two-start convergence
    convergence_equilibrate(
        sim_r,
        sim_o,
        chunk_size=eq_probe_steps,
        max_steps=eq_max_steps,
        qs_sigma_threshold=params.eq_qs_sigma_threshold,
        qs_min_steps=params.eq_qs_min_steps,
    )

    # Use the converged random-start instance for measurement.
    mags, engs = sim_r.run(n_steps=meas_steps)
    mags_arr = np.asarray(mags, dtype=np.float64)
    engs_arr = np.asarray(engs, dtype=np.float64)

    mag = summarize_primary_observable(time_series=mags_arr)
    eng = summarize_primary_observable(time_series=engs_arr)
    chi = summarize_derived_observable(
        magnetization_series=mags_arr,
        temperature=T,
        L=L,
        observable='chi',
    )
    cv = summarize_derived_observable(
        energy_series=engs_arr,
        temperature=T,
        L=L,
        observable='cv',
    )
    return (
        float(mag['value']),
        float(eng['value']),
        float(chi['value']),
        float(cv['value']),
        float(mag['tau_int']),
    )


class _SweepPoint(NamedTuple):
    """Typed worker payload for one temperature point in the sweep."""

    temperature: float
    size: int
    meas_steps: int
    eq_probe_steps: int
    eq_max_steps: int
    eq_qs_sigma_threshold: float
    eq_qs_min_steps: int


class _SeedSweepPoint(NamedTuple):
    """Typed worker payload for one temperature/seed point in the sweep."""

    temperature: float
    size: int
    meas_steps: int
    eq_probe_steps: int
    eq_max_steps: int
    eq_qs_sigma_threshold: float
    eq_qs_min_steps: int
    temperature_index: int
    seed_index: int
    seed: int


class _SeedReplicaAttempt(NamedTuple):
    """Typed worker payload for one full temperature sweep attempt for a seed replica."""

    temperatures: tuple[float, ...]
    size: int
    meas_steps: int
    eq_probe_steps: int
    eq_max_steps: int
    eq_qs_sigma_threshold: float
    eq_qs_min_steps: int
    seed_index: int


def _simulate_seed_temperature(params: _SeedSweepPoint) -> dict[str, float]:
    """Run one seeded sweep point and return summary statistics for all observables."""
    T = params.temperature
    L = params.size
    meas_steps = params.meas_steps

    sim_r = IsingSimulation(size=L, temp=T, init_state='random', seed=params.seed)
    sim_o = IsingSimulation(size=L, temp=T, init_state='ordered', seed=params.seed)
    equilibration_steps, converged = convergence_equilibrate_with_status(
        sim_r,
        sim_o,
        chunk_size=params.eq_probe_steps,
        max_steps=params.eq_max_steps,
        qs_sigma_threshold=params.eq_qs_sigma_threshold,
        qs_min_steps=params.eq_qs_min_steps,
        qs_allow_stuck=(T < _TC_ISING_THEORY),
    )

    if not converged:
        return {
            'temperature': float(T),
            'equilibrated': 0.0,
            'equilibration_steps': float(equilibration_steps),
        }

    # Select the cleanly ordered simulation for measurements at low T
    active_sim = sim_r
    if T < _TC_ISING_THEORY:
        m_r = float(np.abs(sim_r._get_magnetization()))
        m_o = float(np.abs(sim_o._get_magnetization()))
        if m_o > m_r + 0.2:
            active_sim = sim_o

    mags, engs = active_sim.run(n_steps=meas_steps)
    mags_arr = np.asarray(mags, dtype=np.float64)
    engs_arr = np.asarray(engs, dtype=np.float64)

    mag = summarize_primary_observable(
        time_series=mags_arr,
        confidence=_WORKER_CONFIDENCE_LEVEL,
    )
    eng = summarize_primary_observable(
        time_series=engs_arr,
        confidence=_WORKER_CONFIDENCE_LEVEL,
    )
    chi = summarize_derived_observable(
        magnetization_series=mags_arr,
        temperature=T,
        L=L,
        observable='chi',
        method=_WORKER_DERIVED_METHOD,
        confidence=_WORKER_CONFIDENCE_LEVEL,
        bootstrap_resamples=_WORKER_DERIVED_BOOTSTRAP_RESAMPLES,
    )
    cv = summarize_derived_observable(
        energy_series=engs_arr,
        temperature=T,
        L=L,
        observable='cv',
        method=_WORKER_DERIVED_METHOD,
        confidence=_WORKER_CONFIDENCE_LEVEL,
        bootstrap_resamples=_WORKER_DERIVED_BOOTSTRAP_RESAMPLES,
    )

    return {
        'temperature_index': float(params.temperature_index),
        'seed_index': float(params.seed_index),
        'equilibrated_flag': float(converged),
        'equilibration_steps': float(equilibration_steps),
        'avg_m_value': float(mag['value']),
        'avg_m_err': float(mag['err']),
        'avg_m_tau_int': float(mag['tau_int']),
        'avg_m_n_eff': float(mag['n_eff']),
        'avg_e_value': float(eng['value']),
        'avg_e_err': float(eng['err']),
        'avg_e_tau_int': float(eng['tau_int']),
        'avg_e_n_eff': float(eng['n_eff']),
        'susc_value': float(chi['value']),
        'susc_err': float(chi['err']),
        'susc_tau_int': float(chi['tau_int']),
        'susc_n_eff': float(chi['n_eff']),
        'spec_h_value': float(cv['value']),
        'spec_h_err': float(cv['err']),
        'spec_h_tau_int': float(cv['tau_int']),
        'spec_h_n_eff': float(cv['n_eff']),
    }


def _simulate_seed_replica_attempt(
    params: _SeedReplicaAttempt,
) -> dict[str, np.ndarray | float]:
    """Run one full seed replica sweep and stop immediately on the first failed temperature."""
    t_points = len(params.temperatures)
    avg_m_values = np.full(t_points, np.nan, dtype=np.float64)
    avg_m_errors = np.full(t_points, np.nan, dtype=np.float64)
    avg_m_tau = np.full(t_points, np.nan, dtype=np.float64)
    avg_m_n_eff = np.full(t_points, np.nan, dtype=np.float64)
    avg_e_values = np.full(t_points, np.nan, dtype=np.float64)
    avg_e_errors = np.full(t_points, np.nan, dtype=np.float64)
    avg_e_tau = np.full(t_points, np.nan, dtype=np.float64)
    avg_e_n_eff = np.full(t_points, np.nan, dtype=np.float64)
    susc_values = np.full(t_points, np.nan, dtype=np.float64)
    susc_errors = np.full(t_points, np.nan, dtype=np.float64)
    susc_tau = np.full(t_points, np.nan, dtype=np.float64)
    susc_n_eff = np.full(t_points, np.nan, dtype=np.float64)
    spec_h_values = np.full(t_points, np.nan, dtype=np.float64)
    spec_h_errors = np.full(t_points, np.nan, dtype=np.float64)
    spec_h_tau = np.full(t_points, np.nan, dtype=np.float64)
    spec_h_n_eff = np.full(t_points, np.nan, dtype=np.float64)
    equilibrated = np.zeros(t_points, dtype=np.uint8)
    equilibration_steps = np.full(t_points, np.nan, dtype=np.float64)

    for i, T in enumerate(params.temperatures):
        seed = i * 100_000 + params.seed_index * 1_000
        sim_r = IsingSimulation(size=params.size, temp=T, init_state='random', seed=seed)
        sim_o = IsingSimulation(size=params.size, temp=T, init_state='ordered', seed=seed)
        total_steps, converged = convergence_equilibrate_with_status(
            sim_r,
            sim_o,
            chunk_size=params.eq_probe_steps,
            max_steps=params.eq_max_steps,
            qs_sigma_threshold=params.eq_qs_sigma_threshold,
            qs_min_steps=params.eq_qs_min_steps,
            qs_allow_stuck=(T < _TC_ISING_THEORY),
        )

        equilibration_steps[i] = float(total_steps)
        equilibrated[i] = np.uint8(converged)
        if not converged:
            break

        # Use the ordered-start simulation if the random one is stuck
        active_sim = sim_r
        if T < _TC_ISING_THEORY:
            m_r = float(np.abs(sim_r._get_magnetization()))
            m_o = float(np.abs(sim_o._get_magnetization()))
            if m_o > m_r + 0.2:  # Ordered is significantly more ordered
                active_sim = sim_o

        mags, engs = active_sim.run(n_steps=params.meas_steps)
        mags_arr = np.asarray(mags, dtype=np.float64)
        engs_arr = np.asarray(engs, dtype=np.float64)

        mag = summarize_primary_observable(
            time_series=mags_arr,
            confidence=_WORKER_CONFIDENCE_LEVEL,
        )
        eng = summarize_primary_observable(
            time_series=engs_arr,
            confidence=_WORKER_CONFIDENCE_LEVEL,
        )
        chi = summarize_derived_observable(
            magnetization_series=mags_arr,
            temperature=T,
            L=params.size,
            observable='chi',
            method=_WORKER_DERIVED_METHOD,
            confidence=_WORKER_CONFIDENCE_LEVEL,
            bootstrap_resamples=_WORKER_DERIVED_BOOTSTRAP_RESAMPLES,
        )
        cv = summarize_derived_observable(
            energy_series=engs_arr,
            temperature=T,
            L=params.size,
            observable='cv',
            method=_WORKER_DERIVED_METHOD,
            confidence=_WORKER_CONFIDENCE_LEVEL,
            bootstrap_resamples=_WORKER_DERIVED_BOOTSTRAP_RESAMPLES,
        )

        avg_m_values[i] = float(mag['value'])
        avg_m_errors[i] = float(mag['err'])
        avg_m_tau[i] = float(mag['tau_int'])
        avg_m_n_eff[i] = float(mag['n_eff'])
        avg_e_values[i] = float(eng['value'])
        avg_e_errors[i] = float(eng['err'])
        avg_e_tau[i] = float(eng['tau_int'])
        avg_e_n_eff[i] = float(eng['n_eff'])
        susc_values[i] = float(chi['value'])
        susc_errors[i] = float(chi['err'])
        susc_tau[i] = float(chi['tau_int'])
        susc_n_eff[i] = float(chi['n_eff'])
        spec_h_values[i] = float(cv['value'])
        spec_h_errors[i] = float(cv['err'])
        spec_h_tau[i] = float(cv['tau_int'])
        spec_h_n_eff[i] = float(cv['n_eff'])

    return {
        'seed_index': float(params.seed_index),
        'equilibrated_flag': equilibrated,
        'equilibration_steps': equilibration_steps,
        'avg_m_value': avg_m_values,
        'avg_m_err': avg_m_errors,
        'avg_m_tau_int': avg_m_tau,
        'avg_m_n_eff': avg_m_n_eff,
        'avg_e_value': avg_e_values,
        'avg_e_err': avg_e_errors,
        'avg_e_tau_int': avg_e_tau,
        'avg_e_n_eff': avg_e_n_eff,
        'susc_value': susc_values,
        'susc_err': susc_errors,
        'susc_tau_int': susc_tau,
        'susc_n_eff': susc_n_eff,
        'spec_h_value': spec_h_values,
        'spec_h_err': spec_h_errors,
        'spec_h_tau_int': spec_h_tau,
        'spec_h_n_eff': spec_h_n_eff,
    }


def _build_uncertainty_bundle(
    *,
    values_by_seed: np.ndarray,
    errors_by_seed: np.ndarray,
    tau_by_seed: np.ndarray,
    n_eff_by_seed: np.ndarray,
    confidence: float,
) -> dict[str, np.ndarray | float]:
    """Build a standardized uncertainty bundle for one observable across seeds."""
    t_points = values_by_seed.shape[0]
    value = np.empty(t_points, dtype=np.float64)
    err = np.empty(t_points, dtype=np.float64)
    ci_low = np.empty(t_points, dtype=np.float64)
    ci_high = np.empty(t_points, dtype=np.float64)
    tau_int = np.empty(t_points, dtype=np.float64)
    tau_int_err = np.empty(t_points, dtype=np.float64)
    tau_int_ci_low = np.empty(t_points, dtype=np.float64)
    tau_int_ci_high = np.empty(t_points, dtype=np.float64)
    n_eff = np.empty(t_points, dtype=np.float64)

    for i in range(t_points):
        agg = summarize_seed_ensemble(
            values=values_by_seed[i],
            within_seed_errors=errors_by_seed[i],
            confidence=confidence,
        )
        value[i] = float(agg['value'])
        err[i] = float(agg['err'])
        ci_low[i] = float(agg['ci_low'])
        ci_high[i] = float(agg['ci_high'])
        tau_row = np.asarray(tau_by_seed[i], dtype=np.float64)
        tau_agg = summarize_asymmetric_replicate_uncertainty(
            samples=tau_row,
            confidence=confidence,
        )
        tau_int[i] = float(tau_agg['value'])
        tau_int_err[i] = float(tau_agg['err'])
        tau_int_ci_low[i] = float(tau_agg['ci_low'])
        tau_int_ci_high[i] = float(tau_agg['ci_high'])
        n_eff[i] = float(np.nansum(n_eff_by_seed[i]))

    return {
        'value': value,
        'err': err,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'tau_int': tau_int,
        'tau_int_err': tau_int_err,
        'tau_int_ci_low': tau_int_ci_low,
        'tau_int_ci_high': tau_int_ci_high,
        'n_eff': n_eff,
        'samples': values_by_seed.astype(np.float64),
    }


def _build_quality_flags(
    *,
    tau_int: np.ndarray,
    tau_int_ci_low: np.ndarray,
    tau_int_ci_high: np.ndarray,
    n_eff: np.ndarray,
    min_effective_samples: float,
    max_tau_relative_width: float,
) -> dict[str, np.ndarray]:
    """Build per-temperature diagnostics for uncertainty quality."""
    undefined_autocorr = ~np.isfinite(tau_int)
    low_effective_sample = np.isfinite(n_eff) & (n_eff < min_effective_samples)

    tau_span = tau_int_ci_high - tau_int_ci_low
    denom = np.maximum(np.abs(tau_int), 1e-12)
    rel_width = tau_span / denom
    tau_interval_unstable = np.isfinite(rel_width) & (rel_width > max_tau_relative_width)

    return {
        'undefined_autocorr_flag': undefined_autocorr.astype(np.uint8),
        'low_effective_sample_flag': low_effective_sample.astype(np.uint8),
        'tau_interval_unstable_flag': tau_interval_unstable.astype(np.uint8),
    }


def main() -> None:
    """
    Execute the temperature sweep and generate standardized 4-panel plots.
    """
    parser = argparse.ArgumentParser(description='2D Ising Model Temperature Sweep')
    parser.add_argument('--size', type=int, default=64, help='Linear lattice size L')
    parser.add_argument(
        '--eq-probe-steps', type=int, default=500,
        help='Chunk size for convergence check during equilibration',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=20000,
        help='Hard cap on total equilibration steps',
    )
    parser.add_argument(
        '--eq-qs-sigma-threshold', type=float, default=0.05,
        help='Tail-std threshold for quasi-steady stuck detection (0 disables)',
    )
    parser.add_argument(
        '--eq-qs-min-steps', type=int, default=1500,
        help='Minimum accumulated equilibration steps before stuck detection can fire',
    )
    parser.add_argument('--meas-steps', type=int, default=5000, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=4.0, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=40, help='Number of temperature points')
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=1,
        help='Target number of fully converged seed replicas to retain',
    )
    parser.add_argument(
        '--max-seed-attempts',
        type=int,
        default=None,
        help='Hard cap on total seed attempts, including replacements; defaults to 10*n-seeds',
    )
    parser.add_argument(
        '--derived-uncertainty-method',
        type=str,
        default=UNCERTAINTY_METHOD_BLOCKING,
        choices=[UNCERTAINTY_METHOD_BLOCKING, UNCERTAINTY_METHOD_BOOTSTRAP],
        help='Uncertainty method for susceptibility and specific heat',
    )
    parser.add_argument(
        '--derived-bootstrap-resamples',
        type=int,
        default=0,
        help='Bootstrap resamples used when derived-uncertainty-method=bootstrap',
    )
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help='Two-sided confidence level for uncertainty intervals (0 < c < 1)',
    )
    parser.add_argument(
        '--entropy-uncertainty-method',
        type=str,
        default=UNCERTAINTY_METHOD_BOOTSTRAP,
        choices=[UNCERTAINTY_METHOD_BLOCKING, UNCERTAINTY_METHOD_BOOTSTRAP],
        help='Uncertainty method for entropy curves',
    )
    parser.add_argument(
        '--entropy-bootstrap-resamples',
        type=int,
        default=400,
        help='Bootstrap resamples used when entropy-uncertainty-method=bootstrap',
    )
    parser.add_argument(
        '--strict-uncertainty',
        action='store_true',
        help='Fail if undefined autocorrelation points exceed the allowed fraction',
    )
    parser.add_argument(
        '--max-undefined-fraction',
        type=float,
        default=0.25,
        help='Maximum allowed fraction of undefined tau_int points in strict mode',
    )
    parser.add_argument(
        '--min-effective-samples',
        type=float,
        default=20.0,
        help='Threshold used to flag low effective sample size points',
    )
    parser.add_argument(
        '--max-tau-relative-width',
        type=float,
        default=1.0,
        help='Relative width threshold to flag unstable tau intervals',
    )
    parser.add_argument(
        '--transition-preset',
        type=str,
        default='auto',
        choices=['auto', 'none', 'theory'],
        help='Transition overlay preset for plotting',
    )
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

    if not (0.0 < float(args.confidence_level) < 1.0):
        raise ValueError(
            f'confidence-level must satisfy 0 < c < 1, got {args.confidence_level}'
        )
    if args.max_undefined_fraction < 0.0 or args.max_undefined_fraction > 1.0:
        raise ValueError(
            'max-undefined-fraction must satisfy 0 <= f <= 1, '
            f'got {args.max_undefined_fraction}'
        )
    if args.min_effective_samples < 0.0:
        raise ValueError(
            f'min-effective-samples must be >= 0, got {args.min_effective_samples}'
        )
    if args.max_tau_relative_width < 0.0:
        raise ValueError(
            f'max-tau-relative-width must be >= 0, got {args.max_tau_relative_width}'
        )
    if (
        args.derived_uncertainty_method == UNCERTAINTY_METHOD_BOOTSTRAP
        and args.derived_bootstrap_resamples <= 0
    ):
        raise ValueError(
            'derived-bootstrap-resamples must be > 0 when '
            'derived-uncertainty-method=bootstrap'
        )

    global _WORKER_CONFIDENCE_LEVEL
    global _WORKER_DERIVED_METHOD
    global _WORKER_DERIVED_BOOTSTRAP_RESAMPLES
    _WORKER_CONFIDENCE_LEVEL = float(args.confidence_level)
    _WORKER_DERIVED_METHOD = str(args.derived_uncertainty_method)
    _WORKER_DERIVED_BOOTSTRAP_RESAMPLES = int(args.derived_bootstrap_resamples)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)

    target_n_seeds = int(args.n_seeds)
    if target_n_seeds < 1:
        raise ValueError(f'n-seeds must be >= 1, got {target_n_seeds}')

    max_seed_attempts = args.max_seed_attempts
    if max_seed_attempts is None:
        max_seed_attempts = max(target_n_seeds, 10 * target_n_seeds)
    max_seed_attempts = int(max_seed_attempts)
    if max_seed_attempts < target_n_seeds:
        raise ValueError(
            'max-seed-attempts must be >= n-seeds, '
            f'got max-seed-attempts={max_seed_attempts}, n-seeds={target_n_seeds}'
        )

    logger.info(
        'Starting Ising temperature sweep '
        f'(L={L}, target_n_seeds={target_n_seeds}, max_seed_attempts={max_seed_attempts})...'
    )

    temperature_tuple = tuple(float(T) for T in temperatures)
    attempt_records: list[dict[str, np.ndarray | float]] = []
    retained_records: list[dict[str, np.ndarray | float]] = []

    while len(retained_records) < target_n_seeds:
        attempted_n_seeds = len(attempt_records)
        remaining_attempts = max_seed_attempts - attempted_n_seeds
        if remaining_attempts <= 0:
            raise RuntimeError(
                'Could not collect the requested number of fully converged seed replicas. '
                f'Retained {len(retained_records)} of {target_n_seeds} after '
                f'{max_seed_attempts} total attempts. Increase --max-seed-attempts or relax '
                'equilibration settings.'
            )

        batch_size = min(target_n_seeds - len(retained_records), remaining_attempts)
        batch_params = [
            _SeedReplicaAttempt(
                temperatures=temperature_tuple,
                size=L,
                meas_steps=args.meas_steps,
                eq_probe_steps=args.eq_probe_steps,
                eq_max_steps=args.eq_max_steps,
                eq_qs_sigma_threshold=args.eq_qs_sigma_threshold,
                eq_qs_min_steps=args.eq_qs_min_steps,
                seed_index=attempted_n_seeds + s,
            )
            for s in range(batch_size)
        ]
        batch_raw: list[dict[str, np.ndarray | float]] = parallel_sweep(
            worker_func=_simulate_seed_replica_attempt,
            params=batch_params,
        )

        for item in sorted(batch_raw, key=lambda record: int(record['seed_index'])):
            attempt_records.append(item)
            seed_flags = np.asarray(item['equilibrated_flag'], dtype=np.uint8)
            if bool(np.all(seed_flags > 0)):
                retained_records.append(item)

    attempted_n_seeds = len(attempt_records)
    retained_n_seeds = len(retained_records)
    shape = (args.t_points, attempted_n_seeds)
    avg_m_values = np.full(shape, np.nan, dtype=np.float64)
    avg_m_errors = np.full(shape, np.nan, dtype=np.float64)
    avg_m_tau = np.full(shape, np.nan, dtype=np.float64)
    avg_m_n_eff = np.full(shape, np.nan, dtype=np.float64)
    avg_e_values = np.full(shape, np.nan, dtype=np.float64)
    avg_e_errors = np.full(shape, np.nan, dtype=np.float64)
    avg_e_tau = np.full(shape, np.nan, dtype=np.float64)
    avg_e_n_eff = np.full(shape, np.nan, dtype=np.float64)
    susc_values = np.full(shape, np.nan, dtype=np.float64)
    susc_errors = np.full(shape, np.nan, dtype=np.float64)
    susc_tau = np.full(shape, np.nan, dtype=np.float64)
    susc_n_eff = np.full(shape, np.nan, dtype=np.float64)
    spec_h_values = np.full(shape, np.nan, dtype=np.float64)
    spec_h_errors = np.full(shape, np.nan, dtype=np.float64)
    spec_h_tau = np.full(shape, np.nan, dtype=np.float64)
    spec_h_n_eff = np.full(shape, np.nan, dtype=np.float64)
    equilibrated = np.zeros(shape, dtype=np.uint8)
    equilibration_steps = np.full(shape, np.nan, dtype=np.float64)

    for item in attempt_records:
        s = int(item['seed_index'])
        equilibrated[:, s] = np.asarray(item['equilibrated_flag'], dtype=np.uint8)
        equilibration_steps[:, s] = np.asarray(item['equilibration_steps'], dtype=np.float64)
        avg_m_values[:, s] = np.asarray(item['avg_m_value'], dtype=np.float64)
        avg_m_errors[:, s] = np.asarray(item['avg_m_err'], dtype=np.float64)
        avg_m_tau[:, s] = np.asarray(item['avg_m_tau_int'], dtype=np.float64)
        avg_m_n_eff[:, s] = np.asarray(item['avg_m_n_eff'], dtype=np.float64)
        avg_e_values[:, s] = np.asarray(item['avg_e_value'], dtype=np.float64)
        avg_e_errors[:, s] = np.asarray(item['avg_e_err'], dtype=np.float64)
        avg_e_tau[:, s] = np.asarray(item['avg_e_tau_int'], dtype=np.float64)
        avg_e_n_eff[:, s] = np.asarray(item['avg_e_n_eff'], dtype=np.float64)
        susc_values[:, s] = np.asarray(item['susc_value'], dtype=np.float64)
        susc_errors[:, s] = np.asarray(item['susc_err'], dtype=np.float64)
        susc_tau[:, s] = np.asarray(item['susc_tau_int'], dtype=np.float64)
        susc_n_eff[:, s] = np.asarray(item['susc_n_eff'], dtype=np.float64)
        spec_h_values[:, s] = np.asarray(item['spec_h_value'], dtype=np.float64)
        spec_h_errors[:, s] = np.asarray(item['spec_h_err'], dtype=np.float64)
        spec_h_tau[:, s] = np.asarray(item['spec_h_tau_int'], dtype=np.float64)
        spec_h_n_eff[:, s] = np.asarray(item['spec_h_n_eff'], dtype=np.float64)

    retained_seed_mask = np.all(equilibrated > 0, axis=0)
    retained_seed_indices = np.flatnonzero(retained_seed_mask).astype(np.int64)
    dropped_seed_indices = np.flatnonzero(~retained_seed_mask).astype(np.int64)
    dropped_n_seeds = int(dropped_seed_indices.size)

    avg_m_values = avg_m_values[:, retained_seed_mask]
    avg_m_errors = avg_m_errors[:, retained_seed_mask]
    avg_m_tau = avg_m_tau[:, retained_seed_mask]
    avg_m_n_eff = avg_m_n_eff[:, retained_seed_mask]
    avg_e_values = avg_e_values[:, retained_seed_mask]
    avg_e_errors = avg_e_errors[:, retained_seed_mask]
    avg_e_tau = avg_e_tau[:, retained_seed_mask]
    avg_e_n_eff = avg_e_n_eff[:, retained_seed_mask]
    susc_values = susc_values[:, retained_seed_mask]
    susc_errors = susc_errors[:, retained_seed_mask]
    susc_tau = susc_tau[:, retained_seed_mask]
    susc_n_eff = susc_n_eff[:, retained_seed_mask]
    spec_h_values = spec_h_values[:, retained_seed_mask]
    spec_h_errors = spec_h_errors[:, retained_seed_mask]
    spec_h_tau = spec_h_tau[:, retained_seed_mask]
    spec_h_n_eff = spec_h_n_eff[:, retained_seed_mask]

    mag_bundle = _build_uncertainty_bundle(
        values_by_seed=avg_m_values,
        errors_by_seed=avg_m_errors,
        tau_by_seed=avg_m_tau,
        n_eff_by_seed=avg_m_n_eff,
        confidence=float(args.confidence_level),
    )
    eng_bundle = _build_uncertainty_bundle(
        values_by_seed=avg_e_values,
        errors_by_seed=avg_e_errors,
        tau_by_seed=avg_e_tau,
        n_eff_by_seed=avg_e_n_eff,
        confidence=float(args.confidence_level),
    )
    susc_bundle = _build_uncertainty_bundle(
        values_by_seed=susc_values,
        errors_by_seed=susc_errors,
        tau_by_seed=susc_tau,
        n_eff_by_seed=susc_n_eff,
        confidence=float(args.confidence_level),
    )
    cv_bundle = _build_uncertainty_bundle(
        values_by_seed=spec_h_values,
        errors_by_seed=spec_h_errors,
        tau_by_seed=spec_h_tau,
        n_eff_by_seed=spec_h_n_eff,
        confidence=float(args.confidence_level),
    )

    avg_m_arr = np.asarray(mag_bundle['value'], dtype=np.float64)
    avg_e_arr = np.asarray(eng_bundle['value'], dtype=np.float64)
    susc_arr = np.asarray(susc_bundle['value'], dtype=np.float64)
    spec_h_arr = np.asarray(cv_bundle['value'], dtype=np.float64)
    tau_int_arr = np.asarray(mag_bundle['tau_int'], dtype=np.float64)
    tau_int_ci_low_arr = np.asarray(mag_bundle['tau_int_ci_low'], dtype=np.float64)
    tau_int_ci_high_arr = np.asarray(mag_bundle['tau_int_ci_high'], dtype=np.float64)

    entropy_bundle = summarize_entropy_observable(
        temperatures=temperatures,
        specific_heat_samples=np.asarray(cv_bundle['samples'], dtype=np.float64),
        specific_heat_err=np.asarray(cv_bundle['err'], dtype=np.float64),
        confidence=float(args.confidence_level),
        method=str(args.entropy_uncertainty_method),
        bootstrap_resamples=int(args.entropy_bootstrap_resamples),
        rng_seed=0,
    )
    entropy = np.asarray(entropy_bundle['value'], dtype=np.float64)

    flags = _build_quality_flags(
        tau_int=tau_int_arr,
        tau_int_ci_low=tau_int_ci_low_arr,
        tau_int_ci_high=tau_int_ci_high_arr,
        n_eff=np.asarray(mag_bundle['n_eff'], dtype=np.float64),
        min_effective_samples=float(args.min_effective_samples),
        max_tau_relative_width=float(args.max_tau_relative_width),
    )

    unstable_count = int(np.sum(flags['tau_interval_unstable_flag']))
    undefined_count = int(np.sum(flags['undefined_autocorr_flag']))
    low_effective_count = int(np.sum(flags['low_effective_sample_flag']))
    bad_mask = (
        np.asarray(flags['undefined_autocorr_flag'], dtype=np.float64) > 0.0
    ) | (
        np.asarray(flags['tau_interval_unstable_flag'], dtype=np.float64) > 0.0
    ) | (
        np.asarray(flags['low_effective_sample_flag'], dtype=np.float64) > 0.0
    )
    total_points = int(temperatures.size)
    well_conditioned_count = int(total_points - np.sum(bad_mask))
    diagnostics_note = (
        f'n_seeds={retained_n_seeds}/{target_n_seeds}, attempted={attempted_n_seeds}, '
        f'undefined tau={undefined_count}/{temperatures.size}, '
        f'unstable tau intervals={unstable_count}/{temperatures.size}'
    )
    run_metadata_note = (
        f'L={L}, n_seeds={retained_n_seeds}/{target_n_seeds}, attempted={attempted_n_seeds}, '
        f'conf={float(args.confidence_level):.2f}, '
        f'method={UNCERTAINTY_METHOD_BLOCKING}, entropy={args.entropy_uncertainty_method}, '
        f'strict={args.strict_uncertainty}'
    )
    quality_summary: dict[str, int | float] = {
        'total_points': total_points,
        'well_conditioned_count': well_conditioned_count,
        'low_effective_count': low_effective_count,
        'unstable_interval_count': unstable_count,
        'undefined_count': undefined_count,
        'attempted_n_seeds': attempted_n_seeds,
        'retained_n_seeds': retained_n_seeds,
        'dropped_n_seeds': dropped_n_seeds,
    }

    undefined_fraction = float(np.mean(flags['undefined_autocorr_flag']))
    if args.strict_uncertainty and undefined_fraction > float(args.max_undefined_fraction):
        raise RuntimeError(
            'strict uncertainty check failed: undefined tau_int fraction '
            f'{undefined_fraction:.3f} exceeds {args.max_undefined_fraction:.3f}'
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / 'temperature_sweep_data.npz',
        temperatures=temperatures,
        avg_m=avg_m_arr,
        avg_e=avg_e_arr,
        susc=susc_arr,
        spec_h=spec_h_arr,
        entropy=entropy,
        tau_int=tau_int_arr,
        tau_int_err=np.asarray(mag_bundle['tau_int_err'], dtype=np.float64),
        tau_int_ci_low=tau_int_ci_low_arr,
        tau_int_ci_high=tau_int_ci_high_arr,
        avg_m_value=mag_bundle['value'],
        avg_m_err=mag_bundle['err'],
        avg_m_ci_low=mag_bundle['ci_low'],
        avg_m_ci_high=mag_bundle['ci_high'],
        avg_m_tau_int=mag_bundle['tau_int'],
        avg_m_n_eff=mag_bundle['n_eff'],
        avg_m_samples=mag_bundle['samples'],
        avg_e_value=eng_bundle['value'],
        avg_e_err=eng_bundle['err'],
        avg_e_ci_low=eng_bundle['ci_low'],
        avg_e_ci_high=eng_bundle['ci_high'],
        avg_e_tau_int=eng_bundle['tau_int'],
        avg_e_n_eff=eng_bundle['n_eff'],
        avg_e_samples=eng_bundle['samples'],
        susc_value=susc_bundle['value'],
        susc_err=susc_bundle['err'],
        susc_ci_low=susc_bundle['ci_low'],
        susc_ci_high=susc_bundle['ci_high'],
        susc_tau_int=susc_bundle['tau_int'],
        susc_n_eff=susc_bundle['n_eff'],
        susc_samples=susc_bundle['samples'],
        spec_h_value=cv_bundle['value'],
        spec_h_err=cv_bundle['err'],
        spec_h_ci_low=cv_bundle['ci_low'],
        spec_h_ci_high=cv_bundle['ci_high'],
        spec_h_tau_int=cv_bundle['tau_int'],
        spec_h_n_eff=cv_bundle['n_eff'],
        spec_h_samples=cv_bundle['samples'],
        entropy_value=entropy_bundle['value'],
        entropy_err=entropy_bundle['err'],
        entropy_ci_low=entropy_bundle['ci_low'],
        entropy_ci_high=entropy_bundle['ci_high'],
        entropy_samples=entropy_bundle['samples'],
        entropy_uncertainty_method=str(args.entropy_uncertainty_method),
        undefined_autocorr_flag=flags['undefined_autocorr_flag'],
        low_effective_sample_flag=flags['low_effective_sample_flag'],
        tau_interval_unstable_flag=flags['tau_interval_unstable_flag'],
        uncertainty_method=UNCERTAINTY_METHOD_BLOCKING,
        confidence_level=float(args.confidence_level),
        n_seeds=retained_n_seeds,
        requested_n_seeds=target_n_seeds,
        attempted_n_seeds=attempted_n_seeds,
        max_seed_attempts=max_seed_attempts,
        retained_n_seeds=retained_n_seeds,
        dropped_n_seeds=dropped_n_seeds,
        retained_seed_indices=retained_seed_indices,
        dropped_seed_indices=dropped_seed_indices,
        seed_converged_mask=retained_seed_mask.astype(np.uint8),
        convergence_by_temperature_seed=equilibrated,
        equilibration_steps=equilibration_steps,
        bootstrap_resamples=int(args.entropy_bootstrap_resamples),
        nan_or_undefined_count=float(np.isnan(tau_int_arr).sum()),
    )

    def _peak_temperature(values: np.ndarray) -> float:
        valid = np.isfinite(temperatures) & np.isfinite(values)
        if not np.any(valid):
            return float('nan')
        vals = values[valid]
        temps = temperatures[valid]
        return float(temps[int(np.argmax(vals))])

    t_chi_peak = _peak_temperature(susc_arr)
    t_cv_peak = _peak_temperature(spec_h_arr)
    t_tau_peak = _peak_temperature(tau_int_arr)

    auto_marker: dict[str, float] = {}
    if np.isfinite(t_chi_peak):
        auto_marker[r'$T_{\chi}$'] = t_chi_peak
    elif np.isfinite(t_cv_peak):
        auto_marker[r'$T_{C_v}$'] = t_cv_peak
    elif np.isfinite(t_tau_peak):
        auto_marker[r'$T_{\tau}$'] = t_tau_peak
    else:
        auto_marker[r'$T_c$'] = _TC_ISING_THEORY

    if args.transition_preset == 'none':
        transition_markers: dict[str, float] = {}
    elif args.transition_preset == 'theory':
        transition_markers = {r'$T_c$': _TC_ISING_THEORY}
    else:
        transition_markers = auto_marker

    finite_markers = np.asarray(list(transition_markers.values()), dtype=np.float64)
    finite_markers = finite_markers[np.isfinite(finite_markers)]
    transition_window: tuple[float, float] | None = None
    if finite_markers.size > 0:
        diffs = np.diff(np.asarray(temperatures, dtype=np.float64))
        pad = float(np.median(diffs)) if diffs.size > 0 else 0.1
        lo = float(np.min(finite_markers) - pad)
        hi = float(np.max(finite_markers) + pad)
        transition_window = (lo, hi)

    plot_temperature_sweep(
        temperatures=temperatures,
        avg_m=avg_m_arr.tolist(),
        avg_e=avg_e_arr.tolist(),
        susc=susc_arr.tolist(),
        spec_h=spec_h_arr.tolist(),
        avg_m_err=np.asarray(mag_bundle['err'], dtype=np.float64),
        avg_e_err=np.asarray(eng_bundle['err'], dtype=np.float64),
        susc_err=np.asarray(susc_bundle['err'], dtype=np.float64),
        spec_h_err=np.asarray(cv_bundle['err'], dtype=np.float64),
        entropy=entropy,
        entropy_err=np.asarray(entropy_bundle['err'], dtype=np.float64),
        entropy_ci_low=np.asarray(entropy_bundle['ci_low'], dtype=np.float64),
        entropy_ci_high=np.asarray(entropy_bundle['ci_high'], dtype=np.float64),
        tau_int=tau_int_arr,
        tau_int_ci_low=tau_int_ci_low_arr,
        tau_int_ci_high=tau_int_ci_high_arr,
        tau_unstable_flag=flags['tau_interval_unstable_flag'],
        low_effective_sample_flag=flags['low_effective_sample_flag'],
        diagnostics_note=diagnostics_note,
        run_metadata_note=run_metadata_note,
        quality_summary=quality_summary,
        transition_temperatures=transition_markers,
        transition_window=transition_window,
        entropy_reference=None,
        min_visible_rel_error=0.01,
        mark_invalid_uncertainty=True,
        title=f'2D Ising Model: Temperature Sweep (L={L})',
        filename='temperature_sweep.png',
        directory=args.output_dir,
    )


if __name__ == '__main__':
    main()
