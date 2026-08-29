"""
Standardized temperature sweep for the 2D Ising model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, cast

import numpy as np

from models.ising_model import IsingSimulation
from utils.plotting import plot_temperature_sweep
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    UNCERTAINTY_METHOD_BOOTSTRAP,
    summarize_entropy_observable,
)
from utils.sweep_helpers import (
    ThermoPoint,
    build_quality_flags,
    build_uncertainty_bundle,
    simulate_thermo_point,
    validate_sweep_uncertainty_args,
)
from utils.system import parallel_sweep, parse_args_compat, setup_logging

_TC_ISING_THEORY = 2.26918531421


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

    validate_sweep_uncertainty_args(
        confidence_level=float(args.confidence_level),
        max_undefined_fraction=float(args.max_undefined_fraction),
        min_effective_samples=float(args.min_effective_samples),
        max_tau_relative_width=float(args.max_tau_relative_width),
        derived_uncertainty_method=str(args.derived_uncertainty_method),
        derived_bootstrap_resamples=int(args.derived_bootstrap_resamples),
        n_seeds=int(args.n_seeds),
    )

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)
    t_points = len(temperatures)

    target_n_seeds = int(args.n_seeds)
    max_seed_attempts = args.max_seed_attempts
    if max_seed_attempts is None:
        max_seed_attempts = max(target_n_seeds, 10 * target_n_seeds)
    max_seed_attempts = int(max_seed_attempts)

    logger.info(
        'Starting Ising temperature sweep '
        f'(L={L}, target_n_seeds={target_n_seeds}, max_seed_attempts={max_seed_attempts})...'
    )

    # Grid to store results for each (temperature, seed_index)
    # We maintain a list of results for each temperature until target_n_seeds is reached.
    results_grid: list[list[dict[str, float]]] = [[] for _ in range(t_points)]
    attempts_per_temp = np.zeros(t_points, dtype=int)

    # Track overall attempts to respect max_seed_attempts (per temperature)
    while any(len(results_grid[t]) < target_n_seeds for t in range(t_points)):
        # Identify points that still need calculation
        points_to_calculate = []
        for t in range(t_points):
            needed = target_n_seeds - len(results_grid[t])
            if needed > 0:
                # Check if we've exhausted attempts for this temperature
                if attempts_per_temp[t] >= max_seed_attempts:
                    continue

                # Add up to 'needed' attempts
                for _ in range(needed):
                    seed_idx = attempts_per_temp[t]
                    seed = t * 100_000 + seed_idx * 1_000
                    T_val = float(temperatures[t])
                    points_to_calculate.append(ThermoPoint(
                        temperature=T_val,
                        size=L,
                        meas_steps=args.meas_steps,
                        eq_probe_steps=args.eq_probe_steps,
                        eq_max_steps=args.eq_max_steps,
                        eq_qs_sigma_threshold=args.eq_qs_sigma_threshold,
                        eq_qs_min_steps=args.eq_qs_min_steps,
                        qs_allow_stuck=(T_val < _TC_ISING_THEORY),
                        prefer_ordered_start=(T_val < _TC_ISING_THEORY),
                        temperature_index=t,
                        seed_index=seed_idx,
                        seed=seed,
                        model_cls=IsingSimulation,
                        model_kwargs={},
                        confidence=float(args.confidence_level),
                        derived_method=str(args.derived_uncertainty_method),
                        bootstrap_resamples=int(args.derived_bootstrap_resamples),
                    ))
                    attempts_per_temp[t] += 1

        if not points_to_calculate:
            # We've either reached all targets or exhausted all attempts
            break

        # Run the missing points in parallel
        batch_results = parallel_sweep(
            worker_func=simulate_thermo_point,
            params=points_to_calculate,
        )

        # Collect results
        for res in batch_results:
            t_idx = int(res['temperature_index'])
            if res.get('equilibrated_flag', 0.0) > 0:
                results_grid[t_idx].append(res)

    # Final check: did we get enough seeds for all temperatures?
    dropped_counts = [target_n_seeds - len(results_grid[t]) for t in range(t_points)]
    if any(dropped_counts):
        logger.warning(
            f"Could not reach target_n_seeds={target_n_seeds} for all temperatures. "
            f"Missing points: {sum(dropped_counts)}"
        )
        if all(len(r) == 0 for r in results_grid):
            raise RuntimeError("Failed to collect any converged results.")

    # Reconstruct arrays for uncertainty calculation
    # We only use temperatures where we have at least one seed
    valid_t_mask = np.array([len(results_grid[t]) > 0 for t in range(t_points)])
    valid_temperatures = temperatures[valid_t_mask]
    n_valid_t = int(valid_t_mask.sum())

    # For reporting purposes, we use the actual number of seeds we kept per temperature.
    # To simplify bundle logic, we'll pad with NaN if seeds counts vary (though they usually won't).
    max_seeds_retained = max(len(r) for r in results_grid)

    def extract_grid(key: str) -> np.ndarray:
        arr = np.full((n_valid_t, max_seeds_retained), np.nan)
        valid_idx = 0
        for t in range(t_points):
            if not valid_t_mask[t]:
                continue
            for s, res in enumerate(results_grid[t]):
                arr[valid_idx, s] = res.get(key, np.nan)
            valid_idx += 1
        return arr

    avg_m_values = extract_grid('avg_m_value')
    avg_m_errors = extract_grid('avg_m_err')
    avg_m_tau = extract_grid('avg_m_tau_int')
    avg_m_n_eff = extract_grid('avg_m_n_eff')

    avg_e_values = extract_grid('avg_e_value')
    avg_e_errors = extract_grid('avg_e_err')
    avg_e_tau = extract_grid('avg_e_tau_int')
    avg_e_n_eff = extract_grid('avg_e_n_eff')

    susc_values = extract_grid('susc_value')
    susc_errors = extract_grid('susc_err')
    susc_tau = extract_grid('susc_tau_int')
    susc_n_eff = extract_grid('susc_n_eff')

    spec_h_values = extract_grid('spec_h_value')
    spec_h_errors = extract_grid('spec_h_err')
    spec_h_tau = extract_grid('spec_h_tau_int')
    spec_h_n_eff = extract_grid('spec_h_n_eff')

    mag_bundle = build_uncertainty_bundle(
        values_by_seed=avg_m_values,
        errors_by_seed=avg_m_errors,
        tau_by_seed=avg_m_tau,
        n_eff_by_seed=avg_m_n_eff,
        confidence=float(args.confidence_level),
    )
    eng_bundle = build_uncertainty_bundle(
        values_by_seed=avg_e_values,
        errors_by_seed=avg_e_errors,
        tau_by_seed=avg_e_tau,
        n_eff_by_seed=avg_e_n_eff,
        confidence=float(args.confidence_level),
    )
    susc_bundle = build_uncertainty_bundle(
        values_by_seed=susc_values,
        errors_by_seed=susc_errors,
        tau_by_seed=susc_tau,
        n_eff_by_seed=susc_n_eff,
        confidence=float(args.confidence_level),
    )
    spec_h_bundle = build_uncertainty_bundle(
        values_by_seed=spec_h_values,
        errors_by_seed=spec_h_errors,
        tau_by_seed=spec_h_tau,
        n_eff_by_seed=spec_h_n_eff,
        confidence=float(args.confidence_level),
    )

    # Entropy calculation
    entropy_res = summarize_entropy_observable(
        temperatures=valid_temperatures,
        specific_heat_samples=spec_h_bundle['samples'],
        specific_heat_err=spec_h_bundle['err'],
        method=str(args.entropy_uncertainty_method),
        confidence=float(args.confidence_level),
        bootstrap_resamples=int(args.entropy_bootstrap_resamples),
    )

    # Combined quality flags
    quality = build_quality_flags(
        tau_int=mag_bundle['tau_int'],
        ci_low=mag_bundle['ci_low'],
        ci_high=mag_bundle['ci_high'],
        n_eff=mag_bundle['n_eff'],
        min_effective_samples=float(args.min_effective_samples),
        max_tau_relative_width=float(args.max_tau_relative_width),
    )

    if args.strict_uncertainty:
        undef_frac = np.mean(quality['undefined_autocorr_flag'])
        if undef_frac > args.max_undefined_fraction:
            raise RuntimeError(
                f'Strict uncertainty failed: {undef_frac:.1%} of points have '
                'undefined autocorrelation.'
            )

    # Output results
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, 'temperature_sweep_data.npz')

    # Map back to full grid for saving if necessary, or just save valid points
    # We'll save the valid points grid.
    np.savez(
        outpath,
        temperatures=valid_temperatures,
        # Legacy scalar arrays (for notebook compat)
        avg_m=mag_bundle['value'],
        avg_e=eng_bundle['value'],
        susc=susc_bundle['value'],
        spec_h=spec_h_bundle['value'],
        entropy=entropy_res['value'],
        tau_int=mag_bundle['tau_int'],
        # Full uncertainty schema
        **cast(Any, {f'avg_m_{k}': v for k, v in mag_bundle.items()}),
        **cast(Any, {f'avg_e_{k}': v for k, v in eng_bundle.items()}),
        **cast(Any, {f'susc_{k}': v for k, v in susc_bundle.items()}),
        **cast(Any, {f'spec_h_{k}': v for k, v in spec_h_bundle.items()}),
        **cast(Any, {f'entropy_{k}': v for k, v in entropy_res.items()}),
        **cast(Any, quality),
        entropy_uncertainty_method=str(args.entropy_uncertainty_method),
        uncertainty_method=str(args.derived_uncertainty_method),
        confidence_level=float(args.confidence_level),
        n_seeds=max_seeds_retained,
        bootstrap_resamples=int(args.derived_bootstrap_resamples),
        requested_n_seeds=target_n_seeds,
        max_seed_attempts=max_seed_attempts,
        retained_n_seeds=max_seeds_retained,
        nan_or_undefined_count=int(np.sum(quality['undefined_autocorr_flag'])),
    )

    # Build descriptive strings for plot metadata
    metadata = (
        f"L={args.size}, target_n_seeds={args.n_seeds}, "
        f"meas_steps={args.meas_steps}, "
        f"confidence={args.confidence_level}"
    )

    # Optional theory transition for Ising
    transitions = {"Theory": _TC_ISING_THEORY} if args.transition_preset == "theory" else None

    plot_temperature_sweep(
        temperatures=valid_temperatures,
        avg_m=cast(Any, mag_bundle['value']),
        avg_m_err=mag_bundle['err'],
        avg_e=cast(Any, eng_bundle['value']),
        avg_e_err=eng_bundle['err'],
        susc=cast(Any, susc_bundle['value']),
        susc_err=susc_bundle['err'],
        spec_h=cast(Any, spec_h_bundle['value']),
        spec_h_err=spec_h_bundle['err'],
        entropy=entropy_res['value'],
        entropy_err=entropy_res['err'],
        entropy_ci_low=entropy_res['ci_low'],
        entropy_ci_high=entropy_res['ci_high'],
        tau_int=mag_bundle['tau_int'],
        tau_unstable_flag=quality['tau_interval_unstable_flag'],
        low_effective_sample_flag=quality['low_effective_sample_flag'],
        title=f"Ising Model Temperature Sweep ($L={args.size}$)",
        filename="temperature_sweep.png",
        directory=args.output_dir,
        run_metadata_note=metadata,
        transition_temperatures=transitions,
    )


if __name__ == '__main__':
    main()
