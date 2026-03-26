"""
Standardized temperature sweep for the 2D Clock model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, cast

import numpy as np

from models.clock_model import ClockSimulation
from utils.analysis import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    UNCERTAINTY_METHOD_BOOTSTRAP,
    summarize_entropy_observable,
)
from utils.plotting import plot_temperature_sweep
from utils.sweep_helpers import (
    ThermoPoint,
    build_quality_flags,
    build_uncertainty_bundle,
    simulate_thermo_point,
)
from utils.system import parallel_sweep, parse_args_compat, setup_logging

# Default equilibration parameters forwarded to convergence_equilibrate_with_status
# when not exposed as CLI arguments.
_EQ_QS_SIGMA_THRESHOLD_DEFAULT = 0.05
_EQ_QS_MIN_STEPS_DEFAULT = 1500


def main() -> None:
    """
    Execute the temperature sweep and generate standardized 4-panel plots.
    """
    parser = argparse.ArgumentParser(description='2D Clock Model Temperature Sweep')
    parser.add_argument('--size', type=int, default=32, help='Linear lattice size L')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    parser.add_argument('--aniso', type=float, default=0.0, help='Anisotropy parameter A')
    parser.add_argument('--discrete', action='store_true', help='Use discrete angle representation')
    parser.add_argument(
        '--eq-probe-steps', type=int, default=500,
        help='Chunk size for convergence check during equilibration',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=200_000,
        help='Hard cap on total equilibration steps',
    )
    parser.add_argument('--meas-steps', type=int, default=5000, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=2.0, help='Maximum temperature')
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
    parser.add_argument('--output-dir', type=str, default='results/clock', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

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

    variant = "discrete" if args.discrete else f"continuous (A={args.aniso})"
    logger.info(
        f'Starting {args.q}-state Clock temperature sweep '
        f'(L={L}, {variant}, n_seeds={target_n_seeds})...'
    )

    results_grid: list[list[dict[str, float]]] = [[] for _ in range(t_points)]
    attempts_per_temp = np.zeros(t_points, dtype=int)

    while any(len(results_grid[t]) < target_n_seeds for t in range(t_points)):
        points_to_calculate = []
        for t in range(t_points):
            needed = target_n_seeds - len(results_grid[t])
            if needed > 0 and attempts_per_temp[t] < max_seed_attempts:
                for _ in range(needed):
                    seed_idx = attempts_per_temp[t]
                    seed = t * 100_000 + seed_idx * 1_000
                    points_to_calculate.append(ThermoPoint(
                        temperature=float(temperatures[t]),
                        size=L,
                        meas_steps=args.meas_steps,
                        seed=seed,
                        temperature_index=t,
                        seed_index=seed_idx,
                        eq_probe_steps=args.eq_probe_steps,
                        eq_max_steps=args.eq_max_steps,
                        eq_qs_sigma_threshold=0.05,
                        eq_qs_min_steps=1500,
                        qs_allow_stuck=False,
                        prefer_ordered_start=False,
                        model_cls=ClockSimulation,
                        model_kwargs={'q': args.q, 'A': args.aniso},
                        confidence=float(args.confidence_level),
                        derived_method=str(args.derived_uncertainty_method),
                        bootstrap_resamples=int(args.derived_bootstrap_resamples),
                    ))
                    attempts_per_temp[t] += 1

        if not points_to_calculate:
            break

        batch_results = parallel_sweep(
            worker_func=simulate_thermo_point,
            params=points_to_calculate,
        )

        for res in batch_results:
            t_idx = int(res['temperature_index'])
            if res.get('equilibrated_flag', 0.0) > 0:
                results_grid[t_idx].append(res)

    valid_t_mask = np.array([len(results_grid[t]) > 0 for t in range(t_points)])
    valid_temperatures = temperatures[valid_t_mask]
    n_valid_t = int(valid_t_mask.sum())

    if n_valid_t == 0:
        raise RuntimeError("Failed to collect any converged results.")

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

    mag_bundle = build_uncertainty_bundle(
        values_by_seed=extract_grid('avg_m_value'),
        errors_by_seed=extract_grid('avg_m_err'),
        tau_by_seed=extract_grid('avg_m_tau_int'),
        n_eff_by_seed=extract_grid('avg_m_n_eff'),
        confidence=float(args.confidence_level),
    )
    eng_bundle = build_uncertainty_bundle(
        values_by_seed=extract_grid('avg_e_value'),
        errors_by_seed=extract_grid('avg_e_err'),
        tau_by_seed=extract_grid('avg_e_tau_int'),
        n_eff_by_seed=extract_grid('avg_e_n_eff'),
        confidence=float(args.confidence_level),
    )
    susc_bundle = build_uncertainty_bundle(
        values_by_seed=extract_grid('susc_value'),
        errors_by_seed=extract_grid('susc_err'),
        tau_by_seed=extract_grid('susc_tau_int'),
        n_eff_by_seed=extract_grid('susc_n_eff'),
        confidence=float(args.confidence_level),
    )
    spec_h_bundle = build_uncertainty_bundle(
        values_by_seed=extract_grid('spec_h_value'),
        errors_by_seed=extract_grid('spec_h_err'),
        tau_by_seed=extract_grid('spec_h_tau_int'),
        n_eff_by_seed=extract_grid('spec_h_n_eff'),
        confidence=float(args.confidence_level),
    )

    entropy_res = summarize_entropy_observable(
        temperatures=valid_temperatures,
        specific_heat_samples=spec_h_bundle['samples'],
        specific_heat_err=spec_h_bundle['err'],
        method=str(args.entropy_uncertainty_method),
        confidence=float(args.confidence_level),
        bootstrap_resamples=int(args.entropy_bootstrap_resamples),
    )

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
            raise RuntimeError(f'Strict uncertainty failed: {undef_frac:.1%} undefined.')

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, 'temperature_sweep_data.npz')
    np.savez(
        outpath,
        temperatures=valid_temperatures,
        avg_m=mag_bundle['value'],
        avg_e=eng_bundle['value'],
        susc=susc_bundle['value'],
        spec_h=spec_h_bundle['value'],
        entropy=entropy_res['value'],
        tau_int=mag_bundle['tau_int'],
        **cast(Any, {f'avg_m_{k}': v for k, v in mag_bundle.items()}),
        **cast(Any, {f'avg_e_{k}': v for k, v in eng_bundle.items()}),
        **cast(Any, {f'susc_{k}': v for k, v in susc_bundle.items()}),
        **cast(Any, {f'spec_h_{k}': v for k, v in spec_h_bundle.items()}),
        **cast(Any, {f'entropy_{k}': v for k, v in entropy_res.items()}),
        **cast(Any, quality),
        uncertainty_method=str(args.derived_uncertainty_method),
        confidence_level=float(args.confidence_level),
        requested_n_seeds=target_n_seeds,
        max_seed_attempts=max_seed_attempts,
        retained_n_seeds=max_seeds_retained,
        nan_or_undefined_count=int(np.sum(quality['undefined_autocorr_flag'])),
    )

    metadata = f"L={args.size}, q={args.q}, {variant}, target_n_seeds={args.n_seeds}"

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
        title=f"{args.q}-state Clock Temperature Sweep ($L={args.size}$)",
        filename="temperature_sweep.png",
        directory=args.output_dir,
        run_metadata_note=metadata,
    )


if __name__ == '__main__':
    main()
