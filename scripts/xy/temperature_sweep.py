"""
Standardized temperature sweep for the 2D XY model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, NamedTuple, cast

import numpy as np

from models.xy_model import XYSimulation
from utils.cli_helpers import parse_args_compat
from utils.physics_helpers import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    UNCERTAINTY_METHOD_BOOTSTRAP,
    summarize_derived_observable,
    summarize_entropy_observable,
    summarize_primary_observable,
    summarize_seed_ensemble,
)
from utils.system_helpers import (
    convergence_equilibrate_with_status,
    parallel_sweep,
    plot_temperature_sweep,
    setup_logging,
)

_WORKER_CONFIDENCE_LEVEL = DEFAULT_CONFIDENCE_LEVEL
_WORKER_DERIVED_METHOD = UNCERTAINTY_METHOD_BLOCKING
_WORKER_DERIVED_BOOTSTRAP_RESAMPLES = 0
_TBKT_XY_THEORY = 0.893


class _SeedSweepPoint(NamedTuple):
    """Typed worker payload for one temperature/seed point in the sweep."""

    temperature: float
    size: int
    meas_steps: int
    eq_probe_steps: int
    eq_max_steps: int
    temperature_index: int
    seed_index: int
    seed: int


def _simulate_seed_temperature(params: _SeedSweepPoint) -> dict[str, float]:
    """Run one seeded sweep point and return summary statistics for all observables."""
    T = params.temperature
    L = params.size
    meas_steps = params.meas_steps

    sim_r = XYSimulation(size=L, temp=T, init_state='random', seed=params.seed)
    sim_o = XYSimulation(size=L, temp=T, init_state='ordered', seed=params.seed)
    total_steps, converged = convergence_equilibrate_with_status(
        sim_r,
        sim_o,
        chunk_size=params.eq_probe_steps,
        max_steps=params.eq_max_steps,
    )

    if not converged:
        return {
            'temperature_index': float(params.temperature_index),
            'seed_index': float(params.seed_index),
            'equilibrated_flag': 0.0,
            'equilibration_steps': float(total_steps),
        }

    mags, engs = sim_r.run(n_steps=meas_steps)
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
        'equilibrated_flag': 1.0,
        'equilibration_steps': float(total_steps),
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


def _build_uncertainty_bundle(
    *,
    values_by_seed: np.ndarray,
    errors_by_seed: np.ndarray,
    tau_by_seed: np.ndarray,
    n_eff_by_seed: np.ndarray,
    confidence: float,
) -> dict[str, np.ndarray]:
    """Calculate combined uncertainty across multiple seeds for one observable."""
    n_seeds = values_by_seed.shape[1]
    if n_seeds > 1:
        # Hierarchical aggregation: between-seed + within-seed
        res_values = []
        res_errors = []
        res_low = []
        res_high = []

        for t in range(values_by_seed.shape[0]):
            mask = ~np.isnan(values_by_seed[t])
            if not np.any(mask):
                res_values.append(np.nan)
                res_errors.append(np.nan)
                res_low.append(np.nan)
                res_high.append(np.nan)
                continue

            summary = summarize_seed_ensemble(
                values=values_by_seed[t, mask],
                within_seed_errors=errors_by_seed[t, mask],
                confidence=confidence
            )
            res_values.append(summary['value'])
            res_errors.append(summary['err'])
            res_low.append(summary['ci_low'])
            res_high.append(summary['ci_high'])

        res = {
            'value': np.array(res_values),
            'err': np.array(res_errors),
            'ci_low': np.array(res_low),
            'ci_high': np.array(res_high),
        }
        tau_int = np.nanmean(tau_by_seed, axis=1)
        n_eff = np.nanmean(n_eff_by_seed, axis=1)
    else:
        res = {
            'value': values_by_seed[:, 0],
            'err': errors_by_seed[:, 0],
            'ci_low': values_by_seed[:, 0] - errors_by_seed[:, 0],
            'ci_high': values_by_seed[:, 0] + errors_by_seed[:, 0],
        }
        tau_int = tau_by_seed[:, 0]
        n_eff = n_eff_by_seed[:, 0]

    return {
        'value': res['value'],
        'err': res['err'],
        'ci_low': res['ci_low'],
        'ci_high': res['ci_high'],
        'tau_int': tau_int,
        'n_eff': n_eff,
        'samples': values_by_seed.astype(np.float64),
    }


def _build_quality_flags(
    *,
    tau_int: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    n_eff: np.ndarray,
    min_effective_samples: float,
    max_tau_relative_width: float,
) -> dict[str, np.ndarray]:
    """Build per-temperature diagnostics for uncertainty quality."""
    undefined_autocorr = ~np.isfinite(tau_int)
    low_effective_sample = np.isfinite(n_eff) & (n_eff < min_effective_samples)

    tau_span = ci_high - ci_low
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
    parser = argparse.ArgumentParser(description='2D XY Model Temperature Sweep')
    parser.add_argument('--size', type=int, default=48, help='Linear lattice size L')
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
    parser.add_argument(
        '--transition-preset',
        type=str,
        default='auto',
        choices=['auto', 'none', 'theory'],
        help='Transition overlay preset for plotting',
    )
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

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
    t_points = len(temperatures)

    target_n_seeds = int(args.n_seeds)
    max_seed_attempts = args.max_seed_attempts
    if max_seed_attempts is None:
        max_seed_attempts = max(target_n_seeds, 10 * target_n_seeds)
    max_seed_attempts = int(max_seed_attempts)

    logger.info(
        'Starting XY temperature sweep '
        f'(L={L}, target_n_seeds={target_n_seeds}, max_seed_attempts={max_seed_attempts})...'
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
                    points_to_calculate.append(_SeedSweepPoint(
                        temperature=float(temperatures[t]),
                        size=L,
                        meas_steps=args.meas_steps,
                        eq_probe_steps=args.eq_probe_steps,
                        eq_max_steps=args.eq_max_steps,
                        temperature_index=t,
                        seed_index=seed_idx,
                        seed=seed
                    ))
                    attempts_per_temp[t] += 1

        if not points_to_calculate:
            break

        batch_results = parallel_sweep(
            worker_func=_simulate_seed_temperature,
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

    mag_bundle = _build_uncertainty_bundle(
        values_by_seed=extract_grid('avg_m_value'),
        errors_by_seed=extract_grid('avg_m_err'),
        tau_by_seed=extract_grid('avg_m_tau_int'),
        n_eff_by_seed=extract_grid('avg_m_n_eff'),
        confidence=float(args.confidence_level),
    )
    eng_bundle = _build_uncertainty_bundle(
        values_by_seed=extract_grid('avg_e_value'),
        errors_by_seed=extract_grid('avg_e_err'),
        tau_by_seed=extract_grid('avg_e_tau_int'),
        n_eff_by_seed=extract_grid('avg_e_n_eff'),
        confidence=float(args.confidence_level),
    )
    susc_bundle = _build_uncertainty_bundle(
        values_by_seed=extract_grid('susc_value'),
        errors_by_seed=extract_grid('susc_err'),
        tau_by_seed=extract_grid('susc_tau_int'),
        n_eff_by_seed=extract_grid('susc_n_eff'),
        confidence=float(args.confidence_level),
    )
    spec_h_bundle = _build_uncertainty_bundle(
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

    quality = _build_quality_flags(
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
        entropy_uncertainty_method=str(args.entropy_uncertainty_method),
        uncertainty_method=str(args.derived_uncertainty_method),
        confidence_level=float(args.confidence_level),
        requested_n_seeds=target_n_seeds,
        max_seed_attempts=max_seed_attempts,
        retained_n_seeds=max_seeds_retained,
        nan_or_undefined_count=int(np.sum(quality['undefined_autocorr_flag'])),
    )

    transitions = {"Theory (BKT)": _TBKT_XY_THEORY} if args.transition_preset == "theory" else None
    metadata = f"L={args.size}, target_n_seeds={args.n_seeds}, meas_steps={args.meas_steps}"

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
        title=f"XY Model Temperature Sweep ($L={args.size}$)",
        filename="temperature_sweep.png",
        directory=args.output_dir,
        run_metadata_note=metadata,
        transition_temperatures=transitions,
    )


if __name__ == '__main__':
    main()
