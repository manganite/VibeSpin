"""
Standardized temperature sweep for the 2D XY model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from models.xy_model import XYSimulation
from utils.cli_helpers import parse_args_compat
from utils.physics_helpers import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    summarize_derived_observable,
    summarize_entropy_observable,
    summarize_primary_observable,
    summarize_seed_ensemble,
)
from utils.system_helpers import (
    convergence_equilibrate,
    parallel_sweep,
    plot_temperature_sweep,
    setup_logging,
)


def simulate_temperature(
    params: _SweepPoint,
) -> tuple[float, float, float, float, float]:
    """
    Worker function to simulate a single temperature point for the XY model.
    """
    T = params.temperature
    L = params.size
    meas_steps = params.meas_steps
    eq_probe_steps = params.eq_probe_steps
    eq_max_steps = params.eq_max_steps

    # Initialize two simulations for the two-start convergence test
    sim_r = XYSimulation(size=L, temp=T, init_state='random')
    sim_o = XYSimulation(size=L, temp=T, init_state='ordered')

    # Robust equilibration via two-start convergence
    convergence_equilibrate(
        sim_r,
        sim_o,
        chunk_size=eq_probe_steps,
        max_steps=eq_max_steps,
    )

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

    sim_r = XYSimulation(size=L, temp=T, init_state='random', seed=params.seed)
    sim_o = XYSimulation(size=L, temp=T, init_state='ordered', seed=params.seed)
    convergence_equilibrate(
        sim_r,
        sim_o,
        chunk_size=params.eq_probe_steps,
        max_steps=params.eq_max_steps,
    )

    mags, engs = sim_r.run(n_steps=params.meas_steps)
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

    return {
        'temperature_index': float(params.temperature_index),
        'seed_index': float(params.seed_index),
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
            confidence=DEFAULT_CONFIDENCE_LEVEL,
        )
        value[i] = float(agg['value'])
        err[i] = float(agg['err'])
        ci_low[i] = float(agg['ci_low'])
        ci_high[i] = float(agg['ci_high'])
        tau_row = np.asarray(tau_by_seed[i], dtype=np.float64)
        tau_agg = summarize_seed_ensemble(
            values=tau_row,
            within_seed_errors=np.full_like(tau_row, np.nan),
            confidence=DEFAULT_CONFIDENCE_LEVEL,
        )
        tau_int[i] = float(tau_agg['value'])
        tau_finite_count = int(np.isfinite(tau_row).sum())
        if tau_finite_count <= 1:
            tau_int_err[i] = np.nan
            tau_int_ci_low[i] = np.nan
            tau_int_ci_high[i] = np.nan
        else:
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
    parser.add_argument('--n-seeds', type=int, default=1, help='Independent seed replicas')
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)

    n_seeds = int(args.n_seeds)
    if n_seeds < 1:
        raise ValueError(f'n-seeds must be >= 1, got {n_seeds}')

    logger.info(f'Starting XY temperature sweep (L={L}, n_seeds={n_seeds})...')
    sweep_params: list[_SeedSweepPoint] = []
    for i, T in enumerate(temperatures):
        for s in range(n_seeds):
            sweep_params.append(
                _SeedSweepPoint(
                    temperature=float(T),
                    size=L,
                    meas_steps=args.meas_steps,
                    eq_probe_steps=args.eq_probe_steps,
                    eq_max_steps=args.eq_max_steps,
                    temperature_index=i,
                    seed_index=s,
                    seed=i * 100_000 + s * 1_000,
                )
            )

    raw: list[dict[str, float]] = parallel_sweep(
        worker_func=_simulate_seed_temperature,
        params=sweep_params,
    )

    shape = (args.t_points, n_seeds)
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

    for item in raw:
        i = int(item['temperature_index'])
        s = int(item['seed_index'])
        avg_m_values[i, s] = item['avg_m_value']
        avg_m_errors[i, s] = item['avg_m_err']
        avg_m_tau[i, s] = item['avg_m_tau_int']
        avg_m_n_eff[i, s] = item['avg_m_n_eff']
        avg_e_values[i, s] = item['avg_e_value']
        avg_e_errors[i, s] = item['avg_e_err']
        avg_e_tau[i, s] = item['avg_e_tau_int']
        avg_e_n_eff[i, s] = item['avg_e_n_eff']
        susc_values[i, s] = item['susc_value']
        susc_errors[i, s] = item['susc_err']
        susc_tau[i, s] = item['susc_tau_int']
        susc_n_eff[i, s] = item['susc_n_eff']
        spec_h_values[i, s] = item['spec_h_value']
        spec_h_errors[i, s] = item['spec_h_err']
        spec_h_tau[i, s] = item['spec_h_tau_int']
        spec_h_n_eff[i, s] = item['spec_h_n_eff']

    mag_bundle = _build_uncertainty_bundle(
        values_by_seed=avg_m_values,
        errors_by_seed=avg_m_errors,
        tau_by_seed=avg_m_tau,
        n_eff_by_seed=avg_m_n_eff,
    )
    eng_bundle = _build_uncertainty_bundle(
        values_by_seed=avg_e_values,
        errors_by_seed=avg_e_errors,
        tau_by_seed=avg_e_tau,
        n_eff_by_seed=avg_e_n_eff,
    )
    susc_bundle = _build_uncertainty_bundle(
        values_by_seed=susc_values,
        errors_by_seed=susc_errors,
        tau_by_seed=susc_tau,
        n_eff_by_seed=susc_n_eff,
    )
    cv_bundle = _build_uncertainty_bundle(
        values_by_seed=spec_h_values,
        errors_by_seed=spec_h_errors,
        tau_by_seed=spec_h_tau,
        n_eff_by_seed=spec_h_n_eff,
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
        confidence=DEFAULT_CONFIDENCE_LEVEL,
    )
    entropy = np.asarray(entropy_bundle['value'], dtype=np.float64)

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
        uncertainty_method=UNCERTAINTY_METHOD_BLOCKING,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        n_seeds=n_seeds,
        bootstrap_resamples=0,
        nan_or_undefined_count=float(np.isnan(tau_int_arr).sum()),
    )

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
        tau_int=tau_int_arr,
        tau_int_ci_low=tau_int_ci_low_arr,
        tau_int_ci_high=tau_int_ci_high_arr,
        title=f'2D XY Model: Temperature Sweep (L={L})',
        filename='temperature_sweep.png',
        directory=args.output_dir,
    )


if __name__ == '__main__':
    main()
