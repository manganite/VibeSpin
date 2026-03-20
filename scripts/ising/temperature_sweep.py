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
from utils.exceptions import ZeroVarianceAutocorrelationError
from utils.physics_helpers import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    calculate_autocorr,
    calculate_entropy,
    calculate_thermodynamics,
    summarize_replicate_samples,
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
    )

    # Use the converged random-start instance for measurement
    mags, engs = sim_r.run(n_steps=meas_steps)
    mags_arr = np.array(mags)
    thermo = calculate_thermodynamics(mags=mags_arr, engs=np.array(engs), T=T, L=L)
    try:
        _, tau = calculate_autocorr(time_series=mags_arr)
    except ZeroVarianceAutocorrelationError:
        # Fully ordered windows can have zero variance; mark tau as undefined.
        tau = float('nan')
    return (*thermo, tau)


class _SweepPoint(NamedTuple):
    """Typed worker payload for one temperature point in the sweep."""

    temperature: float
    size: int
    meas_steps: int
    eq_probe_steps: int
    eq_max_steps: int


def _build_uncertainty_bundle(
    *,
    value: np.ndarray,
    tau_int: np.ndarray,
    meas_steps: int,
) -> dict[str, np.ndarray | float]:
    """Build a standardized uncertainty bundle for single-seed temperature sweeps."""
    summary = summarize_replicate_samples(samples=value[:, None])
    n_eff = np.where(
        np.isfinite(tau_int) & (tau_int > 0.0),
        np.minimum(float(meas_steps), meas_steps / (2.0 * tau_int)),
        np.nan,
    )
    return {
        'value': np.asarray(summary['value'], dtype=np.float64),
        'err': np.full_like(value, np.nan, dtype=np.float64),
        'ci_low': np.asarray(summary['ci_low'], dtype=np.float64),
        'ci_high': np.asarray(summary['ci_high'], dtype=np.float64),
        'tau_int': tau_int.astype(np.float64),
        'n_eff': n_eff.astype(np.float64),
        'samples': value[:, None].astype(np.float64),
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
        '--eq-max-steps', type=int, default=200000,
        help='Hard cap on total equilibration steps',
    )
    parser.add_argument('--meas-steps', type=int, default=5000, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=4.0, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=40, help='Number of temperature points')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)

    logger.info(f'Starting Ising temperature sweep (L={L})...')
    # Bundle parameters for parallel sweep
    sweep_params: list[_SweepPoint] = [
        _SweepPoint(
            temperature=T,
            size=L,
            meas_steps=args.meas_steps,
            eq_probe_steps=args.eq_probe_steps,
            eq_max_steps=args.eq_max_steps,
        )
        for T in temperatures
    ]

    results: list[tuple[float, float, float, float, float]] = parallel_sweep(
        worker_func=simulate_temperature, params=sweep_params
    )
    avg_m, avg_e, susc, spec_h, tau_int_vals = zip(*results, strict=True)
    avg_m_arr = np.asarray(avg_m, dtype=np.float64)
    avg_e_arr = np.asarray(avg_e, dtype=np.float64)
    susc_arr = np.asarray(susc, dtype=np.float64)
    spec_h_arr = np.asarray(spec_h, dtype=np.float64)
    tau_int_arr = np.asarray(tau_int_vals, dtype=np.float64)

    entropy = calculate_entropy(
        temperatures=temperatures,
        specific_heat=spec_h_arr,
    )

    mag_bundle = _build_uncertainty_bundle(
        value=avg_m_arr,
        tau_int=tau_int_arr,
        meas_steps=args.meas_steps,
    )
    eng_bundle = _build_uncertainty_bundle(
        value=avg_e_arr,
        tau_int=tau_int_arr,
        meas_steps=args.meas_steps,
    )
    susc_bundle = _build_uncertainty_bundle(
        value=susc_arr,
        tau_int=tau_int_arr,
        meas_steps=args.meas_steps,
    )
    cv_bundle = _build_uncertainty_bundle(
        value=spec_h_arr,
        tau_int=tau_int_arr,
        meas_steps=args.meas_steps,
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
        uncertainty_method=UNCERTAINTY_METHOD_BLOCKING,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        n_seeds=1,
        bootstrap_resamples=0,
        nan_or_undefined_count=float(np.isnan(tau_int_arr).sum()),
    )

    plot_temperature_sweep(
        temperatures=temperatures,
        avg_m=avg_m_arr.tolist(),
        avg_e=avg_e_arr.tolist(),
        susc=susc_arr.tolist(),
        spec_h=spec_h_arr.tolist(),
        entropy=entropy,
        tau_int=tau_int_arr,
        title=f'2D Ising Model: Temperature Sweep (L={L})',
        filename='temperature_sweep.png',
        directory=args.output_dir,
    )


if __name__ == '__main__':
    main()
