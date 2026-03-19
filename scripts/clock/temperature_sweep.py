"""
Standardized temperature sweep for the 2D q-state Clock model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from utils.cli_helpers import parse_args_compat
from utils.exceptions import ZeroVarianceAutocorrelationError
from utils.physics_helpers import calculate_autocorr, calculate_entropy, calculate_thermodynamics
from utils.system_helpers import (
    convergence_equilibrate,
    parallel_sweep,
    plot_temperature_sweep,
    setup_logging,
)


def simulate_temperature(
    params: tuple[float, int, int, float, int, int, int, float, int, bool],
) -> tuple[float, float, float, float, float]:
    """
    Worker function to simulate a single temperature point for the Clock model.
    """
    T, L, Q, A, eq_steps, meas_steps, eq_probe_steps, eq_factor, eq_max_steps, discrete = params
    sim_r: ClockSimulation | DiscreteClockSimulation
    sim_o: ClockSimulation | DiscreteClockSimulation
    if discrete:
        sim_r = DiscreteClockSimulation(size=L, temp=T, q=Q, init_state='random')
        sim_o = DiscreteClockSimulation(size=L, temp=T, q=Q, init_state='ordered')
    else:
        sim_r = ClockSimulation(size=L, temp=T, A=A, q=Q, init_state='random')
        sim_o = ClockSimulation(size=L, temp=T, A=A, q=Q, init_state='ordered')

    convergence_equilibrate(
        sim_r,
        sim_o,
        chunk_size=eq_probe_steps,
        max_steps=eq_max_steps,
    )

    mags, engs = sim_r.run(n_steps=meas_steps)
    mags_arr = np.array(mags)
    thermo = calculate_thermodynamics(mags=mags_arr, engs=np.array(engs), T=T, L=L)
    try:
        _, tau = calculate_autocorr(time_series=mags_arr)
    except ZeroVarianceAutocorrelationError:
        # Fully ordered windows can have zero variance; mark tau as undefined.
        tau = float('nan')
    return (*thermo, tau)


def main() -> None:
    """
    Execute the temperature sweep and generate standardized 4-panel plots.
    """
    parser = argparse.ArgumentParser(description='2D Clock Model Temperature Sweep')
    parser.add_argument('--size', type=int, default=48, help='Linear lattice size L')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    parser.add_argument('--aniso', type=float, default=0.1, help='Anisotropy strength A')
    parser.add_argument(
        '--eq-probe-steps', type=int, default=500,
        help='Chunk size for convergence check during equilibration',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=200000,
        help='Hard cap on total equilibration steps',
    )
    parser.add_argument('--meas-steps', type=int, default=20000, help='Measurement steps')
    parser.add_argument(
        '--discrete', action='store_true',
        help='Use discrete integer-spin clock model instead of continuous+anisotropy',
    )
    parser.add_argument('--t-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=2.0, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=40, help='Number of temperature points')
    parser.add_argument('--output-dir', type=str, default='results/clock', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    Q = args.q
    A = args.aniso
    discrete = args.discrete
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)

    variant = 'discrete' if discrete else f'continuous (A={A})'
    logger.info(f'Starting {Q}-state Clock temperature sweep (L={L}, {variant})...')
    # Bundle parameters for parallel sweep
    sweep_params = [
        (
            T,
            L,
            Q,
            A,
            0,  # placeholder for removed eq_steps
            args.meas_steps,
            args.eq_probe_steps,
            0.0, # placeholder for removed eq_factor
            args.eq_max_steps,
            discrete,
        )
        for T in temperatures
    ]

    results: list[tuple[float, float, float, float, float]] = parallel_sweep(
        worker_func=simulate_temperature, params=sweep_params
    )
    avg_m, avg_e, susc, spec_h, tau_int_vals = zip(*results, strict=True)
    entropy = calculate_entropy(
        temperatures=temperatures, specific_heat=np.array(spec_h),
        s_ref=np.log(Q),
    )

    plot_temperature_sweep(
        temperatures=temperatures,
        avg_m=avg_m,
        avg_e=avg_e,
        susc=susc,
        spec_h=spec_h,
        entropy=entropy,
        tau_int=tau_int_vals,
        title=f'2D {Q}-state Clock Model: Temperature Sweep (L={L}, {variant})',
        filename='temperature_sweep.png',
        directory=args.output_dir,
    )


if __name__ == '__main__':
    main()
