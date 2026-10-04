"""
Analysis of the Berezinskii-Kosterlitz-Thouless (BKT) transition in the 2D XY model.
Measures the average vortex density as a function of temperature.
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.xy_model import XYSimulation
from utils.equilibration import prepare_equilibrated_simulation
from utils.plotting import ensure_results_dir, save_plot
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    summarize_primary_observable,
)
from utils.sweep_helpers import build_single_run_schema, derive_point_seed
from utils.system import parallel_sweep, parse_args_compat, setup_logging


def simulate_bkt_point(params: tuple[float, int, int, int, int, int]) -> dict[str, float]:
    """
    Worker function to simulate a single temperature and measure average vortex density.

    Parameters
    ----------
    params : tuple[float, int, int, int, int, int]
        Tuple of (T, L, eq_probe_steps, eq_max_steps, meas_steps, seed). The
        seed belongs to the random start; the ordered start derives its own.

    Returns
    -------
    dict[str, float]
        Blocking summary of the vortex density n_v (``value``, ``err``,
        ``ci_low``, ``ci_high``, ``tau_int``, ``n_eff``, ``samples``).
    """
    T, L, eq_probe_steps, eq_max_steps, meas_steps, seed = params
    # The XY model has no metastable domain states, but its random start can
    # relax for thousands of sweeps near T_BKT; the stuck detector would end
    # such runs early, so the pair runs until it converges. If it never does,
    # the ordered start is measured.
    sim, _ = prepare_equilibrated_simulation(
        model_cls=XYSimulation, model_kwargs={}, size=L, temp=T, seed=seed,
        chunk_size=eq_probe_steps, max_steps=eq_max_steps, detect_stuck=False,
    )

    densities = np.empty(meas_steps, dtype=np.float64)
    for k in range(meas_steps):
        sim.step()
        densities[k] = sim.get_vortex_density()

    return summarize_primary_observable(
        time_series=densities, confidence=DEFAULT_CONFIDENCE_LEVEL,
    )


def main() -> None:
    """Run temperature sweep to observe vortex-density growth near T_BKT."""
    parser = argparse.ArgumentParser(description='2D XY Model BKT Transition Analysis')
    parser.add_argument('--size', type=int, default=40, help='Linear lattice size L')
    parser.add_argument(
        '--eq-probe-steps', type=int, default=1000,
        help='Chunk size for convergence check during equilibration (default: 1000)',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=200000,
        help='Hard cap on equilibration steps (default: 200000)',
    )
    parser.add_argument('--meas-steps', type=int, default=100, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.5, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=1.5, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=21, help='Number of temperature points')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Replica index selecting a reproducible seed per temperature (default: 0)',
    )
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Generate temperature points
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)
    T_BKT_THEORETICAL: float = 0.893

    logger.info(f'Starting BKT transition study (vortex density) for L={args.size}...')
    logger.info(f'Range: [{args.t_min}, {args.t_max}] with {args.t_points} points.')

    sweep_params = [
        (
            float(T), args.size, args.eq_probe_steps, args.eq_max_steps, args.meas_steps,
            derive_point_seed(temperature_index=i, seed_index=args.seed),
        )
        for i, T in enumerate(temperatures)
    ]
    summaries: list[dict[str, float]] = parallel_sweep(
        worker_func=simulate_bkt_point, params=sweep_params
    )
    vortex_densities = np.array([s['value'] for s in summaries])
    density_err = np.array([s['err'] for s in summaries])

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        temperatures, vortex_densities, yerr=density_err, fmt='o-', markersize=5,
        capsize=2, label='Vortex Density $n_v$',
    )

    plt.axvline(
        x=T_BKT_THEORETICAL,
        color='r',
        linestyle='--',
        alpha=0.7,
        label=f'Theoretical $T_{{BKT}} \\approx {T_BKT_THEORETICAL}$',
    )
    plt.xlabel('Temperature (T)')
    plt.ylabel('Average Vortex Density $n_v$')
    plt.title(f'Vortex Density in 2D XY Model (L={args.size})')
    plt.grid(True)
    plt.legend()

    output_dir: str = ensure_results_dir(directory=args.output_dir)
    save_plot(filename='bkt_transition.png', directory=output_dir)

    # Save data for notebook consumption
    npz_path = f'{output_dir}/bkt_transition.npz'
    np.savez_compressed(
        npz_path,
        temperatures=temperatures,
        vortex_densities=vortex_densities,
        **build_single_run_schema(
            prefix='vortex_density', summaries=summaries,
            uncertainty_method=UNCERTAINTY_METHOD_BLOCKING,
            confidence=DEFAULT_CONFIDENCE_LEVEL,
        ),
        T_BKT_theoretical=T_BKT_THEORETICAL,
        L=args.size,
        eq_probe_steps=args.eq_probe_steps,
        eq_max_steps=args.eq_max_steps,
        meas_steps=args.meas_steps,
        seed_index=args.seed,
    )
    logger.info(f'Data saved to {npz_path}')


if __name__ == '__main__':
    main()
