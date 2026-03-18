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
from utils.system_helpers import parallel_sweep, save_plot, setup_logging


def simulate_bkt_point(params: tuple[float, int, int, int]) -> float:
    """
    Worker function to simulate a single temperature and measure average vortex density.

    Parameters
    ----------
        params: Tuple of (T, L, eq_steps, meas_steps).

    Returns
    -------
        Average vortex density n_v.
    """
    T, L, eq_steps, meas_steps = params
    sim = XYSimulation(size=L, temp=T)
    sim.equilibrate(n_steps=eq_steps)

    total_vortex_density = 0.0
    for _ in range(meas_steps):
        sim.step()
        total_vortex_density += sim._get_vortex_density()

    return total_vortex_density / meas_steps


def main() -> None:
    """Run temperature sweep to observe vortex-density growth near T_BKT."""
    parser = argparse.ArgumentParser(description='2D XY Model BKT Transition Analysis')
    parser.add_argument('--size', type=int, default=40, help='Linear lattice size L')
    parser.add_argument('--eq-steps', type=int, default=10000, help='Equilibration steps')
    parser.add_argument('--meas-steps', type=int, default=100, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.5, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=1.5, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=21, help='Number of temperature points')
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Generate temperature points
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)
    T_BKT_THEORETICAL: float = 0.893

    logger.info(f'Starting BKT transition study (vortex density) for L={args.size}...')
    logger.info(f'Range: [{args.t_min}, {args.t_max}] with {args.t_points} points.')

    sweep_params = [(T, args.size, args.eq_steps, args.meas_steps) for T in temperatures]
    vortex_densities: list[float] = parallel_sweep(
        worker_func=simulate_bkt_point, params=sweep_params
    )

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, vortex_densities, 'o-', markersize=5, label='Vortex Density $n_v$')

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

    save_plot(filename='bkt_transition.png', directory=args.output_dir)


if __name__ == '__main__':
    main()
