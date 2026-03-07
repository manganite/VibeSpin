"""
Analysis of the Berezinskii-Kosterlitz-Thouless (BKT) transition in the 2D XY model.
Counts the average density of vortices as a function of temperature.
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.xy_model import XYSimulation
from utils.system_helpers import parallel_sweep, save_plot, setup_logging


def simulate_bkt_point(params: tuple[float, int, int, int]) -> float:
    """
    Worker function to simulate a single temperature and count average vortex density.

    Args:
        params: Tuple of (T, L, eq_steps, meas_steps).

    Returns:
        Average number of vortices found in the lattice.
    """
    T, L, eq_steps, meas_steps = params
    sim = XYSimulation(L, T)
    sim.equilibrate(eq_steps)

    total_vortex_count: int = 0
    for _ in range(meas_steps):
        sim.step()
        vorticity = sim._calculate_vorticity()
        # Count all non-zero winding numbers (vortices and anti-vortices)
        total_vortex_count += int(np.sum(np.abs(vorticity)))

    return total_vortex_count / meas_steps


def run_bkt_study() -> None:
    """Run temperature sweep to observe vortex proliferation near T_BKT."""
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

    logger.info(f'Starting BKT transition study (Vortex counting) for L={args.size}...')
    logger.info(f'Range: [{args.t_min}, {args.t_max}] with {args.t_points} points.')

    sweep_params = [(T, args.size, args.eq_steps, args.meas_steps) for T in temperatures]
    vortex_counts: list[float] = parallel_sweep(simulate_bkt_point, sweep_params)

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, vortex_counts, 'o-', markersize=5, label='Total Vortex Count')

    plt.axvline(
        x=T_BKT_THEORETICAL,
        color='r',
        linestyle='--',
        alpha=0.7,
        label=f'Theoretical $T_{{BKT}} \\approx {T_BKT_THEORETICAL}$',
    )
    plt.xlabel('Temperature (T)')
    plt.ylabel('Average Vortex Count')
    plt.title(f'Vortex Proliferation in 2D XY Model (L={args.size})')
    plt.grid(True)
    plt.legend()

    save_plot('bkt_transition.png', directory=args.output_dir)


if __name__ == '__main__':
    run_bkt_study()
