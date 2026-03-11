from __future__ import annotations
"""
Analysis of the helicity modulus (superfluid stiffness) in the 2D XY model.
Used to identify the universal jump at the BKT transition.
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.xy_model import XYSimulation
from utils.system_helpers import parallel_sweep, save_plot, setup_logging


def simulate_helicity(params: tuple[float, int, int, int]) -> float:
    """
    Run simulation for a single temperature and compute the helicity modulus.

    Args:
        params: Tuple of (T, L, eq_steps, meas_steps).

    Returns:
        Helicity modulus (Upsilon) for this temperature.

    Raises:
        ValueError: If ``T`` is less than or equal to 0.
    """
    T, L, eq_steps, meas_steps = params
    if T <= 0.0:
        raise ValueError(f'Temperature must be positive to compute helicity modulus, got {T}')

    sim = XYSimulation(size=L, temp=T)
    sim.equilibrate(n_steps=eq_steps)

    cos_sums: np.ndarray = np.empty(meas_steps)
    sin_sums: np.ndarray = np.empty(meas_steps)

    for k in range(meas_steps):
        sim.step()
        cos_sums[k], sin_sums[k] = sim._get_helicity_data()

    # Formula: Upsilon = (1/L^2) * (<Sum cos> - (1/T) * <(Sum sin)^2>)
    avg_cos: float = float(np.mean(cos_sums))
    avg_sq_sin: float = float(np.mean(sin_sums**2))

    upsilon: float = (avg_cos - (1.0 / T) * avg_sq_sin) / (L**2)
    return upsilon


def run_helicity_sweep() -> None:
    """
    Run parallel sweep to calculate helicity modulus across the BKT region.
    """
    parser = argparse.ArgumentParser(description='2D XY Model Helicity Modulus Analysis')
    parser.add_argument('--size', type=int, default=64, help='Linear lattice size L')
    parser.add_argument('--eq-steps', type=int, default=10000, help='Equilibration steps')
    parser.add_argument('--meas-steps', type=int, default=20000, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=1.5, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=30, help='Number of temperature points')
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Generate temperature points
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)

    logger.info(f'Starting Helicity Modulus sweep for L={args.size}...')
    logger.info(f'Range: [{args.t_min}, {args.t_max}] with {args.t_points} points.')

    sweep_params = [(T, args.size, args.eq_steps, args.meas_steps) for T in temperatures]
    upsilons: list[float] = parallel_sweep(worker_func=simulate_helicity, params=sweep_params)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, upsilons, 'o-', label=r'Helicity Modulus $\Upsilon$')

    # Plot the universal jump line: 2/pi * T
    t_line: np.ndarray = np.linspace(args.t_min, args.t_max, 100)
    plt.plot(t_line, 2 * t_line / np.pi, 'r--', label=r'Universal Jump $\frac{2}{\pi} k_B T$')

    plt.xlabel('Temperature (T)')
    plt.ylabel(r'Helicity Modulus $\Upsilon$')
    plt.title('BKT Transition: Superfluid Stiffness')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)

    save_plot(filename='helicity_modulus.png', directory=args.output_dir)


if __name__ == '__main__':
    run_helicity_sweep()
