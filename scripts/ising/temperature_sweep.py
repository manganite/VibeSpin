"""
Standardized temperature sweep for the 2D Ising model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""

import argparse

import numpy as np

from models.ising_model import IsingSimulation
from utils.physics_helpers import calculate_thermodynamics
from utils.system_helpers import parallel_sweep, plot_temperature_sweep


def simulate_temperature(params: tuple[float, int, int, int]) -> tuple[float, float, float, float]:
    """
    Worker function to simulate a single temperature point for the Ising model.
    """
    T, L, eq_steps, meas_steps = params
    sim = IsingSimulation(L, T)
    sim.equilibrate(eq_steps)
    mags, engs = sim.run(meas_steps)
    return calculate_thermodynamics(np.array(mags), np.array(engs), T, L)


def run_sweep() -> None:
    """
    Execute the temperature sweep and generate standardized 4-panel plots.
    """
    parser = argparse.ArgumentParser(description='2D Ising Model Temperature Sweep')
    parser.add_argument('--size', type=int, default=64, help='Linear lattice size L')
    parser.add_argument('--eq-steps', type=int, default=5000, help='Equilibration steps')
    parser.add_argument('--meas-steps', type=int, default=5000, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=4.0, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=40, help='Number of temperature points')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    L = args.size
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)

    print(f"Starting Ising temperature sweep (L={L})...")
    # Bundle parameters for parallel sweep
    sweep_params = [(T, L, args.eq_steps, args.meas_steps) for T in temperatures]

    results: list[tuple[float, float, float, float]] = parallel_sweep(
        simulate_temperature, sweep_params
    )
    avg_m, avg_e, susc, spec_h = zip(*results, strict=True)

    plot_temperature_sweep(
        temperatures,
        avg_m,
        avg_e,
        susc,
        spec_h,
        title=f'2D Ising Model: Temperature Sweep (L={L})',
        filename='temperature_sweep.png',
        directory=args.output_dir,
    )


if __name__ == "__main__":
    run_sweep()
