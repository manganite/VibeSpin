"""
Analysis of the Berezinskii-Kosterlitz-Thouless (BKT) transition in the 2D XY model.
Counts the average density of vortices as a function of temperature.
"""

import matplotlib.pyplot as plt
import numpy as np

from models.xy_model import XYSimulation
from utils.system_helpers import parallel_sweep, save_plot

# Simulation Parameters
L: int = 40
EQUILIBRATION_STEPS: int = 10000
MEASUREMENT_STEPS: int = 100

# Sweep Parameters
T_MIN: float = 0.5
T_MAX: float = 1.5
T_POINTS: int = 21

# Physical Constants
T_BKT_THEORETICAL: float = 0.893


def simulate_bkt_point(T: float) -> float:
    """
    Worker function to simulate a single temperature and count average vortex density.

    Args:
        T: Temperature.

    Returns:
        Average number of vortices found in the lattice.
    """
    sim = XYSimulation(L, T)
    sim.equilibrate(EQUILIBRATION_STEPS)

    total_vortex_count: int = 0
    for _ in range(MEASUREMENT_STEPS):
        sim.step()
        vorticity = sim._calculate_vorticity()
        # Count all non-zero winding numbers (vortices and anti-vortices)
        total_vortex_count += int(np.sum(np.abs(vorticity)))

    return total_vortex_count / MEASUREMENT_STEPS


def run_bkt_study() -> None:
    """Run temperature sweep to observe vortex proliferation near T_BKT."""
    # Generate temperature points
    temperatures: np.ndarray = np.linspace(T_MIN, T_MAX, T_POINTS)

    print(f"Starting BKT transition study (Vortex counting) for L={L}...")
    print(f"Range: [{T_MIN}, {T_MAX}] with {T_POINTS} points.")

    vortex_counts: list[float] = parallel_sweep(simulate_bkt_point, temperatures)

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
    plt.title(f'Vortex Proliferation in 2D XY Model (L={L})')
    plt.grid(True)
    plt.legend()

    save_plot('bkt_transition.png', directory='results/xy')


if __name__ == "__main__":
    run_bkt_study()
