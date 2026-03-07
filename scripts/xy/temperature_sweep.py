"""
Standardized temperature sweep for the 2D XY model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""

import numpy as np

from models.xy_model import XYSimulation
from utils.physics_helpers import calculate_thermodynamics
from utils.system_helpers import parallel_sweep, plot_temperature_sweep

# Simulation Parameters
L: int = 32
EQUILIBRATION_STEPS: int = 5000
MEASUREMENT_STEPS: int = 5000

# Sweep Parameters
T_MIN: float = 0.1
T_MAX: float = 2.0
T_POINTS: int = 40


def simulate_temperature(T: float) -> tuple[float, float, float, float]:
    """
    Worker function to simulate a single temperature point for the XY model.
    """
    sim = XYSimulation(L, T)
    sim.equilibrate(EQUILIBRATION_STEPS)
    mags, engs = sim.run(MEASUREMENT_STEPS)
    return calculate_thermodynamics(np.array(mags), np.array(engs), T, L)


def run_sweep() -> None:
    """
    Execute the temperature sweep and generate standardized 4-panel plots.
    """
    temperatures: np.ndarray = np.linspace(T_MIN, T_MAX, T_POINTS)

    print(f"Starting XY temperature sweep (L={L})...")
    results: list[tuple[float, float, float, float]] = parallel_sweep(
        simulate_temperature, temperatures
    )
    avg_m, avg_e, susc, spec_h = zip(*results, strict=True)

    plot_temperature_sweep(
        temperatures,
        avg_m,
        avg_e,
        susc,
        spec_h,
        title=f'2D XY Model: Temperature Sweep (L={L})',
        filename='temperature_sweep.png',
        directory='results/xy',
    )


if __name__ == "__main__":
    run_sweep()
