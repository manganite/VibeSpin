"""
Standardized temperature sweep for the 2D q-state Clock model.
Calculates and plots magnetization, energy, susceptibility, and specific heat.
"""

import numpy as np

from models.clock_model import ClockSimulation
from utils.physics_helpers import calculate_thermodynamics
from utils.system_helpers import parallel_sweep, plot_temperature_sweep

# Simulation Parameters
L: int = 24
Q: int = 6
A: float = 0.1
EQUILIBRATION_STEPS: int = 20000
MEASUREMENT_STEPS: int = 20000

# Sweep Parameters
T_MIN: float = 0.1
T_MAX: float = 2.0
T_POINTS: int = 40


def simulate_temperature(T: float) -> tuple[float, float, float, float]:
    """
    Worker function to simulate a single temperature point for the Clock model.
    """
    sim = ClockSimulation(L, T, A=A, q=Q)
    sim.equilibrate(EQUILIBRATION_STEPS)
    mags, engs = sim.run(MEASUREMENT_STEPS)
    return calculate_thermodynamics(np.array(mags), np.array(engs), T, L)


def run_sweep() -> None:
    """
    Execute the temperature sweep and generate standardized 4-panel plots.
    """
    temperatures: np.ndarray = np.linspace(T_MIN, T_MAX, T_POINTS)

    print(f"Starting {Q}-state Clock temperature sweep (L={L}, A={A})...")
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
        title=f'2D {Q}-state Clock Model: Temperature Sweep (L={L}, A={A})',
        filename='temperature_sweep.png',
        directory='results/clock',
    )


if __name__ == "__main__":
    run_sweep()
