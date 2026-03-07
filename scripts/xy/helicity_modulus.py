"""
Analysis of the helicity modulus (superfluid stiffness) in the 2D XY model.
Used to identify the universal jump at the BKT transition.
"""

import matplotlib.pyplot as plt
import numpy as np

from models.xy_model import XYSimulation
from utils.system_helpers import parallel_sweep, save_plot

# Simulation Parameters
L: int = 64
EQUILIBRATION_STEPS: int = 10000
MEASUREMENT_STEPS: int = 20000

# Sweep Parameters
T_MIN: float = 0.1
T_MAX: float = 1.5
T_POINTS: int = 30

def simulate_helicity(T: float) -> float:
    """
    Run simulation for a single temperature and compute the helicity modulus.

    Args:
        T: Temperature.

    Returns:
        Helicity modulus (Upsilon) for this temperature.

    Raises:
        ValueError: If ``T`` is less than or equal to 0.
    """
    if T <= 0.0:
        raise ValueError(f"Temperature must be positive to compute helicity modulus, got {T}")

    sim = XYSimulation(L, T)
    sim.equilibrate(EQUILIBRATION_STEPS)

    cos_sums: np.ndarray = np.empty(MEASUREMENT_STEPS)
    sin_sums: np.ndarray = np.empty(MEASUREMENT_STEPS)

    for k in range(MEASUREMENT_STEPS):
        sim.step()
        cos_sums[k], sin_sums[k] = sim._get_helicity_data()

    # Formula: Upsilon = (1/L^2) * (<Sum cos> - (1/T) * <(Sum sin)^2>)
    avg_cos: float = float(np.mean(cos_sums))
    avg_sq_sin: float = float(np.mean(sin_sums**2))

    upsilon: float = (avg_cos - (1.0/T) * avg_sq_sin) / (L**2)
    return upsilon

def run_helicity_sweep() -> None:
    """
    Run parallel sweep to calculate helicity modulus across the BKT region.
    """
    # Generate temperature points
    temperatures: np.ndarray = np.linspace(T_MIN, T_MAX, T_POINTS)

    print(f"Starting Helicity Modulus sweep for L={L}...")
    print(f"Range: [{T_MIN}, {T_MAX}] with {T_POINTS} points.")

    upsilons: list[float] = parallel_sweep(simulate_helicity, temperatures)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, upsilons, 'o-', label=r'Helicity Modulus $\Upsilon$')

    # Plot the universal jump line: 2/pi * T
    t_line: np.ndarray = np.linspace(T_MIN, T_MAX, 100)
    plt.plot(t_line, 2 * t_line / np.pi, 'r--', label=r'Universal Jump $\frac{2}{\pi} k_B T$')

    plt.xlabel('Temperature (T)')
    plt.ylabel(r'Helicity Modulus $\Upsilon$')
    plt.title('BKT Transition: Superfluid Stiffness')
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)

    save_plot('helicity_modulus.png', directory='results/xy')

if __name__ == "__main__":
    run_helicity_sweep()
