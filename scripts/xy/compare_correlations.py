"""
Comparison of spin-spin correlation functions G(r) for the XY model.
Contrasts power-law decay (low T) with exponential decay (high T).
"""

import numpy as np
import matplotlib.pyplot as plt

from models.xy_model import XYSimulation
from utils.system_helpers import save_plot, ensure_results_dir, parallel_sweep
from utils.physics_helpers import get_averaged_correlation

# Global Parameters
L: int = 50
STEPS: int = 10000
EQUILIBRATION_STEPS: int = 2000
SAMPLE_INTERVAL: int = 20

# Temperatures
T_LOW: float = 0.4   # Well below BKT (Power law expected)
T_HIGH: float = 1.5  # Well above BKT (Exponential expected)


def simulate_correlation(T: float) -> tuple[np.ndarray, np.ndarray]:
    """Worker function: simulate and return the averaged correlation function at temperature T.

    Args:
        T: Temperature.

    Returns:
        A tuple of (r, G_r) — radial distances and averaged correlations.
    """
    print(f"Collecting data for T={T}...")
    sim = XYSimulation(L, T)
    sim.equilibrate(EQUILIBRATION_STEPS)
    return get_averaged_correlation(sim, STEPS, SAMPLE_INTERVAL)


if __name__ == "__main__":
    temperatures = [T_LOW, T_HIGH]
    (r_low, G_low), (r_high, G_high) = parallel_sweep(simulate_correlation, temperatures)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Log-Log Plot (Best for Power Law / Low T)
    # We skip r=0 to avoid log(0)
    ax1.loglog(r_low[1:], G_low[1:], 'o-', label=f'T={T_LOW} (Low Temp)')
    ax1.loglog(r_high[1:], G_high[1:], 'x-', label=f'T={T_HIGH} (High Temp)')
    ax1.set_title('Log-Log Plot\n(Straight line = Power Law Decay)')
    ax1.set_xlabel('Distance r')
    ax1.set_ylabel('Correlation G(r)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # 2. Semi-Log Plot (Best for Exponential / High T)
    ax2.plot(r_low, G_low, 'o-', label=f'T={T_LOW} (Low Temp)')
    ax2.plot(r_high, G_high, 'x-', label=f'T={T_HIGH} (High Temp)')
    ax2.set_yscale('log')
    ax2.set_title('Semi-Log Plot\n(Straight line = Exponential Decay)')
    ax2.set_xlabel('Distance r')
    ax2.set_ylabel('Correlation G(r)')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    output_dir: str = ensure_results_dir('results/xy')
    save_plot('correlation_comparison.png', directory=output_dir)
    print(f"Comparison plot saved to {output_dir}/correlation_comparison.png")
