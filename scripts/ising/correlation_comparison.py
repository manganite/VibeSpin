"""
Comparison of spin-spin correlation functions for the 2D Ising model.
Analyzes correlation behavior in ferromagnetic, critical, and paramagnetic phases.
"""

import matplotlib.pyplot as plt
import numpy as np

from models.ising_model import IsingSimulation
from utils.physics_helpers import get_averaged_correlation
from utils.system_helpers import ensure_results_dir, parallel_sweep, save_plot

# Global Parameters
L: int = 64
STEPS: int = 10000
EQUILIBRATION_STEPS: int = 2000
SAMPLE_INTERVAL: int = 20

# Ising 2D Critical Temperature approx 2.269
T_FERRO: float = 1.8   # Below Tc (Long range order)
T_CRIT: float = 2.269  # At Tc (Power law decay)
T_PARA: float = 3.0    # Above Tc (Exponential decay)

# Fitting Parameters
FIT_START_R: int = 2
FIT_END_R: int = 15


def simulate_correlation(T: float) -> tuple[np.ndarray, np.ndarray]:
    """Worker function: simulate and return the averaged correlation function at temperature T.

    Args:
        T: Temperature.

    Returns:
        A tuple of (r, G_r) — radial distances and averaged correlations.
    """
    print(f"Collecting data for T={T}...")
    sim = IsingSimulation(L, T)
    sim.equilibrate(EQUILIBRATION_STEPS)
    return get_averaged_correlation(sim, STEPS, SAMPLE_INTERVAL)


if __name__ == "__main__":
    temperatures = [T_FERRO, T_CRIT, T_PARA]
    (r, G_ferro), (_, G_crit), (_, G_para) = parallel_sweep(simulate_correlation, temperatures)

    # --- Fit for correlation length xi in paramagnetic phase ---
    r_fit: np.ndarray = r[FIT_START_R:FIT_END_R]
    G_para_fit: np.ndarray = G_para[FIT_START_R:FIT_END_R]

    # Ensure we only fit positive values to avoid log(0) errors
    valid_indices: np.ndarray = G_para_fit > 1e-10
    if np.any(valid_indices):
        log_G_para_fit: np.ndarray = np.log(G_para_fit[valid_indices])
        r_fit_valid: np.ndarray = r_fit[valid_indices]

        try:
            slope, intercept = np.polyfit(r_fit_valid, log_G_para_fit, 1)
            if slope == 0.0:
                raise ValueError("Fitted slope is zero; cannot compute correlation length.")
            xi: float = -1.0 / slope
            print(f"\nFitted correlation length for T={T_PARA} (paramagnetic): xi = {xi:.4f}")
            fit_line: np.ndarray = np.exp(intercept + slope * r)
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(f"Warning: exponential fit failed for T={T_PARA}: {exc}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Log-Log Plot (Best for Power Law / Critical)
    ax1.loglog(r[1:], G_ferro[1:], 's-', label=f'T={T_FERRO} (T < Tc)', alpha=0.7)
    ax1.loglog(r[1:], G_crit[1:], 'o-', label=f'T={T_CRIT} (T ~ Tc)', alpha=0.7)
    ax1.loglog(r[1:], G_para[1:], 'x-', label=f'T={T_PARA} (T > Tc)', alpha=0.7)
    ax1.set_title('Log-Log Plot')
    ax1.set_xlabel('Distance r')
    ax1.set_ylabel('Correlation G(r)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # 2. Semi-Log Plot (Best for Exponential / High T)
    ax2.plot(r, G_ferro, 's-', label=f'T={T_FERRO} (T < Tc)', alpha=0.7)
    ax2.plot(r, G_crit, 'o-', label=f'T={T_CRIT} (T ~ Tc)', alpha=0.7)
    ax2.plot(r, G_para, 'x-', label=f'T={T_PARA} (T > Tc)', alpha=0.7)
    if 'xi' in locals():
        ax2.plot(r, fit_line, 'r--', linewidth=2, label=f'Fit ($\\xi={xi:.2f}$)')
    ax2.set_yscale('log')
    ax2.set_title('Semi-Log Plot')
    ax2.set_xlabel('Distance r')
    ax2.set_ylabel('Correlation G(r)')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    output_dir: str = ensure_results_dir('results/ising')
    save_plot('correlation_comparison.png', directory=output_dir)
    print(f"Comparison plot saved to {output_dir}/correlation_comparison.png")
