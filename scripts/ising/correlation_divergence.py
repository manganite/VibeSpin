"""
Analysis of correlation length divergence in the 2D Ising model.
Extracts the critical exponent nu by fitting correlation lengths near Tc.
"""

import matplotlib.pyplot as plt
import numpy as np

from models.ising_model import IsingSimulation
from utils.physics_helpers import get_averaged_correlation
from utils.system_helpers import ensure_results_dir, parallel_sweep, save_plot

# Global Parameters
L: int = 128
STEPS: int = 50000
EQUILIBRATION_STEPS: int = 10000
SAMPLE_INTERVAL: int = 20

# Physical Constants
TC_THEORETICAL: float = 2.269

# Sweep Temperatures (Paramagnetic phase T > Tc)
TEMPERATURES: list[float] = [2.4, 2.45, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5]


def get_correlation_length(T: float) -> tuple[float, float]:
    """Simulate and extract correlation length xi for a given temperature.

    Args:
        T: Temperature.

    Returns:
        A tuple of (T, xi).
    """
    sim = IsingSimulation(L, T)
    sim.equilibrate(EQUILIBRATION_STEPS)

    r, G_r = get_averaged_correlation(sim, STEPS, SAMPLE_INTERVAL)

    # Filter for valid range
    # r > 1 to avoid short-range lattice effects
    # r < L/4 to avoid finite size effects / periodic boundary artifacts
    # G_r > noise floor
    mask: np.ndarray = (r > 1) & (r < L // 4) & (G_r > 1e-4)

    if np.sum(mask) < 3:
        return T, np.nan

    r_fit: np.ndarray = r[mask]
    log_G: np.ndarray = np.log(G_r[mask])

    try:
        slope, intercept = np.polyfit(r_fit, log_G, 1)
        xi: float = -1.0 / slope
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        xi = np.nan

    return T, xi


def run_divergence_analysis() -> None:
    """Run parallel simulation to extract the critical exponent nu from xi(T) divergence."""
    print(f"Calculating correlation lengths for T > Tc (L={L})...")
    print(f"Approaching Tc={TC_THEORETICAL} with {len(TEMPERATURES)} points.")

    results: list[tuple[float, float]] = parallel_sweep(get_correlation_length, TEMPERATURES)

    temps_list, xis_list = zip(*results, strict=True)
    temps: np.ndarray = np.array(temps_list)
    xis: np.ndarray = np.array(xis_list)

    # Filter out failed fits
    valid: np.ndarray = ~np.isnan(xis)
    temps = temps[valid]
    xis = xis[valid]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Linear Plot: xi vs T
    ax1.plot(temps, xis, 'o-', markersize=6)
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel(r'Correlation Length $\xi$')
    ax1.set_title(r'Divergence of $\xi$ approaching $T_c$')
    ax1.grid(True)

    # 2. Log-Log Plot: xi vs (T - Tc)
    # Theory: xi ~ |T - Tc|^(-nu)
    reduced_T: np.ndarray = temps - TC_THEORETICAL

    # Fit power law (only possible when all reduced_T > 0 and xis > 0)
    nu: float | None = None
    try:
        log_t: np.ndarray = np.log(reduced_T)
        log_xi: np.ndarray = np.log(xis)
        slope, intercept = np.polyfit(log_t, log_xi, 1)
        nu = -slope
    except (np.linalg.LinAlgError, ValueError) as exc:
        print(f"Warning: power-law fit failed: {exc}")

    ax2.loglog(reduced_T, xis, 'o', label='Simulation Data')

    if nu is not None:
        # Plot fit line
        fit_x: np.ndarray = np.linspace(min(reduced_T), max(reduced_T), 100)
        fit_y: np.ndarray = np.exp(intercept) * fit_x ** (-nu)
        ax2.loglog(fit_x, fit_y, 'r--', label=f'Fit ($\\nu \\approx {nu:.2f}$)')

        # Plot theoretical slope (nu=1) for comparison
        theory_y: np.ndarray = fit_y[len(fit_y) // 2] * (
            fit_x / fit_x[len(fit_x) // 2]
        ) ** (-1)
        ax2.loglog(fit_x, theory_y, 'g:', label=r'Theory ($\nu=1$)')

    ax2.set_xlabel(r'$T - T_c$')
    ax2.set_ylabel(r'Correlation Length $\xi$')
    ax2.set_title(r'Critical Exponent $\nu$ Extraction')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    output_dir: str = ensure_results_dir('results/ising')
    save_plot('correlation_divergence.png', directory=output_dir)
    print(f"Analysis finished. Plot saved to {output_dir}")


if __name__ == "__main__":
    run_divergence_analysis()
