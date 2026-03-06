"""
Physics-related utility functions for calculating thermodynamic observables and correlations.
"""

import numpy as np

from models.simulation_base import MonteCarloSimulation


def calculate_thermodynamics(
    mags: np.ndarray, engs: np.ndarray, T: float, L: int
) -> tuple[float, float, float, float]:
    """
    Calculate average magnetization, energy, susceptibility, and specific heat.

    Args:
        mags: Array of magnetization measurements.
        engs: Array of energy measurements.
        T: Temperature.
        L: Linear lattice size.

    Returns:
        A tuple of (avg_mag, avg_eng, susceptibility, specific_heat).

    Raises:
        ValueError: If ``T`` is not positive or ``L`` is not a positive integer.
    """
    if T <= 0.0:
        raise ValueError(f"T must be positive (T > 0), got {T}")
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError(f"L must be a positive integer, got {L!r}")
    avg_mag = float(np.mean(mags))
    avg_eng = float(np.mean(engs))
    N = L * L

    # Susceptibility: chi = N * Var(M) / T
    susceptibility = float(N * np.var(mags) / T)

    # Specific Heat: Cv = N * Var(E) / T^2
    specific_heat = float(N * np.var(engs) / (T**2))

    return avg_mag, avg_eng, susceptibility, specific_heat


def get_averaged_correlation(
    sim: MonteCarloSimulation, total_steps: int, sample_interval: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run simulation and average the correlation function over multiple configurations.

    Args:
        sim: An instance of MonteCarloSimulation.
        total_steps: Total number of MC steps to run.
        sample_interval: Interval between correlation samples. Must be ≥ 1.

    Returns:
        r: Radial distances.
        G_r_avg: Averaged correlation values.

    Raises:
        ValueError: If ``sample_interval`` is less than 1 or ``total_steps`` is negative.
    """
    if sample_interval < 1:
        raise ValueError(f"sample_interval must be >= 1, got {sample_interval}")
    if total_steps < 0:
        raise ValueError(f"total_steps must be non-negative, got {total_steps}")
    G_r_avg: np.ndarray | None = None
    count = 0
    r: np.ndarray = np.array([])

    for i in range(total_steps):
        sim.step()
        if i % sample_interval == 0:
            r, G_r = sim._calculate_correlation_function()
            if G_r_avg is None:
                G_r_avg = np.zeros_like(G_r)
            G_r_avg += G_r
            count += 1

    if G_r_avg is not None and count > 0:
        G_r_avg /= count
    else:
        # Fallback if no samples were taken
        _, G_r_dummy = sim._calculate_correlation_function()
        G_r_avg = np.zeros_like(G_r_dummy)
        r = _

    return r, G_r_avg
