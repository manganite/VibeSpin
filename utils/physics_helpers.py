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


def radial_average_sk(spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the circularly averaged structure factor S(|k|).

    Bins S(k) by integer pixel radius from the DC centre of the shifted FFT,
    then averages within each annular bin.

    Args:
        spins: (N, N) or (N, N, 2) spin array.

    Returns:
        k_vals: Wavevector magnitudes in units of 2π/N (reciprocal lattice).
        S_radial: Mean S(k) value for each annular bin.
    """
    N = spins.shape[0]
    if spins.ndim == 3:  # Vector spins (XY/Clock)
        sx, sy = spins[..., 0], spins[..., 1]
        Sk_raw = np.abs(np.fft.fft2(sx.astype(float)))**2 + np.abs(np.fft.fft2(sy.astype(float)))**2
    else:  # Scalar spins (Ising)
        Sk_raw = np.abs(np.fft.fft2(spins.astype(float)))**2
        
    Sk = np.fft.fftshift(Sk_raw) / (N * N)

    cx = N // 2
    iy, ix = np.indices((N, N))
    r_int = np.sqrt((ix - cx)**2 + (iy - cx)**2).astype(int)

    # Average within each annular bin up to the Nyquist radius
    r_max = cx
    mask = r_int <= r_max
    tbin = np.bincount(r_int[mask].ravel(), Sk[mask].ravel())
    nbin = np.bincount(r_int[mask].ravel())
    S_radial = np.where(nbin > 0, tbin / nbin, 0.0)

    # Convert bin index → |k| in reciprocal lattice units (2π/N per bin)
    k_vals = np.arange(len(S_radial)) * (2.0 * np.pi / N)

    return k_vals, S_radial


def pair_correlation_x(spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the real-space spin-spin pair correlation G(r) along x.

    Uses the Wiener-Khinchin theorem: the autocorrelation of each row is the
    inverse FFT of the row's power spectrum.  Results are averaged over all
    rows (y positions) and normalised so G(0) = 1.

    Args:
        spins: (N, N) or (N, N, 2) spin array.

    Returns:
        r_vals: Lag distances r = 0 … N//2 in lattice units.
        G: Normalised pair correlation G(r) / G(0).
    """
    N = spins.shape[0]
    
    if spins.ndim == 3:  # Vector spins (XY/Clock)
        sx, sy = spins[..., 0].astype(float), spins[..., 1].astype(float)
        Fx = np.fft.rfft(sx, axis=1)
        Fy = np.fft.rfft(sy, axis=1)
        autocorr = np.fft.irfft(np.abs(Fx)**2 + np.abs(Fy)**2, n=N, axis=1)
    else:  # Scalar spins (Ising)
        s = spins.astype(float)
        F = np.fft.rfft(s, axis=1)
        autocorr = np.fft.irfft(np.abs(F)**2, n=N, axis=1)

    # Average over rows and normalise by N
    G_full = np.mean(autocorr, axis=0) / N
    r_half = N // 2 + 1
    r_vals = np.arange(r_half)
    G = G_full[:r_half]

    if G[0] != 0.0:
        G = G / G[0]

    return r_vals, G
