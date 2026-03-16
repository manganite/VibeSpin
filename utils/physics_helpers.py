"""
Physics-related utility functions for calculating thermodynamic observables and correlations.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from models.simulation_base import MonteCarloSimulation
from utils.exceptions import ZeroVarianceAutocorrelationError


def calculate_thermodynamics(
    *, mags: np.ndarray, engs: np.ndarray, T: float, L: int
) -> tuple[float, float, float, float]:
    """
    Calculate average magnetization, energy, susceptibility, and specific heat.

    Parameters
    ----------
        mags: Array of magnetization measurements.
        engs: Array of energy measurements.
        T: Temperature.
        L: Linear lattice size.

    Returns
    -------
        A tuple of (avg_mag, avg_eng, susceptibility, specific_heat).

    Raises
    ------
        ValueError: If ``T`` is not positive or ``L`` is not a positive integer.
    """
    if T <= 0.0:
        raise ValueError(f'T must be positive (T > 0), got {T}')
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError(f'L must be a positive integer, got {L!r}')
    avg_mag = float(np.mean(mags))
    avg_eng = float(np.mean(engs))
    N = L * L

    # Susceptibility: chi = N * Var(M) / T
    susceptibility = float(N * np.var(mags) / T)

    # Specific Heat: Cv = N * Var(E) / T^2
    specific_heat = float(N * np.var(engs) / (T**2))

    return avg_mag, avg_eng, susceptibility, specific_heat


def calculate_entropy(
    *,
    temperatures: np.ndarray,
    specific_heat: np.ndarray,
    s_ref: float = 0.0,
) -> np.ndarray:
    """
    Compute entropy by integrating specific heat over a temperature sweep.

    Integrates C(T)/T from high temperature downward so that
    S(T_max) = ``s_ref`` and S(T) = s_ref - integral from T to T_max of C/T dT'.
    This avoids the C/T divergence near T = 0 and anchors the result at the
    high-temperature limit where the entropy is analytically known.

    Parameters
    ----------
        temperatures: 1-D array of temperature values. Need not be sorted.
        specific_heat: Specific heat C_v at each temperature (same length).
        s_ref: Reference entropy at the highest temperature. Pass ``0.0``
            for relative entropy or ``np.log(q)`` (per site, in units of
            k_B) for absolute entropy of a q-state model.

    Returns
    -------
        1-D array of entropy values, one per temperature, in the original
        unsorted order of ``temperatures``.

    Raises
    ------
        ValueError: If arrays differ in length, contain fewer than 2 points,
            or include non-positive temperatures.
    """
    temperatures = np.asarray(temperatures, dtype=np.float64)
    specific_heat = np.asarray(specific_heat, dtype=np.float64)

    if temperatures.shape != specific_heat.shape:
        raise ValueError(
            f'temperatures and specific_heat must have the same shape, '
            f'got {temperatures.shape} and {specific_heat.shape}'
        )
    if temperatures.size < 2:
        raise ValueError('Need at least 2 temperature points for integration')
    if np.any(temperatures <= 0.0):
        raise ValueError('All temperatures must be positive')

    # Sort ascending; remember original order to restore later.
    sort_idx = np.argsort(temperatures)
    t_sorted = temperatures[sort_idx]
    cv_sorted = specific_heat[sort_idx]

    # Integrand: C(T) / T
    integrand = cv_sorted / t_sorted

    # Cumulative integral from T_1 (lowest) to each T_i.
    partial = cumulative_trapezoid(integrand, t_sorted)
    # Prepend 0 for the first (lowest-T) point.
    partial = np.concatenate(([0.0], partial))

    # S(T) = s_ref - (total_integral - partial_integral_up_to_T)
    total = partial[-1]
    entropy_sorted = s_ref - (total - partial)

    # Restore original ordering.
    entropy: np.ndarray = np.empty_like(entropy_sorted)
    entropy[sort_idx] = entropy_sorted
    return entropy


def calculate_autocorr(
    *,
    time_series: np.ndarray,
    max_lag: int | None = None,
    window_constant: float = 6.0,
) -> tuple[np.ndarray, float]:
    """
    Compute the normalized autocorrelation function and integrated autocorrelation time.

    Uses an FFT-based estimator for C(t): zero-pads to 2N, computes the power
    spectrum, IFFTs, and normalizes by variance and the number of pairs at each
    lag.  Applies the Madras-Sokal automatic windowing rule to stop integration
    at the lag W where W = ``window_constant`` * tau_int, preventing the estimator
    from accumulating noise at large lags.

    Parameters
    ----------
        time_series: 1-D array of sequential measurements (e.g. magnetization).
        max_lag: Maximum lag to compute.  Defaults to ``len(time_series) // 2``.
        window_constant: Multiplier for the Madras-Sokal window (default 6.0).

    Returns
    -------
        A tuple ``(C_t, tau_int)`` where ``C_t`` is the normalized autocorrelation
        array truncated at the window endpoint, and ``tau_int`` is the integrated
        autocorrelation time (scalar).

    Raises
    ------
        ValueError: If ``time_series`` has fewer than 3 elements.
        ZeroVarianceAutocorrelationError: If ``time_series`` has zero variance.
    """
    x = np.asarray(time_series, dtype=np.float64)
    N = len(x)
    if N < 3:
        raise ValueError(f'time_series must have at least 3 elements, got {N}')
    variance = float(np.var(x))
    if variance == 0.0:
        raise ZeroVarianceAutocorrelationError(
            'time_series has zero variance; autocorrelation is undefined'
        )

    if max_lag is None:
        max_lag = N // 2

    # FFT-based unnormalized autocorrelation via power spectrum
    x_centered = x - np.mean(x)
    n_fft = 2 * N
    fx = np.fft.rfft(x_centered, n=n_fft)
    power = fx * np.conj(fx)
    acf_raw: np.ndarray = np.fft.irfft(power, n=n_fft)[:N].real

    # Normalize: divide by variance and by (N - t) pairs at each lag t
    lags = np.arange(N, dtype=np.float64)
    norm = variance * (N - lags)
    C_t_full: np.ndarray = acf_raw / norm

    # Madras-Sokal automatic window: integrate until window W = window_constant * tau_int
    tau_int = 0.5
    window_end = max_lag
    for t in range(1, max_lag + 1):
        tau_int += float(C_t_full[t])
        if t >= window_constant * tau_int:
            window_end = t
            break

    C_t: np.ndarray = C_t_full[:window_end + 1]
    return C_t, tau_int


def get_averaged_correlation(
    *, sim: MonteCarloSimulation, total_steps: int, sample_interval: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run simulation and average the correlation function over multiple configurations.

    Parameters
    ----------
        sim: An instance of MonteCarloSimulation.
        total_steps: Total number of MC steps to run.
        sample_interval: Interval between correlation samples. Must be ≥ 1.

    Returns
    -------
        r: Radial distances.
        G_r_avg: Averaged correlation values.

    Raises
    ------
        ValueError: If ``sample_interval`` is less than 1 or ``total_steps`` is negative.
    """
    if sample_interval < 1:
        raise ValueError(f'sample_interval must be >= 1, got {sample_interval}')
    if total_steps < 0:
        raise ValueError(f'total_steps must be non-negative, got {total_steps}')
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


def radial_average_sk(*, spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the circularly averaged structure factor :math:`S(|k|)`.

    Bins :math:`S(k)` by integer pixel radius from the DC centre of the shifted FFT,
    then averages within each annular bin.

    Parameters
    ----------
        spins: (N, N) or (N, N, 2) spin array.

    Returns
    -------
        k_vals: Wavevector magnitudes in units of 2π/N (reciprocal lattice).
        S_radial: Mean S(k) value for each annular bin.
    """
    N = spins.shape[0]
    if spins.ndim == 3:  # Vector spins (XY/Clock)
        sx, sy = spins[..., 0], spins[..., 1]
        Sk_raw = (
            np.abs(np.fft.fft2(sx.astype(float))) ** 2 + np.abs(np.fft.fft2(sy.astype(float))) ** 2
        )
    else:  # Scalar spins (Ising)
        Sk_raw = np.abs(np.fft.fft2(spins.astype(float))) ** 2

    Sk = np.fft.fftshift(Sk_raw) / (N * N)

    cx = N // 2
    iy, ix = np.indices((N, N))
    r_int = np.sqrt((ix - cx) ** 2 + (iy - cx) ** 2).astype(int)

    # Average within each annular bin up to the Nyquist radius
    r_max = cx
    mask = r_int <= r_max
    tbin = np.bincount(r_int[mask].ravel(), Sk[mask].ravel())
    nbin = np.bincount(r_int[mask].ravel())
    S_radial = np.where(nbin > 0, tbin / nbin, 0.0)

    # Convert bin index → |k| in reciprocal lattice units (2π/N per bin)
    k_vals = np.arange(len(S_radial)) * (2.0 * np.pi / N)

    return k_vals, S_radial


def pair_correlation_x(*, spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the real-space spin-spin pair correlation G(r) along x.

    Uses the Wiener-Khinchin theorem: the autocorrelation of each row is the
    inverse FFT of the row's power spectrum.  Results are averaged over all
    rows (y positions) and normalised so G(0) = 1.

    Parameters
    ----------
        spins: (N, N) or (N, N, 2) spin array.

    Returns
    -------
        r_vals: Lag distances r = 0 … N//2 in lattice units.
        G: Normalised pair correlation G(r) / G(0).
    """
    N = spins.shape[0]

    if spins.ndim == 3:  # Vector spins (XY/Clock)
        sx, sy = spins[..., 0].astype(float), spins[..., 1].astype(float)
        Fx = np.fft.rfft(sx, axis=1)
        Fy = np.fft.rfft(sy, axis=1)
        autocorr = np.fft.irfft(np.abs(Fx) ** 2 + np.abs(Fy) ** 2, n=N, axis=1)
    else:  # Scalar spins (Ising)
        s = spins.astype(float)
        F = np.fft.rfft(s, axis=1)
        autocorr = np.fft.irfft(np.abs(F) ** 2, n=N, axis=1)

    # Average over rows and normalise by N
    G_full = np.mean(autocorr, axis=0) / N
    r_half = N // 2 + 1
    r_vals = np.arange(r_half)
    G = G_full[:r_half]

    if G[0] != 0.0:
        G = G / G[0]

    return r_vals, G


def compute_kinetics_metrics(*, sim: MonteCarloSimulation) -> dict[str, float]:
    """
    Calculate common kinetics metrics (R_sk, xi) for a simulation state.

    Parameters
    ----------
        sim: MonteCarloSimulation instance.

    Returns
    -------
        Dictionary containing 'R_sk' and 'xi'.
    """
    if sim.spins is None:
        return {'R_sk': 0.0, 'xi': 0.0}

    # 1. R_sk from structure factor first moment
    kvals, S_radial = radial_average_sk(spins=sim.spins)
    S_k = S_radial[1:]
    K_k = kvals[1:]
    denom = float(np.sum(K_k * S_k))
    R_sk = (2.0 * np.pi * float(np.sum(S_k) / denom)) if denom != 0 else 0.0

    # 2. xi from G(r) 1/e decay
    r_vals, G = pair_correlation_x(spins=sim.spins)
    inv_e = 1.0 / np.e
    below = np.where(G < inv_e)[0]
    if len(below) == 0:
        xi = float(r_vals[-1])
    elif below[0] == 0:
        xi = float(r_vals[0])
    else:
        idx = below[0]
        r0, r1 = float(r_vals[idx - 1]), float(r_vals[idx])
        g0, g1 = float(G[idx - 1]), float(G[idx])
        xi = r0 + (inv_e - g0) * (r1 - r0) / (g1 - g0)

    return {'R_sk': R_sk, 'xi': xi}


def power_fit(
    *, t_arr: np.ndarray, y_arr: np.ndarray, mask: np.ndarray
) -> tuple[float, float] | tuple[None, None]:
    """
    Fit a power law y = prefactor * t^exponent via log-log linear regression.

    Parameters
    ----------
        t_arr: Independent variable array (e.g., time or length scale).
        y_arr: Dependent variable array; only positive values are used in the fit.
        mask: Boolean mask selecting the subset of points to include.

    Returns
    -------
        exponent: Fitted power-law exponent, or None if fewer than 3 valid points.
        prefactor: Fitted prefactor, or None if fewer than 3 valid points.
    """
    valid = mask & (y_arr > 0)
    if valid.sum() < 3:
        return None, None
    coeffs = np.polyfit(np.log(t_arr[valid]), np.log(y_arr[valid]), 1)
    return float(coeffs[0]), float(np.exp(coeffs[1]))
