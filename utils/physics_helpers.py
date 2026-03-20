"""
Physics-related utility functions for calculating thermodynamic observables and correlations.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

from models.simulation_base import MonteCarloSimulation
from utils.exceptions import ZeroVarianceAutocorrelationError

DEFAULT_CONFIDENCE_LEVEL = 0.68
UNCERTAINTY_METHOD_BLOCKING = 'blocking'
UNCERTAINTY_METHOD_BOOTSTRAP = 'bootstrap'

UNCERTAINTY_FIELDS = (
    'value',
    'err',
    'ci_low',
    'ci_high',
    'tau_int',
    'n_eff',
    'samples',
)

UNCERTAINTY_METADATA_FIELDS = (
    'uncertainty_method',
    'confidence_level',
    'n_seeds',
    'bootstrap_resamples',
    'nan_or_undefined_count',
)


def _validate_confidence(*, confidence: float) -> None:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f'confidence must satisfy 0 < confidence < 1, got {confidence}')


def _as_1d_float_array(*, time_series: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(time_series, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be 1-D, got shape {arr.shape}')
    if arr.size < 2:
        raise ValueError(f'{name} must contain at least 2 elements, got {arr.size}')
    return arr


def _z_multiplier(*, confidence: float) -> float:
    """Return Gaussian z-score for two-sided confidence interval."""
    return float(norm.ppf(0.5 + 0.5 * confidence))


def _blocking_block_sizes(*, n: int, min_block_size: int, max_block_size: int) -> list[int]:
    """Generate increasing block sizes for blocking analysis."""
    sizes: list[int] = []
    block = max(2, min_block_size)
    upper = min(max_block_size, n // 2)
    while block <= upper:
        n_blocks = n // block
        if n_blocks >= 2:
            sizes.append(block)
        block *= 2
    return sizes


def _select_blocking_plateau(*, standard_errors: np.ndarray) -> int:
    """Pick the first stable blocking plateau index."""
    if standard_errors.size == 1:
        return 0
    rel_tol = 0.05
    for idx in range(1, standard_errors.size):
        prev = float(standard_errors[idx - 1])
        cur = float(standard_errors[idx])
        if prev == 0.0 and cur == 0.0:
            return idx
        denom = max(abs(prev), 1e-16)
        if abs(cur - prev) / denom <= rel_tol:
            return idx
    return int(standard_errors.size - 1)


def _safe_n_eff_from_tau(*, n_samples: int, tau_int: float) -> float:
    """Convert integrated autocorrelation time to effective sample size."""
    if not np.isfinite(tau_int) or tau_int <= 0.0:
        return float('nan')
    return float(min(n_samples, max(1.0, n_samples / (2.0 * tau_int))))


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


def estimate_effective_sample_size(
    *,
    time_series: np.ndarray,
    tau_int: float | None = None,
) -> float:
    """Estimate effective sample size from integrated autocorrelation time."""
    arr = _as_1d_float_array(time_series=time_series, name='time_series')
    if tau_int is None:
        _, tau_est = calculate_autocorr(time_series=arr)
    else:
        if tau_int <= 0.0:
            raise ValueError(f'tau_int must be positive when provided, got {tau_int}')
        tau_est = float(tau_int)
    return _safe_n_eff_from_tau(n_samples=arr.size, tau_int=tau_est)


def blocking_error(
    *,
    time_series: np.ndarray,
    min_block_size: int = 2,
    max_block_size: int | None = None,
) -> dict[str, float]:
    """Estimate standard error using standard blocking/binning analysis."""
    arr = _as_1d_float_array(time_series=time_series, name='time_series')
    if min_block_size < 2:
        raise ValueError(f'min_block_size must be >= 2, got {min_block_size}')
    if max_block_size is None:
        max_block_size = arr.size // 2
    if max_block_size < min_block_size:
        raise ValueError(
            f'max_block_size must be >= min_block_size, got {max_block_size} < {min_block_size}'
        )

    std = float(np.std(arr, ddof=1))
    naive_stderr = std / np.sqrt(arr.size)
    if std == 0.0:
        return {
            'stderr': 0.0,
            'stderr_naive': 0.0,
            'block_size': float(min_block_size),
            'n_blocks': float(arr.size // min_block_size),
            'tau_int_from_blocking': float('nan'),
        }

    block_sizes = _blocking_block_sizes(
        n=arr.size,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
    )
    if not block_sizes:
        return {
            'stderr': float(naive_stderr),
            'stderr_naive': float(naive_stderr),
            'block_size': 1.0,
            'n_blocks': float(arr.size),
            'tau_int_from_blocking': 0.5,
        }

    stderr_values: list[float] = []
    n_blocks_values: list[float] = []
    for block in block_sizes:
        n_blocks = arr.size // block
        trimmed = arr[:n_blocks * block]
        block_means = trimmed.reshape(n_blocks, block).mean(axis=1)
        stderr = float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))
        stderr_values.append(stderr)
        n_blocks_values.append(float(n_blocks))

    stderr_arr = np.asarray(stderr_values, dtype=np.float64)
    selected_idx = _select_blocking_plateau(standard_errors=stderr_arr)
    selected_stderr = float(stderr_arr[selected_idx])

    tau_block = float('nan')
    if naive_stderr > 0.0:
        tau_block = 0.5 * (selected_stderr / naive_stderr) ** 2

    return {
        'stderr': selected_stderr,
        'stderr_naive': float(naive_stderr),
        'block_size': float(block_sizes[selected_idx]),
        'n_blocks': float(n_blocks_values[selected_idx]),
        'tau_int_from_blocking': tau_block,
    }


def summarize_primary_observable(
    *,
    time_series: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, float]:
    """Summarize a primary observable with blocking-based uncertainty."""
    _validate_confidence(confidence=confidence)
    arr = _as_1d_float_array(time_series=time_series, name='time_series')

    value = float(np.mean(arr))
    block = blocking_error(time_series=arr)
    err = float(block['stderr'])
    z = _z_multiplier(confidence=confidence)

    try:
        _, tau_int = calculate_autocorr(time_series=arr)
    except ZeroVarianceAutocorrelationError:
        tau_int = float('nan')

    n_eff = _safe_n_eff_from_tau(n_samples=arr.size, tau_int=tau_int)

    return {
        'value': value,
        'err': err,
        'ci_low': float(value - z * err),
        'ci_high': float(value + z * err),
        'tau_int': float(tau_int),
        'n_eff': n_eff,
        'samples': float(arr.size),
    }


def _derived_point_estimate(
    *,
    series: np.ndarray,
    temperature: float,
    L: int,
    observable: str,
) -> float:
    """Compute derived thermodynamic point estimates from a time series."""
    N = float(L * L)
    variance = float(np.var(series))
    if observable == 'chi':
        return float(N * variance / temperature)
    if observable == 'cv':
        return float(N * variance / (temperature**2))
    raise ValueError(f"observable must be one of 'chi' or 'cv', got {observable!r}")


def summarize_derived_observable(
    *,
    magnetization_series: np.ndarray | None = None,
    energy_series: np.ndarray | None = None,
    temperature: float,
    L: int,
    observable: str,
    method: str = UNCERTAINTY_METHOD_BLOCKING,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_resamples: int = 0,
) -> dict[str, float]:
    """Summarize derived observables (chi, Cv) with uncertainty estimates."""
    _validate_confidence(confidence=confidence)
    if temperature <= 0.0:
        raise ValueError(f'temperature must be positive, got {temperature}')
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError(f'L must be a positive integer, got {L!r}')
    if method not in {UNCERTAINTY_METHOD_BLOCKING, UNCERTAINTY_METHOD_BOOTSTRAP}:
        raise ValueError(f'unsupported method {method!r}')

    if observable == 'chi':
        if magnetization_series is None:
            raise ValueError("magnetization_series is required for observable='chi'")
        base = _as_1d_float_array(time_series=magnetization_series, name='magnetization_series')
    elif observable == 'cv':
        if energy_series is None:
            raise ValueError("energy_series is required for observable='cv'")
        base = _as_1d_float_array(time_series=energy_series, name='energy_series')
    else:
        raise ValueError(f"observable must be one of 'chi' or 'cv', got {observable!r}")

    value = _derived_point_estimate(
        series=base,
        temperature=temperature,
        L=L,
        observable=observable,
    )
    if np.std(base, ddof=1) == 0.0:
        return {
            'value': value,
            'err': 0.0,
            'ci_low': value,
            'ci_high': value,
            'tau_int': float('nan'),
            'n_eff': float('nan'),
            'samples': float(base.size),
        }

    block = blocking_error(time_series=base)
    block_size = int(block['block_size'])
    n_blocks = base.size // block_size
    trimmed = base[:n_blocks * block_size]
    block_reshaped = trimmed.reshape(n_blocks, block_size)

    per_block = np.array(
        [
            _derived_point_estimate(
                series=block_reshaped[idx],
                temperature=temperature,
                L=L,
                observable=observable,
            )
            for idx in range(n_blocks)
        ],
        dtype=np.float64,
    )

    if n_blocks < 2:
        err = float('nan')
        ci_low = float('nan')
        ci_high = float('nan')
    elif method == UNCERTAINTY_METHOD_BLOCKING:
        err = float(np.std(per_block, ddof=1) / np.sqrt(n_blocks))
        z = _z_multiplier(confidence=confidence)
        ci_low = float(value - z * err)
        ci_high = float(value + z * err)
    else:
        if bootstrap_resamples <= 0:
            raise ValueError('bootstrap_resamples must be > 0 when method is bootstrap')
        rng = np.random.default_rng(0)
        boot = np.empty(bootstrap_resamples, dtype=np.float64)
        for i in range(bootstrap_resamples):
            sample = rng.choice(per_block, size=n_blocks, replace=True)
            boot[i] = float(np.mean(sample))
        alpha = 0.5 * (1.0 - confidence)
        ci_low = float(np.quantile(boot, alpha))
        ci_high = float(np.quantile(boot, 1.0 - alpha))
        err = float(np.std(boot, ddof=1))

    try:
        _, tau_int = calculate_autocorr(time_series=base)
    except ZeroVarianceAutocorrelationError:
        tau_int = float('nan')

    n_eff = _safe_n_eff_from_tau(n_samples=base.size, tau_int=tau_int)
    return {
        'value': value,
        'err': err,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'tau_int': float(tau_int),
        'n_eff': n_eff,
        'samples': float(base.size),
    }


def summarize_replicate_samples(
    *,
    samples: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, np.ndarray | float]:
    """Summarize replicate samples using NaN-safe median and percentile bands."""
    _validate_confidence(confidence=confidence)
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        raise ValueError('samples must be non-empty')

    alpha = 0.5 * (1.0 - confidence)
    if arr.ndim == 1:
        value = float(np.nanmedian(arr))
        ci_low = float(np.nanquantile(arr, alpha))
        ci_high = float(np.nanquantile(arr, 1.0 - alpha))
        return {
            'value': value,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'samples': float(arr.size),
            'nan_or_undefined_count': float(np.isnan(arr).sum()),
        }

    value_arr = np.nanmedian(arr, axis=1)
    ci_low_arr = np.nanquantile(arr, alpha, axis=1)
    ci_high_arr = np.nanquantile(arr, 1.0 - alpha, axis=1)
    return {
        'value': value_arr,
        'ci_low': ci_low_arr,
        'ci_high': ci_high_arr,
        'samples': float(arr.shape[1]),
        'nan_or_undefined_count': float(np.isnan(arr).sum()),
    }


def summarize_seed_ensemble(
    *,
    values: np.ndarray,
    within_seed_errors: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, float]:
    """Aggregate per-seed estimates into one mean with hierarchical uncertainty.

    The total variance combines between-seed and within-seed components:
    Var(mean) = Var_between / n_seeds + mean(Var_within) / n_seeds.
    """
    _validate_confidence(confidence=confidence)
    values_arr = np.asarray(values, dtype=np.float64)
    errs_arr = np.asarray(within_seed_errors, dtype=np.float64)
    if values_arr.ndim != 1:
        raise ValueError(f'values must be 1-D, got shape {values_arr.shape}')
    if errs_arr.ndim != 1:
        raise ValueError(f'within_seed_errors must be 1-D, got shape {errs_arr.shape}')
    if values_arr.size != errs_arr.size:
        raise ValueError(
            'values and within_seed_errors must have matching lengths, '
            f'got {values_arr.size} and {errs_arr.size}'
        )
    if values_arr.size == 0:
        raise ValueError('values must be non-empty')

    finite_values = np.isfinite(values_arr)
    if not np.any(finite_values):
        return {
            'value': float('nan'),
            'err': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'between_seed_component': float('nan'),
            'within_seed_component': float('nan'),
            'samples': 0.0,
            'nan_or_undefined_count': float(values_arr.size),
        }

    v = values_arr[finite_values]
    e = errs_arr[finite_values]
    n = v.size
    value = float(np.mean(v))

    between_component = 0.0
    if n > 1:
        between_component = float(np.var(v, ddof=1) / n)

    finite_errs = np.isfinite(e)
    within_component = 0.0
    if np.any(finite_errs):
        within_component = float(np.mean(e[finite_errs] ** 2) / n)

    total_err = float(np.sqrt(max(0.0, between_component + within_component)))
    z = _z_multiplier(confidence=confidence)

    return {
        'value': value,
        'err': total_err,
        'ci_low': float(value - z * total_err),
        'ci_high': float(value + z * total_err),
        'between_seed_component': float(np.sqrt(max(0.0, between_component))),
        'within_seed_component': float(np.sqrt(max(0.0, within_component))),
        'samples': float(n),
        'nan_or_undefined_count': float(values_arr.size - n),
    }


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


def _valid_prefix(x: np.ndarray) -> np.ndarray:
    """Return the non-NaN prefix from a padded trajectory."""
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.empty(0, dtype=float)
    end = int(np.where(valid)[0][-1]) + 1
    return x[:end]


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, min(window, len(x)))
    return np.convolve(x, np.ones(window) / window, mode='valid')


def estimate_relaxation_time_two_start(
    trace_random: np.ndarray,
    trace_ordered: np.ndarray,
    *,
    k: float = 1.0,
    smooth_window: int = 30,
    dwell_window: int = 30,
    min_fraction_inside: float = 0.85,
) -> int:
    """Estimate thermalization time from convergence of random- and ordered-start traces.

    Compares a trajectory started from a random spin configuration against one
    started from a fully ordered configuration.  Returns the first step at which
    both smoothed traces enter and sustain a common equilibrium band.

    Parameters
    ----------
        trace_random: 1-D magnetization trace from a random initial state.
        trace_ordered: 1-D magnetization trace from an ordered initial state.
        k: Half-width of the convergence band in units of the tail standard
            deviation.  Larger values are more permissive.
        smooth_window: Window length (steps) for the moving-average smoother.
        dwell_window: Window length (steps) for the sustained-convergence test.
        min_fraction_inside: Fraction of steps within ``dwell_window`` that must
            lie inside the band to declare convergence.

    Returns
    -------
        Estimated relaxation time in MC steps.  Returns the full trace length
        if no convergence is detected, or 0 for very short traces.
    """
    r = np.abs(_valid_prefix(np.asarray(trace_random, dtype=float)))
    o = np.abs(_valid_prefix(np.asarray(trace_ordered, dtype=float)))

    n = min(len(r), len(o))
    if n < 8:
        return 0
    r = r[:n]
    o = o[:n]

    r_sm = _moving_average(r, smooth_window)
    o_sm = _moving_average(o, smooth_window)
    m = min(len(r_sm), len(o_sm))
    r_sm = r_sm[:m]
    o_sm = o_sm[:m]

    half = m // 2
    tail_combined = np.concatenate([r_sm[half:], o_sm[half:]])
    mean_eq = tail_combined.mean()
    sigma_eq = max(tail_combined.std(), 1e-12)
    band = k * sigma_eq

    both_inside = (
        (np.abs(r_sm - mean_eq) <= band)
        & (np.abs(o_sm - mean_eq) <= band)
        & (np.abs(r_sm - o_sm) <= band)
    ).astype(float)

    dwell_window = max(3, min(dwell_window, len(both_inside)))
    sustained_fraction = (
        np.convolve(both_inside, np.ones(dwell_window), mode='valid') / dwell_window
    )

    hits = np.where(sustained_fraction >= min_fraction_inside)[0]
    return int(hits[0]) if hits.size else n
