"""
Statistical estimators and uncertainty schemas for Monte Carlo time series:
blocking analysis, autocorrelation, replicate and seed-ensemble aggregation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from utils.exceptions import ZeroVarianceAutocorrelationError
from utils.observables import calculate_entropy, derived_thermo_estimate

DEFAULT_CONFIDENCE_LEVEL = 0.68
UNCERTAINTY_METHOD_BLOCKING = 'blocking'
UNCERTAINTY_METHOD_BOOTSTRAP = 'bootstrap'
UNCERTAINTY_METHOD_REPLICATE = 'replicate_percentile'

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


def _entropy_result(
    *,
    value: np.ndarray,
    err: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    samples: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the standard entropy summary dict (one construction site)."""
    return {
        'value': value,
        'err': err,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'samples': samples,
    }



def _propagate_entropy_uncertainty_from_cv_errors(
    *,
    temperatures: np.ndarray,
    specific_heat_err: np.ndarray,
) -> np.ndarray:
    """Propagate pointwise specific-heat errors into entropy uncertainty.

    Uses independent-error propagation through the trapezoidal integration used
    in :func:`calculate_entropy` for S(T).
    """
    t = np.asarray(temperatures, dtype=np.float64)
    cv_err = np.asarray(specific_heat_err, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f'temperatures must be 1-D, got shape {t.shape}')
    if cv_err.shape != t.shape:
        raise ValueError(
            'specific_heat_err must match temperatures shape, '
            f'got {cv_err.shape} and {t.shape}'
        )

    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    cv_err_sorted = cv_err[sort_idx]
    sigma_f = cv_err_sorted / t_sorted
    dt = np.diff(t_sorted)
    seg_var = 0.25 * (dt * dt) * (sigma_f[:-1] * sigma_f[:-1] + sigma_f[1:] * sigma_f[1:])

    # Tail sums of the segment variances: var_sorted[i] = sum(seg_var[i:]).
    # The reversed cumulative sum accumulates from the high-temperature anchor
    # downward, exactly like the previous explicit loop. A non-finite segment
    # poisons its own point and every lower-temperature point (whose tail sum
    # would include it), while higher-temperature points stay exact because
    # their tail sums never touch it.
    n_t = t_sorted.size
    var_sorted = np.zeros(n_t, dtype=np.float64)
    if n_t > 1:
        var_sorted[:-1] = np.cumsum(seg_var[::-1])[::-1]
        nonfinite = np.where(~np.isfinite(seg_var))[0]
        if nonfinite.size:
            var_sorted[: int(nonfinite.max()) + 1] = np.nan

    err_sorted = np.sqrt(var_sorted)
    err = np.empty_like(err_sorted)
    err[sort_idx] = err_sorted
    return err


def summarize_entropy_observable(
    *,
    temperatures: np.ndarray,
    specific_heat_samples: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    s_ref: float = 0.0,
    specific_heat_err: np.ndarray | None = None,
    method: str = UNCERTAINTY_METHOD_BOOTSTRAP,
    bootstrap_resamples: int = 400,
    rng_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Summarize entropy from replicate specific-heat curves.

    Each replicate column is converted to an entropy curve via
    :func:`calculate_entropy`, then uncertainty is computed either via
    replicate spread (blocking-like) or bootstrap over replicate curves.
    For a single finite replicate, uncertainty is reported as NaN unless
    ``specific_heat_err`` is provided, in which case entropy uncertainty is
    propagated from pointwise specific-heat errors.
    """
    _validate_confidence(confidence=confidence)
    if method not in {UNCERTAINTY_METHOD_BLOCKING, UNCERTAINTY_METHOD_BOOTSTRAP}:
        raise ValueError(f'unsupported method {method!r}')
    if method == UNCERTAINTY_METHOD_BOOTSTRAP and bootstrap_resamples <= 0:
        raise ValueError('bootstrap_resamples must be > 0 when method is bootstrap')
    temperatures_arr = np.asarray(temperatures, dtype=np.float64)
    samples_arr = np.asarray(specific_heat_samples, dtype=np.float64)

    if temperatures_arr.ndim != 1:
        raise ValueError(f'temperatures must be 1-D, got shape {temperatures_arr.shape}')
    if samples_arr.ndim != 2:
        raise ValueError(
            f'specific_heat_samples must be 2-D, got shape {samples_arr.shape}'
        )
    if samples_arr.shape[0] != temperatures_arr.size:
        raise ValueError(
            'specific_heat_samples first dimension must match temperatures size, '
            f'got {samples_arr.shape[0]} and {temperatures_arr.size}'
        )
    if samples_arr.shape[1] < 1:
        raise ValueError('specific_heat_samples must contain at least one replicate column')

    n_t, n_rep = samples_arr.shape
    entropy_samples = np.empty((n_t, n_rep), dtype=np.float64)
    for j in range(n_rep):
        entropy_samples[:, j] = calculate_entropy(
            temperatures=temperatures_arr,
            specific_heat=samples_arr[:, j],
            s_ref=s_ref,
        )

    value = np.empty(n_t, dtype=np.float64)
    err = np.empty(n_t, dtype=np.float64)
    ci_low = np.empty(n_t, dtype=np.float64)
    ci_high = np.empty(n_t, dtype=np.float64)
    alpha = 0.5 * (1.0 - confidence)
    z = _z_multiplier(confidence=confidence)

    if n_rep < 2 and specific_heat_err is not None:
        value[:] = np.nanmean(entropy_samples, axis=1)
        err[:] = _propagate_entropy_uncertainty_from_cv_errors(
            temperatures=temperatures_arr,
            specific_heat_err=np.asarray(specific_heat_err, dtype=np.float64),
        )
        ci_low[:] = value - z * err
        ci_high[:] = value + z * err
        return _entropy_result(
            value=value, err=err, ci_low=ci_low, ci_high=ci_high,
            samples=entropy_samples,
        )

    if method == UNCERTAINTY_METHOD_BOOTSTRAP:
        if n_rep < 2:
            value[:] = np.nanmean(entropy_samples, axis=1)
            err[:] = np.nan
            ci_low[:] = np.nan
            ci_high[:] = np.nan
            return _entropy_result(
                value=value, err=err, ci_low=ci_low, ci_high=ci_high,
                samples=entropy_samples,
            )
        rng = np.random.default_rng(rng_seed)
        boot_curves = np.empty((n_t, bootstrap_resamples), dtype=np.float64)
        for b in range(bootstrap_resamples):
            idx = rng.choice(n_rep, size=n_rep, replace=True)
            boot_curves[:, b] = np.nanmean(entropy_samples[:, idx], axis=1)

        value[:] = np.nanmean(entropy_samples, axis=1)
        err[:] = np.nanstd(boot_curves, axis=1, ddof=1)
        ci_low[:] = np.nanquantile(boot_curves, alpha, axis=1)
        ci_high[:] = np.nanquantile(boot_curves, 1.0 - alpha, axis=1)
        return _entropy_result(
            value=value, err=err, ci_low=ci_low, ci_high=ci_high,
            samples=entropy_samples,
        )

    for i in range(n_t):
        row = entropy_samples[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            value[i] = np.nan
            err[i] = np.nan
            ci_low[i] = np.nan
            ci_high[i] = np.nan
            continue

        mean_i = float(np.mean(finite))
        value[i] = mean_i
        if finite.size == 1:
            err[i] = np.nan
            ci_low[i] = np.nan
            ci_high[i] = np.nan
            continue

        stderr_i = float(np.std(finite, ddof=1) / np.sqrt(finite.size))
        err[i] = stderr_i
        ci_low[i] = float(mean_i - z * stderr_i)
        ci_high[i] = float(mean_i + z * stderr_i)

    return _entropy_result(
        value=value, err=err, ci_low=ci_low, ci_high=ci_high,
        samples=entropy_samples,
    )


def summarize_asymmetric_replicate_uncertainty(
    *,
    samples: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, float]:
    """Summarize 1-D replicate samples with asymmetric percentile intervals.

    Parameters
    ----------
    samples : np.ndarray
        1-D replicate values; NaN entries are ignored.
    confidence : float
        Two-sided confidence level for the percentile band.

    Returns
    -------
    dict[str, float]
        Dict with ``value`` (median), asymmetric ``ci_low``/``ci_high``,
        the symmetric ``err`` (half band width) plus ``err_low``/``err_high``
        offsets, ``samples``, and ``nan_or_undefined_count``.
    """
    _validate_confidence(confidence=confidence)
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f'samples must be 1-D, got shape {arr.shape}')
    if arr.size == 0:
        raise ValueError('samples must be non-empty')

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            'value': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'err': float('nan'),
            'err_low': float('nan'),
            'err_high': float('nan'),
            'samples': 0.0,
            'nan_or_undefined_count': float(arr.size),
        }

    alpha = 0.5 * (1.0 - confidence)
    value = float(np.nanmedian(finite))
    if finite.size == 1:
        return {
            'value': value,
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'err': float('nan'),
            'err_low': float('nan'),
            'err_high': float('nan'),
            'samples': 1.0,
            'nan_or_undefined_count': float(arr.size - 1),
        }

    ci_low = float(np.nanquantile(finite, alpha))
    ci_high = float(np.nanquantile(finite, 1.0 - alpha))
    err_low = float(max(0.0, value - ci_low))
    err_high = float(max(0.0, ci_high - value))
    return {
        'value': value,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'err': float(max(err_low, err_high)),
        'err_low': err_low,
        'err_high': err_high,
        'samples': float(finite.size),
        'nan_or_undefined_count': float(arr.size - finite.size),
    }


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
    time_series : np.ndarray
        1-D array of sequential measurements (e.g. magnetization).
    max_lag : int | None
        Maximum lag to compute.  Defaults to ``len(time_series) // 2``.
    window_constant : float
        Multiplier for the Madras-Sokal window (default 6.0).

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple ``(C_t, tau_int)`` where ``C_t`` is the normalized autocorrelation
        array truncated at the window endpoint, and ``tau_int`` is the integrated
        autocorrelation time (scalar).

    Raises
    ------
    ValueError
        If ``time_series`` has fewer than 3 elements.
    ZeroVarianceAutocorrelationError
        If ``time_series`` has zero variance.
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
    power = np.abs(fx) ** 2
    acf_raw: np.ndarray = np.fft.irfft(power, n=n_fft)[:max_lag + 1]

    # Normalize: divide by variance and by (N - t) pairs at each lag t.
    # Only lags up to max_lag are ever consumed, so the full-length division
    # of the previous implementation is avoided.
    lags = np.arange(max_lag + 1, dtype=np.float64)
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
    """Estimate effective sample size from integrated autocorrelation time.

    Parameters
    ----------
    time_series : np.ndarray
        1-D time series of at least 2 samples.
    tau_int : float | None
        Optional precomputed integrated autocorrelation time; when
        omitted it is estimated from the series. Must be positive when
        provided.

    Returns
    -------
    float
        N_eff = N / (2 tau_int), clipped to [1, N]; NaN when tau_int is
        undefined, non-finite, or non-positive.
    """
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
    """Estimate standard error using standard blocking/binning analysis.

    Any series containing NaN values (e.g. from failed measurements) yields
    the all-NaN result dict: a partially NaN series would otherwise poison
    the block means and produce a meaningless plateau selection.

    Parameters
    ----------
    time_series : np.ndarray
        1-D time series of at least 2 samples.
    min_block_size : int
        Smallest block size to test (>= 2).
    max_block_size : int | None
        Largest block size to test; defaults to half the
        series length.

    Returns
    -------
    dict[str, float]
        Dict with keys ``stderr`` (plateau-selected standard error),
        ``stderr_naive``, ``block_size``, ``n_blocks``, and
        ``tau_int_from_blocking``.
    """
    arr = _as_1d_float_array(time_series=time_series, name='time_series')
    if min_block_size < 2:
        raise ValueError(f'min_block_size must be >= 2, got {min_block_size}')
    if max_block_size is None:
        max_block_size = arr.size // 2

    # The NaN early-out precedes range validation so that NaN-poisoned series
    # (whose length may be too short for any blocking) still return the
    # graceful NaN dict instead of a range error.
    if np.isnan(arr).any():
        return {
            'stderr': float('nan'),
            'stderr_naive': float('nan'),
            'block_size': float('nan'),
            'n_blocks': 0.0,
            'tau_int_from_blocking': float('nan'),
        }

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


def estimate_tau_int_or_nan(*, time_series: np.ndarray) -> float:
    """Return the integrated autocorrelation time, or NaN when undefined.

    Wraps ``calculate_autocorr`` with the standard zero-variance fallback so
    callers can precompute tau_int once and share it between the primary and
    derived summarizers instead of recomputing the FFT per summarizer.
    """
    try:
        _, tau_int = calculate_autocorr(time_series=time_series)
    except ZeroVarianceAutocorrelationError:
        return float('nan')
    return float(tau_int)


def summarize_primary_observable(
    *,
    time_series: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    blocking: dict[str, float] | None = None,
    tau_int: float | None = None,
) -> dict[str, float]:
    """Summarize a primary observable with blocking-based uncertainty.

    ``blocking`` and ``tau_int`` accept precomputed results from
    ``blocking_error`` and ``estimate_tau_int_or_nan`` on the same series, so
    a caller that also summarizes a derived observable of that series does
    not pay for the blocking scan and autocorrelation FFT twice. When omitted
    they are computed here. A passed ``tau_int`` may be NaN (undefined).

    Parameters
    ----------
    time_series : np.ndarray
        1-D time series of at least 2 samples.
    confidence : float
        Two-sided confidence level for the CI bounds.
    blocking : dict[str, float] | None
        Optional precomputed ``blocking_error`` result dict.
    tau_int : float | None
        Optional precomputed integrated autocorrelation time.

    Returns
    -------
    dict[str, float]
        Dict with the standard uncertainty-schema fields ``value``, ``err``,
        ``ci_low``, ``ci_high``, ``tau_int``, ``n_eff``, and ``samples``.
    """
    _validate_confidence(confidence=confidence)
    arr = _as_1d_float_array(time_series=time_series, name='time_series')

    value = float(np.mean(arr))
    block = blocking if blocking is not None else blocking_error(time_series=arr)
    err = float(block['stderr'])
    z = _z_multiplier(confidence=confidence)

    if tau_int is None:
        tau_int = estimate_tau_int_or_nan(time_series=arr)

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
    return derived_thermo_estimate(
        series=series, temperature=temperature, L=L, observable=observable,
    )


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
    rng_seed: int = 0,
    blocking: dict[str, float] | None = None,
    tau_int: float | None = None,
) -> dict[str, float]:
    """Summarize derived observables (chi, Cv) with uncertainty estimates.

    ``rng_seed`` seeds the block-bootstrap resampling (default 0, matching
    the previously hardcoded stream). ``blocking`` and ``tau_int`` accept
    precomputed results for the base series, as in
    ``summarize_primary_observable``.

    Parameters
    ----------
    magnetization_series : np.ndarray | None
        Base series for ``observable='chi'``.
    energy_series : np.ndarray | None
        Base series for ``observable='cv'``.
    temperature : float
        Measurement temperature (> 0).
    L : int
        Linear lattice size.
    observable : str
        ``'chi'`` or ``'cv'``.
    method : str
        ``'blocking'`` (default) or ``'bootstrap'``.
    confidence : float
        Two-sided confidence level for the CI bounds.
    bootstrap_resamples : int
        Resample count; required > 0 for bootstrap.
    rng_seed : int
        Seed for the block-bootstrap RNG.
    blocking : dict[str, float] | None
        Optional precomputed ``blocking_error`` result dict.
    tau_int : float | None
        Optional precomputed integrated autocorrelation time.

    Returns
    -------
    dict[str, float]
        Dict with the standard uncertainty-schema fields ``value``, ``err``,
        ``ci_low``, ``ci_high``, ``tau_int``, ``n_eff``, and ``samples``.
    """
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

    block = blocking if blocking is not None else blocking_error(time_series=base)
    block_size = int(block['block_size'])
    n_blocks = base.size // block_size
    trimmed = base[:n_blocks * block_size]
    block_reshaped = trimmed.reshape(n_blocks, block_size)

    per_block = np.empty(0, dtype=np.float64)
    if n_blocks >= 2:
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
        rng = np.random.default_rng(rng_seed)
        boot = np.empty(bootstrap_resamples, dtype=np.float64)
        for i in range(bootstrap_resamples):
            sample = rng.choice(per_block, size=n_blocks, replace=True)
            boot[i] = float(np.mean(sample))
        alpha = 0.5 * (1.0 - confidence)
        ci_low = float(np.quantile(boot, alpha))
        ci_high = float(np.quantile(boot, 1.0 - alpha))
        err = float(np.std(boot, ddof=1))

    if tau_int is None:
        tau_int = estimate_tau_int_or_nan(time_series=base)

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
    """Summarize replicate samples using NaN-safe median and percentile bands.

    Parameters
    ----------
    samples : np.ndarray
        Replicate values, either 1-D (one grid point) or 2-D with
        shape ``(n_points, n_replicates)``.
    confidence : float
        Two-sided confidence level; the band spans the
        ``alpha`` and ``1 - alpha`` quantiles with
        ``alpha = (1 - confidence) / 2``.

    Returns
    -------
    dict[str, np.ndarray | float]
        Dict with ``value`` (median), ``err`` (half the band width),
        ``ci_low``, ``ci_high``, ``samples`` (replicate count), and
        ``nan_or_undefined_count``.
    """
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
            # Half the percentile band width serves as the scalar error field so
            # replicate summaries conform to the standard uncertainty schema.
            'err': 0.5 * (ci_high - ci_low),
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
        'err': 0.5 * (ci_high_arr - ci_low_arr),
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


def power_fit(
    *, t_arr: np.ndarray, y_arr: np.ndarray, mask: np.ndarray
) -> tuple[float, float] | tuple[None, None]:
    """
    Fit a power law y = prefactor * t^exponent via log-log linear regression.

    Parameters
    ----------
    t_arr : np.ndarray
        Independent variable array (e.g., time or length scale); only
        positive values are used in the fit.
    y_arr : np.ndarray
        Dependent variable array; only positive values are used in the fit.
    mask : np.ndarray
        Boolean mask selecting the subset of points to include.

    Returns
    -------
    exponent
        Fitted power-law exponent, or None if fewer than 3 valid points.
    prefactor
        Fitted prefactor, or None if fewer than 3 valid points.
    """
    valid = mask & (y_arr > 0) & (t_arr > 0)
    if valid.sum() < 3:
        return None, None
    coeffs = np.polyfit(np.log(t_arr[valid]), np.log(y_arr[valid]), 1)
    return float(coeffs[0]), float(np.exp(coeffs[1]))


