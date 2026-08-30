"""Equilibration algorithms for Monte Carlo simulations.

Provides adaptive and two-start convergence equilibration routines, together
with the underlying relaxation-time estimation and quasi-steady-state detection
helpers that support them.  These algorithms were previously split between
``utils.system`` (equilibrate functions) and ``utils.analysis``
(relaxation diagnostic helpers), which required a lazy cross-import to avoid
circular dependencies.  Unifying them here eliminates that workaround.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from utils.exceptions import ZeroVarianceAutocorrelationError

# ---------------------------------------------------------------------------
# Structural type
# ---------------------------------------------------------------------------


class _Sim(Protocol):
    """Structural type for any MonteCarloSimulation; avoids a models/ → utils/ import."""

    size: int

    def equilibrate(self, *, n_steps: int) -> None: ...

    def run(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray]: ...


# ---------------------------------------------------------------------------
# Private signal-processing helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Relaxation-time estimation
# ---------------------------------------------------------------------------

_QS_SIGMA_REF_L: int = 32
"""Reference lattice side-length for ``_detect_quasi_steady_stuck`` threshold scaling.

The ``qs_sigma_threshold`` parameter is calibrated at this L and scaled
proportionally as ``threshold * _QS_SIGMA_REF_L / lattice_size`` for other sizes.
"""

# Shared defaults for the two-start criteria. The convergence loop forwards
# user overrides via **kwargs to both the relaxation estimator and the stuck
# detector; keeping the fallback values here (instead of repeating literals)
# guarantees the two criteria cannot silently drift apart.
_TWO_START_DEFAULT_K: float = 1.0
_TWO_START_DEFAULT_SMOOTH_WINDOW: int = 60
_TWO_START_DEFAULT_SIGMA_FLOOR: float = 0.02


def _prepare_smoothed_traces(
    *,
    trace_random: np.ndarray,
    trace_ordered: np.ndarray,
    smooth_window: int,
    skip_validation: bool,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Shared preprocessing for the two-start criteria.

    Takes the absolute values (optionally after finite-prefix validation),
    truncates both traces to a common length ``n``, and applies the
    moving-average smoother. Returns ``(r_smoothed, o_smoothed, n)``, or
    ``None`` when the traces are too short to analyze (fewer than 8 common
    raw samples).
    """
    if skip_validation:
        r = np.abs(trace_random)
        o = np.abs(trace_ordered)
    else:
        r = np.abs(_valid_prefix(np.asarray(trace_random, dtype=float)))
        o = np.abs(_valid_prefix(np.asarray(trace_ordered, dtype=float)))

    n = min(len(r), len(o))
    if n < 8:
        return None

    r_sm = _moving_average(r[:n], smooth_window)
    o_sm = _moving_average(o[:n], smooth_window)
    m = min(len(r_sm), len(o_sm))
    return r_sm[:m], o_sm[:m], n


def estimate_relaxation_time_two_start(
    *,
    trace_random: np.ndarray,
    trace_ordered: np.ndarray,
    k: float = _TWO_START_DEFAULT_K,
    smooth_window: int = _TWO_START_DEFAULT_SMOOTH_WINDOW,
    dwell_window: int = 60,
    min_fraction_inside: float = 0.85,
    sigma_floor: float = _TWO_START_DEFAULT_SIGMA_FLOOR,
    skip_validation: bool = False,
) -> int:
    """Estimate thermalization time from convergence of random- and ordered-start traces.

    Compares a trajectory started from a random spin configuration against one
    started from a fully ordered configuration.  Returns the first step at which
    both smoothed traces pass a mutual cross-band test sustained over a dwell window.

    The criterion is asymmetric in the correct physical sense: the smoothed
    random-start trace must lie within ``k`` standard deviations of the
    ordered-start tail mean, and the ordered-start trace must simultaneously
    lie within ``k`` standard deviations of the random-start tail mean.
    Tail statistics are computed from the second half of each smoothed trace
    independently.  A ``sigma_floor`` prevents band collapse when one trace is
    nearly variance-free (e.g. in the deep ordered phase near T=0).

    Parameters
    ----------
    trace_random : np.ndarray
        1-D magnetization trace from a random initial state.
    trace_ordered : np.ndarray
        1-D magnetization trace from an ordered initial state.
    k : float
        Half-width of each convergence band in units of the respective tail
        standard deviation.  Larger values are more permissive.
    smooth_window : int
        Window length (steps) for the moving-average smoother.
    dwell_window : int
        Window length (steps) for the sustained-convergence test.
    min_fraction_inside : float
        Fraction of steps within ``dwell_window`` that must
        satisfy the mutual cross-band condition to declare convergence.
    sigma_floor : float
        Minimum allowed standard deviation for each tail, preventing
        band collapse when a trajectory is nearly flat.
    skip_validation : bool
        If True, skip np.asarray and finite-prefix checks.

    Returns
    -------
    int
        Estimated relaxation time as a 0-indexed position in the original input
        traces (i.e. a sweep count minus one).  The first index at which both
        smoothed traces have been mutually inside each other's band for
        ``dwell_window`` steps is mapped back to the corresponding raw-trace
        position as ``hits[0] + smooth_window - 1``.  Returns ``n`` (full trace
        length) if no convergence is detected, or ``0`` for very short traces.
    """
    prepared = _prepare_smoothed_traces(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        smooth_window=smooth_window,
        skip_validation=skip_validation,
    )
    if prepared is None:
        return 0
    r_sm, o_sm, n = prepared
    m = len(r_sm)

    # Compute tail statistics independently for each trace
    half = m // 2
    tail_r = r_sm[half:]
    tail_o = o_sm[half:]
    mu_r = tail_r.mean()
    mu_o = tail_o.mean()
    sig_r = max(float(tail_r.std()), sigma_floor)
    sig_o = max(float(tail_o.std()), sigma_floor)

    # Mutual cross-band test: each trace must be inside the other trace's band
    in_o_band = np.abs(r_sm - mu_o) <= k * sig_o  # random inside ordered's band
    in_r_band = np.abs(o_sm - mu_r) <= k * sig_r  # ordered inside random's band
    both_inside = (in_o_band & in_r_band).astype(float)

    dwell_window = max(3, min(dwell_window, len(both_inside)))
    sustained_fraction = (
        np.convolve(both_inside, np.ones(dwell_window), mode='valid') / dwell_window
    )

    hits = np.where(sustained_fraction >= min_fraction_inside)[0]
    return int(hits[0]) + smooth_window - 1 if hits.size else n


def _detect_quasi_steady_stuck(
    *,
    trace_random: np.ndarray,
    trace_ordered: np.ndarray,
    k: float = _TWO_START_DEFAULT_K,
    smooth_window: int = _TWO_START_DEFAULT_SMOOTH_WINDOW,
    qs_sigma_threshold: float = 0.05,
    sigma_floor: float = _TWO_START_DEFAULT_SIGMA_FLOOR,
    lattice_size: int | None = None,
    skip_validation: bool = False,
) -> bool:
    """Detect low-temperature stuck states in two-start equilibration.

    A stuck state is declared when the ordered-start trace has settled
    (small tail variance, as expected for any equilibrated state) but the
    two smoothed traces still fail the mutual cross-band condition, indicating
    the random-start trace is stranded in a metastable plateau.

    The guard uses only the ordered-trace tail variance, not the random-trace
    variance.  The random trace in a multi-domain stuck state has domain-wall
    fluctuations of similar amplitude to the thermal fluctuations of a
    thermalized disordered system, making it an unreliable discriminator.
    The ordered trace is always the reliable anchor: in the deep ordered phase
    its variance is essentially zero; in the disordered phase it carries thermal
    fluctuations of magnitude ``sigma ~ 1 / lattice_size``.

    When ``lattice_size`` is provided, the effective threshold scales as
    ``qs_sigma_threshold * _QS_SIGMA_REF_L / lattice_size``.  This keeps a
    constant safety margin between the threshold and the thermal ordered-trace
    variance at any system size.  At ``lattice_size = _QS_SIGMA_REF_L`` the
    effective threshold equals ``qs_sigma_threshold`` exactly.
    """
    prepared = _prepare_smoothed_traces(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        smooth_window=smooth_window,
        skip_validation=skip_validation,
    )
    if prepared is None:
        return False
    r_sm, o_sm, _ = prepared
    m = len(r_sm)
    if m < 8:
        return False

    half = m // 2
    tail_r = r_sm[half:]
    tail_o = o_sm[half:]

    raw_sig_r = float(tail_r.std())
    raw_sig_o = float(tail_o.std())

    effective_threshold = (
        qs_sigma_threshold * _QS_SIGMA_REF_L / lattice_size
        if lattice_size is not None
        else qs_sigma_threshold
    )
    # Guard on the ordered trace only: it is the reliable anchor.
    # The random trace in a stuck multi-domain state has domain-wall fluctuations
    # of similar amplitude to thermal noise in the disordered phase, making
    # raw_sig_r an unreliable discriminator between stuck and thermalized.
    if raw_sig_o >= effective_threshold:
        return False

    mu_r = float(tail_r.mean())
    mu_o = float(tail_o.mean())
    sig_r = max(raw_sig_r, sigma_floor)
    sig_o = max(raw_sig_o, sigma_floor)

    random_inside_ordered_band = abs(mu_r - mu_o) <= k * sig_o
    ordered_inside_random_band = abs(mu_o - mu_r) <= k * sig_r
    mutually_inside = random_inside_ordered_band and ordered_inside_random_band

    return not mutually_inside


# ---------------------------------------------------------------------------
# Adaptive equilibration
# ---------------------------------------------------------------------------


def adaptive_equilibrate(
    sim: _Sim,
    *,
    min_steps: int,
    probe_steps: int = 500,
    factor: float = 50.0,
    max_steps: int = 200_000,
) -> int:
    """
    Equilibrate a simulation adaptively, extending the burn-in until the probe
    window covers ``factor`` integrated autocorrelation times.

    After the mandatory ``min_steps`` burn-in, the function repeatedly runs a
    ``probe_steps`` measurement via ``sim.run()`` and computes ``tau_int`` from
    the resulting magnetization series. If ``probe_steps >= factor * tau_int``,
    the probe spans enough correlation times for the initial state to be
    forgotten and equilibration is declared complete. Otherwise the probe has
    advanced the system state and the loop continues. In the ordered phase the
    magnetization series has zero variance; this is treated as full
    equilibration and the function returns immediately.

    Parameters
    ----------
    sim : _Sim
        Any simulation object implementing ``equilibrate`` and ``run``.
    min_steps : int
        Mandatory burn-in passed to ``sim.equilibrate`` before probing.
    probe_steps : int
        MC steps per probe run (default 500).
    factor : float
        Required ratio ``probe_steps / tau_int`` (default 50.0).
    max_steps : int
        Hard cap on total steps to prevent unbounded runtime near
        criticality (default 200 000).

    Returns
    -------
    int
        Total number of MC steps run (burn-in + probes).

    Raises
    ------
    ValueError
        If the adaptive-equilibration parameters are invalid.
    """
    from utils.statistics import calculate_autocorr  # lazy import; avoids pickle issues

    if min_steps < 0:
        raise ValueError(f'min_steps must be non-negative, got {min_steps}')
    if probe_steps < 3:
        raise ValueError(f'probe_steps must be >= 3, got {probe_steps}')
    if factor <= 0.0:
        raise ValueError(f'factor must be positive, got {factor}')
    if max_steps < min_steps:
        raise ValueError(
            f'max_steps must be >= min_steps, got max_steps={max_steps} and min_steps={min_steps}'
        )

    logger = logging.getLogger('vibespin')
    sim.equilibrate(n_steps=min_steps)
    total = min_steps

    while total < max_steps:
        mags, _ = sim.run(n_steps=probe_steps)
        total += probe_steps
        try:
            _, tau_int = calculate_autocorr(time_series=mags)
        except ZeroVarianceAutocorrelationError:
            # Zero variance: fully ordered phase, equilibration is trivially satisfied.
            return total
        if probe_steps >= factor * tau_int:
            return total

    logger.warning(
        f'adaptive_equilibrate: reached max_steps={max_steps} without satisfying '
        f'criterion probe_steps({probe_steps}) >= factor({factor}) * tau_int; '
        'proceeding anyway.'
    )
    return total


# ---------------------------------------------------------------------------
# Two-start convergence equilibration
# ---------------------------------------------------------------------------


def convergence_equilibrate(
    *,
    sim_random: _Sim,
    sim_ordered: _Sim,
    chunk_size: int = 500,
    max_steps: int = 200_000,
    qs_sigma_threshold: float = 0.05,
    qs_min_steps: int = 1500,
    **kwargs: Any,
) -> int:
    """
    Equilibrate two simulations (random- and ordered-start) until they converge.

    Uses ``estimate_relaxation_time_two_start`` to detect convergence via the
    mutual cross-band criterion: each smoothed trajectory must enter and sustain
    a band defined by the other trajectory's tail statistics.  A sigma floor
    prevents false positives when one trace is nearly flat.  This is more robust
    than one-start adaptive methods for complex energy landscapes.

    Parameters
    ----------
    sim_random : _Sim
        Simulation instance started from a random state.
    sim_ordered : _Sim
        Simulation instance started from an ordered state.
    chunk_size : int
        Number of steps to run between convergence checks.
    max_steps : int
        Hard cap on total steps.
    qs_sigma_threshold : float
        Tail-std threshold used to detect a quasi-steady,
        non-converged stuck state and exit early.
    qs_min_steps : int
        Minimum accumulated steps before stuck detection is allowed
        to fire.  Ensures tail statistics are computed from enough data to be
        reliable, preventing false positives when traces have not yet had time
        to settle.
    **kwargs : Any
        Passed to ``estimate_relaxation_time_two_start`` (k, smooth_window,
        dwell_window, min_fraction_inside, sigma_floor, etc.).

    Returns
    -------
    int
        Total number of MC steps run per simulation.
    """
    total, _ = convergence_equilibrate_with_status(
        sim_random=sim_random,
        sim_ordered=sim_ordered,
        chunk_size=chunk_size,
        max_steps=max_steps,
        qs_sigma_threshold=qs_sigma_threshold,
        qs_min_steps=qs_min_steps,
        **kwargs,
    )
    return total


def convergence_equilibrate_with_status(
    *,
    sim_random: _Sim,
    sim_ordered: _Sim,
    chunk_size: int = 500,
    max_steps: int = 200_000,
    qs_sigma_threshold: float = 0.05,
    qs_min_steps: int = 1500,
    qs_allow_stuck: bool = False,
    **kwargs: Any,
) -> tuple[int, bool]:
    """
    Equilibrate two simulations and report whether convergence was reached.

    Uses ``estimate_relaxation_time_two_start`` to detect convergence via the
    mutual cross-band criterion: each smoothed trajectory must enter and sustain
    a band defined by the other trajectory's tail statistics.  A sigma floor
    prevents false positives when one trace is nearly flat.  This variant returns
    both the total number of MC steps executed and a boolean convergence flag.

    Parameters
    ----------
    sim_random : _Sim
        Simulation instance started from a random state.
    sim_ordered : _Sim
        Simulation instance started from an ordered state.
    chunk_size : int
        Number of steps to run between convergence checks.
    max_steps : int
        Hard cap on total steps.
    qs_sigma_threshold : float
        Tail-std threshold used to detect a quasi-steady,
        non-converged stuck state and exit early.
    qs_min_steps : int
        Minimum accumulated steps before stuck detection is allowed
        to fire.  Ensures tail statistics are computed from enough data to be
        reliable, preventing false positives when traces have not yet had time
        to settle.
    qs_allow_stuck : bool
        If True, detecting a quasi-steady stuck state (where the
        ordered trace is stable but the random trace is stranded) is treated
        as a successful equilibration. Useful for Ising sweeps where
        random-start domain-wall trapping is physically expected.
    **kwargs : Any
        Passed to ``estimate_relaxation_time_two_start`` (k, smooth_window,
        dwell_window, min_fraction_inside, sigma_floor, etc.).

    Returns
    -------
    tuple[int, bool]
        Tuple ``(total_steps, converged)``.
    """
    logger = logging.getLogger('vibespin')
    mags_r = np.full(max_steps, np.nan, dtype=float)
    mags_o = np.full(max_steps, np.nan, dtype=float)
    total = 0
    # Every convergence check re-analyzes the full accumulated trace, so
    # checking after every chunk costs O(total^2 / chunk_size) over a long
    # run. Once the trace is much longer than a chunk, space the checks
    # geometrically (25% growth); the summed analysis cost then stays linear
    # in max_steps. Convergence between two checks is caught at the next
    # check, so equilibration can only end slightly later, never earlier.
    next_check_total = 100

    while total < max_steps:
        # Run next chunk
        mr, _ = sim_random.run(n_steps=chunk_size)
        mo, _ = sim_ordered.run(n_steps=chunk_size)

        # Clip chunk if it would exceed max_steps
        actual_len = min(len(mr), max_steps - total)
        if actual_len <= 0:
            break

        mags_r[total : total + actual_len] = mr[:actual_len]
        mags_o[total : total + actual_len] = mo[:actual_len]
        total += actual_len

        # Only check if we have enough data for a meaningful estimate
        # smooth_window defaults to 60, dwell_window to 60.
        if total >= next_check_total:
            next_check_total = total + max(chunk_size, total // 4)
            # Pass slices (views) to avoid copying
            trace_r = mags_r[:total]
            trace_o = mags_o[:total]

            tau = estimate_relaxation_time_two_start(
                trace_random=trace_r,
                trace_ordered=trace_o,
                skip_validation=True,
                **kwargs,
            )
            # If estimate_relaxation_time_two_start returns a value < total,
            # it means convergence was detected at some point in the past.
            if tau < total:
                return total, True

            if total >= qs_min_steps and _detect_quasi_steady_stuck(
                trace_random=trace_r,
                trace_ordered=trace_o,
                k=float(kwargs.get('k', _TWO_START_DEFAULT_K)),
                smooth_window=int(kwargs.get('smooth_window', _TWO_START_DEFAULT_SMOOTH_WINDOW)),
                qs_sigma_threshold=qs_sigma_threshold,
                sigma_floor=float(kwargs.get('sigma_floor', _TWO_START_DEFAULT_SIGMA_FLOOR)),
                lattice_size=getattr(sim_random, 'size', None),
                skip_validation=True,
            ):
                if qs_allow_stuck:
                    logger.info(
                        'convergence_equilibrate: detected stable quasi-steady stuck state; '
                        'stopping early and accepting ordered start.'
                    )
                    return total, True

                logger.warning(
                    'convergence_equilibrate: detected quasi-steady stuck state '
                    f'before max_steps={max_steps}; stopping early without convergence.'
                )
                return total, False

    logger.warning(
        f'convergence_equilibrate: reached max_steps={max_steps} without convergence; '
        'proceeding anyway.'
    )
    return total, False
