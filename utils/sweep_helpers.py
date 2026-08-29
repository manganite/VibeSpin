"""
Shared infrastructure for per-temperature Monte Carlo sweep workers.

This module provides the two-layer decomposition used by all three model
temperature sweeps (Ising, XY, Clock).

``simulate_at_temperature`` owns equilibration and raw measurement.
``compute_thermo_observables`` owns statistical summarization.
``simulate_thermo_point`` chains both layers and is the worker function passed
to ``parallel_sweep``.

Post-processing helpers ``build_uncertainty_bundle`` and ``build_quality_flags``
aggregate per-seed results over a full temperature axis and are also shared here,
eliminating identical copies that previously lived in each sweep script.
"""
from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np

from utils.equilibration import convergence_equilibrate_with_status
from utils.statistics import (
    UNCERTAINTY_METHOD_BOOTSTRAP,
    _z_multiplier,
    blocking_error,
    estimate_tau_int_or_nan,
    summarize_derived_observable,
    summarize_primary_observable,
    summarize_seed_ensemble,
)


def validate_sweep_uncertainty_args(
    *,
    confidence_level: float,
    max_undefined_fraction: float,
    min_effective_samples: float,
    max_tau_relative_width: float,
    derived_uncertainty_method: str,
    derived_bootstrap_resamples: int,
    n_seeds: int,
) -> None:
    """Validate the shared uncertainty-related CLI arguments of sweep scripts.

    Centralized here so that the Ising, XY, and Clock temperature sweeps
    reject invalid inputs identically instead of failing deep inside worker
    statistics after simulations have already started.

    Parameters
    ----------
    confidence_level : float
        Two-sided confidence level; must satisfy 0 < c < 1.
    max_undefined_fraction : float
        Strict-mode threshold; must satisfy 0 <= f <= 1.
    min_effective_samples : float
        Quality-flag threshold; must be >= 0.
    max_tau_relative_width : float
        Quality-flag threshold; must be >= 0.
    derived_uncertainty_method : str
        Method name for derived observables; bootstrap requires resamples.
    derived_bootstrap_resamples : int
        Bootstrap resample count; must be > 0 when the method is bootstrap.
    n_seeds : int
        Target converged seed replicas; must be >= 1.

    Raises
    ------
    ValueError
        If any argument is outside its documented range.
    """
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError(
            f'confidence-level must satisfy 0 < c < 1, got {confidence_level}'
        )
    if max_undefined_fraction < 0.0 or max_undefined_fraction > 1.0:
        raise ValueError(
            f'max-undefined-fraction must satisfy 0 <= f <= 1, got {max_undefined_fraction}'
        )
    if min_effective_samples < 0.0:
        raise ValueError(
            f'min-effective-samples must be >= 0, got {min_effective_samples}'
        )
    if max_tau_relative_width < 0.0:
        raise ValueError(
            f'max-tau-relative-width must be >= 0, got {max_tau_relative_width}'
        )
    if (
        derived_uncertainty_method == UNCERTAINTY_METHOD_BOOTSTRAP
        and derived_bootstrap_resamples <= 0
    ):
        raise ValueError(
            'derived-bootstrap-resamples must be > 0 when '
            'derived-uncertainty-method=bootstrap'
        )
    if n_seeds < 1:
        raise ValueError(f'n-seeds must be >= 1, got {n_seeds}')

# ---------------------------------------------------------------------------
# Input type
# ---------------------------------------------------------------------------

class ThermoPoint(NamedTuple):
    """Typed worker payload for one (temperature, seed) point in a sweep.

    Parameters
    ----------
    temperature : float
        Simulation temperature T.
    size : int
        Linear lattice dimension L.
    meas_steps : int
        Number of Monte Carlo sweeps to record after equilibration.
    eq_probe_steps : int
        Chunk size passed to ``convergence_equilibrate_with_status``.
    eq_max_steps : int
        Hard cap on total equilibration steps.
    eq_qs_sigma_threshold : float
        Tail-std threshold for quasi-steady stuck-state detection.
    eq_qs_min_steps : int
        Minimum steps before stuck detection is allowed to fire.
    qs_allow_stuck : bool
        If True, a quasi-steady stuck state counts as convergence (Ising low-T).
    prefer_ordered_start : bool
        If True, switch to the ordered-start simulation when its magnetisation
        substantially exceeds the random-start value.  Appropriate for Ising below T_c.
    temperature_index : int
        Index of this temperature in the sweep's temperature array.
    seed_index : int
        Replica index within the multi-seed ensemble for this temperature.
    seed : int
        RNG seed for both the random- and ordered-start simulations.
    model_cls : type
        Simulation class to instantiate (e.g. ``IsingSimulation``). Must be
        importable by its qualified name so it survives multiprocessing pickle.
    model_kwargs : dict[str, Any]
        Extra keyword arguments forwarded to the model constructor beyond
        ``size``, ``temp``, ``init_state``, and ``seed``.  Empty dict for
        Ising and XY; carries ``{'q': q, 'A': aniso}`` for Clock.
    confidence : float
        Two-sided confidence level for uncertainty intervals (0 < c < 1).
    derived_method : str
        Uncertainty method for susceptibility and specific heat
        (``'blocking'`` or ``'bootstrap'``).
    bootstrap_resamples : int
        Bootstrap resamples used when ``derived_method='bootstrap'``.
    """

    temperature: float
    size: int
    meas_steps: int
    eq_probe_steps: int
    eq_max_steps: int
    eq_qs_sigma_threshold: float
    eq_qs_min_steps: int
    qs_allow_stuck: bool
    prefer_ordered_start: bool
    temperature_index: int
    seed_index: int
    seed: int
    model_cls: type
    model_kwargs: dict[str, Any]
    confidence: float
    derived_method: str
    bootstrap_resamples: int


# ---------------------------------------------------------------------------
# Intermediate result type
# ---------------------------------------------------------------------------

class RawThermoData(NamedTuple):
    """Raw simulation output from one (temperature, seed) equilibration + measurement.

    Parameters
    ----------
    temperature_index : int
        Index of this temperature in the sweep's temperature array.
    seed_index : int
        Replica index within the multi-seed ensemble for this temperature.
    equilibrated_flag : float
        1.0 if equilibration converged, 0.0 otherwise.
    equilibration_steps : float
        Total equilibration steps taken (for diagnostics).
    temperature : float
        Temperature at which the simulation was run.
    size : int
        Linear lattice dimension L.
    mags_arr : np.ndarray or None
        Magnetisation time series of length ``meas_steps``.
        ``None`` when equilibration did not converge.
    engs_arr : np.ndarray or None
        Energy per site time series of length ``meas_steps``.
        ``None`` when equilibration did not converge.
    """

    temperature_index: int
    seed_index: int
    equilibrated_flag: float
    equilibration_steps: float
    temperature: float
    size: int
    mags_arr: np.ndarray | None
    engs_arr: np.ndarray | None


# ---------------------------------------------------------------------------
# Layer 1: simulation + equilibration
# ---------------------------------------------------------------------------

def simulate_at_temperature(point: ThermoPoint) -> RawThermoData:
    """Run equilibration and measurement for one (temperature, seed) point.

    Creates a random-start and an ordered-start simulation with the same seed,
    runs two-start convergence equilibration, then records ``meas_steps`` sweeps
    on the active simulation.  Returns raw magnetisation and energy time series
    for downstream statistical processing.

    Parameters
    ----------
    point : ThermoPoint
        Fully configured worker payload.

    Returns
    -------
    RawThermoData
        Raw time series; ``mags_arr`` and ``engs_arr`` are ``None`` when
        equilibration did not converge.
    """
    T = point.temperature
    L = point.size

    sim_r = point.model_cls(
        size=L, temp=T, init_state='random', seed=point.seed,
        **point.model_kwargs,
    )
    sim_o = point.model_cls(
        size=L, temp=T, init_state='ordered', seed=point.seed,
        **point.model_kwargs,
    )

    total_steps, converged = convergence_equilibrate_with_status(
        sim_random=sim_r,
        sim_ordered=sim_o,
        chunk_size=point.eq_probe_steps,
        max_steps=point.eq_max_steps,
        qs_sigma_threshold=point.eq_qs_sigma_threshold,
        qs_min_steps=point.eq_qs_min_steps,
        qs_allow_stuck=point.qs_allow_stuck,
    )

    if not converged:
        return RawThermoData(
            temperature_index=point.temperature_index,
            seed_index=point.seed_index,
            equilibrated_flag=0.0,
            equilibration_steps=float(total_steps),
            temperature=T,
            size=L,
            mags_arr=None,
            engs_arr=None,
        )

    # For models where the ordered start is physically cleaner (Ising below T_c),
    # switch to it when its magnetisation is substantially higher.
    active_sim = sim_r
    if point.prefer_ordered_start:
        m_r = float(np.abs(sim_r.get_magnetization()))
        m_o = float(np.abs(sim_o.get_magnetization()))
        if m_o > m_r + 0.2:
            active_sim = sim_o

    mags, engs = active_sim.run(n_steps=point.meas_steps)

    return RawThermoData(
        temperature_index=point.temperature_index,
        seed_index=point.seed_index,
        equilibrated_flag=1.0,
        equilibration_steps=float(total_steps),
        temperature=T,
        size=L,
        mags_arr=np.asarray(mags, dtype=np.float64),
        engs_arr=np.asarray(engs, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Layer 2: statistical summarization
# ---------------------------------------------------------------------------

def compute_thermo_observables(
    raw: RawThermoData,
    point: ThermoPoint,
) -> dict[str, float]:
    """Compute blocked/summarized thermodynamic observables from raw time series.

    Takes the raw output of ``simulate_at_temperature`` and applies
    autocorrelation-aware blocking (primary observables) and variance-based
    summarization (derived observables) to produce the standard per-point
    result dictionary.

    Parameters
    ----------
    raw : RawThermoData
        Raw time series from ``simulate_at_temperature``.
    point : ThermoPoint
        Originating worker payload; provides temperature, size, and statistical
        configuration (confidence, derived_method, bootstrap_resamples).

    Returns
    -------
    dict[str, float]
        Keys: ``temperature_index``, ``seed_index``, ``equilibrated_flag``,
        ``equilibration_steps``, and — when equilibrated — ``avg_m_*``,
        ``avg_e_*``, ``susc_*``, ``spec_h_*`` with suffixes ``_value``,
        ``_err``, ``_tau_int``, ``_n_eff``.
    """
    base: dict[str, float] = {
        'temperature_index': float(raw.temperature_index),
        'seed_index': float(raw.seed_index),
        'equilibrated_flag': raw.equilibrated_flag,
        'equilibration_steps': raw.equilibration_steps,
    }

    if raw.equilibrated_flag == 0.0 or raw.mags_arr is None or raw.engs_arr is None:
        return base

    T = raw.temperature
    L = raw.size
    mags_arr = raw.mags_arr
    engs_arr = raw.engs_arr

    # Blocking and autocorrelation are shared between the primary summary and
    # the derived observable of the same series, so compute each exactly once.
    mag_blocking = blocking_error(time_series=mags_arr)
    mag_tau = estimate_tau_int_or_nan(time_series=mags_arr)
    eng_blocking = blocking_error(time_series=engs_arr)
    eng_tau = estimate_tau_int_or_nan(time_series=engs_arr)

    mag = summarize_primary_observable(
        time_series=mags_arr,
        confidence=point.confidence,
        blocking=mag_blocking,
        tau_int=mag_tau,
    )
    eng = summarize_primary_observable(
        time_series=engs_arr,
        confidence=point.confidence,
        blocking=eng_blocking,
        tau_int=eng_tau,
    )
    chi = summarize_derived_observable(
        magnetization_series=mags_arr,
        temperature=T,
        L=L,
        observable='chi',
        method=point.derived_method,
        confidence=point.confidence,
        bootstrap_resamples=point.bootstrap_resamples,
        blocking=mag_blocking,
        tau_int=mag_tau,
    )
    cv = summarize_derived_observable(
        energy_series=engs_arr,
        temperature=T,
        L=L,
        observable='cv',
        method=point.derived_method,
        confidence=point.confidence,
        bootstrap_resamples=point.bootstrap_resamples,
        blocking=eng_blocking,
        tau_int=eng_tau,
    )

    return {
        **base,
        'avg_m_value': float(mag['value']),
        'avg_m_err': float(mag['err']),
        'avg_m_tau_int': float(mag['tau_int']),
        'avg_m_n_eff': float(mag['n_eff']),
        'avg_e_value': float(eng['value']),
        'avg_e_err': float(eng['err']),
        'avg_e_tau_int': float(eng['tau_int']),
        'avg_e_n_eff': float(eng['n_eff']),
        'susc_value': float(chi['value']),
        'susc_err': float(chi['err']),
        'susc_tau_int': float(chi['tau_int']),
        'susc_n_eff': float(chi['n_eff']),
        'spec_h_value': float(cv['value']),
        'spec_h_err': float(cv['err']),
        'spec_h_tau_int': float(cv['tau_int']),
        'spec_h_n_eff': float(cv['n_eff']),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper (used by parallel_sweep workers)
# ---------------------------------------------------------------------------

def simulate_thermo_point(point: ThermoPoint) -> dict[str, float]:
    """Run a full (equilibration + measurement + summarization) point.

    Thin wrapper over ``simulate_at_temperature`` followed by
    ``compute_thermo_observables``.  This is the function passed as
    ``worker_func`` to ``parallel_sweep`` in temperature-sweep scripts.

    Parameters
    ----------
    point : ThermoPoint
        Fully configured worker payload.

    Returns
    -------
    dict[str, float]
        Standard per-point result dictionary; see ``compute_thermo_observables``
        for the full key set.
    """
    return compute_thermo_observables(simulate_at_temperature(point), point)


# ---------------------------------------------------------------------------
# Post-processing helpers (shared by all temperature-sweep scripts)
# ---------------------------------------------------------------------------

def build_uncertainty_bundle(
    *,
    values_by_seed: np.ndarray,
    errors_by_seed: np.ndarray,
    tau_by_seed: np.ndarray,
    n_eff_by_seed: np.ndarray,
    confidence: float,
) -> dict[str, np.ndarray]:
    """Aggregate per-seed results for one observable across a temperature axis.

    For multi-seed sweeps, applies hierarchical seed-ensemble aggregation via
    ``summarize_seed_ensemble``.  Falls back to direct single-seed blocking
    results when only one seed is available.

    Parameters
    ----------
    values_by_seed : np.ndarray
        Shape ``(n_temps, n_seeds)``.  Per-seed point estimates.
    errors_by_seed : np.ndarray
        Shape ``(n_temps, n_seeds)``.  Per-seed within-seed uncertainties.
    tau_by_seed : np.ndarray
        Shape ``(n_temps, n_seeds)``.  Per-seed integrated autocorrelation times.
    n_eff_by_seed : np.ndarray
        Shape ``(n_temps, n_seeds)``.  Per-seed effective sample sizes.
    confidence : float
        Two-sided confidence level for the returned CI bounds.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``'value'``, ``'err'``, ``'ci_low'``, ``'ci_high'``,
        ``'tau_int'``, ``'n_eff'``, ``'samples'``.
        All arrays have length ``n_temps`` except ``'samples'`` which has
        shape ``(n_temps, n_seeds)``.
    """
    n_seeds = values_by_seed.shape[1]
    if n_seeds > 1:
        res_values = []
        res_errors = []
        res_low = []
        res_high = []

        for t in range(values_by_seed.shape[0]):
            mask = ~np.isnan(values_by_seed[t])
            if not np.any(mask):
                res_values.append(np.nan)
                res_errors.append(np.nan)
                res_low.append(np.nan)
                res_high.append(np.nan)
                continue

            summary = summarize_seed_ensemble(
                values=values_by_seed[t, mask],
                within_seed_errors=errors_by_seed[t, mask],
                confidence=confidence,
            )
            res_values.append(summary['value'])
            res_errors.append(summary['err'])
            res_low.append(summary['ci_low'])
            res_high.append(summary['ci_high'])

        res = {
            'value': np.array(res_values),
            'err': np.array(res_errors),
            'ci_low': np.array(res_low),
            'ci_high': np.array(res_high),
        }

        # Ensemble-averaged diagnostics; guard against all-NaN temperature rows.
        nan_mask = np.all(np.isnan(tau_by_seed), axis=1)
        tau_int = np.full(nan_mask.shape, np.nan)
        n_eff = np.full(nan_mask.shape, np.nan)

        if not np.all(nan_mask):
            with np.errstate(all='ignore'):
                tau_int[~nan_mask] = np.nanmean(tau_by_seed[~nan_mask], axis=1)
                n_eff[~nan_mask] = np.nanmean(n_eff_by_seed[~nan_mask], axis=1)

        if np.any(nan_mask):
            logger = logging.getLogger('vibespin')
            logger.warning(
                'Autocorrelation diagnostics are undefined for %d temperature '
                'points (all seeds returned NaN). This is expected in the deep '
                'ordered/frozen phase.',
                int(np.sum(nan_mask)),
            )
    else:
        # Apply the same Gaussian z-multiplier as the multi-seed path so that
        # 'ci_low'/'ci_high' honor the requested confidence level here too.
        z = _z_multiplier(confidence=confidence)
        res = {
            'value': values_by_seed[:, 0],
            'err': errors_by_seed[:, 0],
            'ci_low': values_by_seed[:, 0] - z * errors_by_seed[:, 0],
            'ci_high': values_by_seed[:, 0] + z * errors_by_seed[:, 0],
        }
        tau_int = tau_by_seed[:, 0]
        n_eff = n_eff_by_seed[:, 0]

    return {
        'value': res['value'],
        'err': res['err'],
        'ci_low': res['ci_low'],
        'ci_high': res['ci_high'],
        'tau_int': tau_int,
        'n_eff': n_eff,
        'samples': values_by_seed.astype(np.float64),
    }


def build_quality_flags(
    *,
    tau_int: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    n_eff: np.ndarray,
    min_effective_samples: float,
    max_tau_relative_width: float,
) -> dict[str, np.ndarray]:
    """Build per-temperature diagnostic flags for uncertainty quality.

    Parameters
    ----------
    tau_int : np.ndarray
        Shape ``(n_temps,)``.  Integrated autocorrelation time estimates.
    ci_low : np.ndarray
        Shape ``(n_temps,)``.  Lower CI bound on ``tau_int``.
    ci_high : np.ndarray
        Shape ``(n_temps,)``.  Upper CI bound on ``tau_int``.
    n_eff : np.ndarray
        Shape ``(n_temps,)``.  Effective sample size estimates.
    min_effective_samples : float
        Threshold below which a point is flagged as low-N_eff.
    max_tau_relative_width : float
        CI-width-to-tau ratio above which the tau estimate is considered
        unstable.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``'undefined_autocorr_flag'``, ``'low_effective_sample_flag'``,
        ``'tau_interval_unstable_flag'``.  All uint8 arrays of length
        ``n_temps``.
    """
    undefined_autocorr = ~np.isfinite(tau_int)
    low_effective_sample = np.isfinite(n_eff) & (n_eff < min_effective_samples)

    tau_span = ci_high - ci_low
    denom = np.maximum(np.abs(tau_int), 1e-12)
    rel_width = tau_span / denom
    tau_interval_unstable = np.isfinite(rel_width) & (rel_width > max_tau_relative_width)

    return {
        'undefined_autocorr_flag': undefined_autocorr.astype(np.uint8),
        'low_effective_sample_flag': low_effective_sample.astype(np.uint8),
        'tau_interval_unstable_flag': tau_interval_unstable.astype(np.uint8),
    }
