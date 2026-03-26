"""
Unit tests for physics-related utility functions in utils/physics_helpers.py.
Covers thermodynamic averages, entropy, autocorrelation, spatial diagnostics,
kinetics metrics, and power-law fitting.
"""
from __future__ import annotations

# mypy: disable-error-code=no-untyped-def
import numpy as np
import pytest

from models.ising_model import IsingSimulation
from utils.equilibration import _detect_quasi_steady_stuck, estimate_relaxation_time_two_start
from utils.exceptions import ZeroVarianceAutocorrelationError
from utils.observables import (
    calculate_entropy,
    calculate_thermodynamics,
)
from utils.statistics import (
    UNCERTAINTY_METHOD_BOOTSTRAP,
    _as_1d_float_array,
    _validate_confidence,
    blocking_error,
    calculate_autocorr,
    estimate_effective_sample_size,
    power_fit,
    summarize_asymmetric_replicate_uncertainty,
    summarize_derived_observable,
    summarize_entropy_observable,
    summarize_primary_observable,
    summarize_replicate_samples,
    summarize_seed_ensemble,
)


@pytest.fixture
def ising_sim():
    """Fixture for a small Ising simulation."""
    return IsingSimulation(size=10, temp=2.0)


# ---- calculate_thermodynamics ----


def test_calculate_thermodynamics_basic():
    """Verify thermodynamic observable calculations with known values."""
    mags = np.array([0.5, 0.7, 0.6])
    engs = np.array([-1.2, -1.0, -1.1])
    T, L = 2.0, 10

    avg_mag, avg_eng, chi, cv = calculate_thermodynamics(mags=mags, engs=engs, T=T, L=L)

    assert avg_mag == pytest.approx(0.6)
    assert avg_eng == pytest.approx(-1.1)
    assert chi == pytest.approx(100 * np.var(mags) / T)
    assert cv == pytest.approx(100 * np.var(engs) / (T**2))


def test_calculate_thermodynamics_returns_four_floats():
    """Should return exactly four float values."""
    mags = np.array([0.8, 0.9, 0.85, 0.82])
    engs = np.array([-1.5, -1.6, -1.55, -1.52])
    result = calculate_thermodynamics(mags=mags, engs=engs, T=2.0, L=10)
    assert len(result) == 4
    for val in result:
        assert isinstance(val, float)


def test_average_magnetization():
    """avg_mag should be the mean of the input magnetization array."""
    mags = np.array([0.4, 0.6, 0.8, 1.0])
    engs = np.array([-1.0, -1.0, -1.0, -1.0])
    avg_mag, _, _, _ = calculate_thermodynamics(mags=mags, engs=engs, T=2.0, L=5)
    assert pytest.approx(avg_mag) == float(np.mean(mags))


def test_average_energy():
    """avg_eng should be the mean of the input energy array."""
    mags = np.array([0.5, 0.5])
    engs = np.array([-2.0, -4.0])
    _, avg_eng, _, _ = calculate_thermodynamics(mags=mags, engs=engs, T=1.0, L=4)
    assert pytest.approx(avg_eng) == -3.0


def test_susceptibility_zero_variance():
    """Susceptibility should be zero for constant magnetization (no fluctuations)."""
    mags = np.ones(20) * 0.9
    engs = np.ones(20) * -1.5
    _, _, susc, _ = calculate_thermodynamics(mags=mags, engs=engs, T=1.0, L=10)
    assert pytest.approx(susc) == 0.0


def test_specific_heat_zero_variance():
    """Specific heat should be zero for constant energy (no fluctuations)."""
    mags = np.ones(20) * 0.5
    engs = np.ones(20) * -1.0
    _, _, _, spec_h = calculate_thermodynamics(mags=mags, engs=engs, T=1.0, L=10)
    assert pytest.approx(spec_h) == 0.0


def test_susceptibility_scales_with_n():
    """Susceptibility chi = N * Var(M) / T should scale with lattice size N = L^2."""
    mags = np.array([0.0, 1.0])
    engs = np.array([-1.0, -1.0])
    T, L = 1.0, 10
    _, _, susc, _ = calculate_thermodynamics(mags=mags, engs=engs, T=T, L=L)
    expected = (L**2) * np.var(mags) / T
    assert pytest.approx(susc) == expected


def test_thermodynamics_invalid_inputs():
    """calculate_thermodynamics should raise ValueError for invalid T or L."""
    mags = np.ones(10)
    engs = np.ones(10)
    with pytest.raises(ValueError):
        calculate_thermodynamics(mags=mags, engs=engs, T=0, L=10)
    with pytest.raises(ValueError):
        calculate_thermodynamics(mags=mags, engs=engs, T=-1, L=10)
    with pytest.raises(ValueError):
        calculate_thermodynamics(mags=mags, engs=engs, T=1, L=0)
    with pytest.raises(ValueError):
        calculate_thermodynamics(mags=mags, engs=engs, T=1, L=-5)


def test_calculate_thermodynamics_invalid_params():
    """Verify specific ValueError messages for invalid inputs."""
    mags = np.array([0.5])
    engs = np.array([-1.0])
    with pytest.raises(ValueError, match="T must be positive"):
        calculate_thermodynamics(mags=mags, engs=engs, T=0.0, L=10)
    with pytest.raises(ValueError, match="L must be a positive integer"):
        calculate_thermodynamics(mags=mags, engs=engs, T=1.0, L=0)


# ---- calculate_entropy ----


def test_entropy_constant_cv():
    """For constant Cv, total entropy change equals Cv * ln(T_max / T_min)."""
    T = np.linspace(1.0, 10.0, 200)
    Cv = np.full_like(T, 3.0)
    S = calculate_entropy(temperatures=T, specific_heat=Cv)
    expected_delta = 3.0 * np.log(10.0 / 1.0)
    assert pytest.approx(S[-1] - S[0], rel=1e-3) == expected_delta


def test_entropy_linear_cv():
    """For Cv = a*T, integral of a*T/T dT = a*(T_max - T_min)."""
    T = np.linspace(1.0, 5.0, 500)
    a = 2.0
    Cv = a * T
    S = calculate_entropy(temperatures=T, specific_heat=Cv)
    expected_delta = a * (5.0 - 1.0)
    assert pytest.approx(S[-1] - S[0], rel=1e-3) == expected_delta


def test_entropy_monotonicity():
    """Positive Cv implies non-decreasing entropy with temperature."""
    T = np.linspace(0.5, 5.0, 100)
    Cv = np.abs(np.sin(T)) + 0.1
    S = calculate_entropy(temperatures=T, specific_heat=Cv)
    diffs = np.diff(S)
    assert np.all(diffs >= -1e-14)


def test_entropy_s_ref_shift():
    """Changing s_ref shifts all entropy values by a constant."""
    T = np.linspace(1.0, 5.0, 50)
    Cv = np.ones_like(T)
    S_base = calculate_entropy(temperatures=T, specific_heat=Cv, s_ref=0.0)
    S_shifted = calculate_entropy(temperatures=T, specific_heat=Cv, s_ref=np.log(6))
    np.testing.assert_allclose(S_shifted - S_base, np.log(6), atol=1e-14)


def test_entropy_unsorted_temperatures():
    """Entropy should be correct even when temperatures are not sorted."""
    T_sorted = np.linspace(1.0, 5.0, 50)
    Cv_sorted = 2.0 * T_sorted
    S_sorted = calculate_entropy(temperatures=T_sorted, specific_heat=Cv_sorted)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(T_sorted))
    S_perm = calculate_entropy(temperatures=T_sorted[perm], specific_heat=Cv_sorted[perm])
    np.testing.assert_allclose(S_perm, S_sorted[perm], atol=1e-12)


def test_entropy_invalid_too_few_points():
    """Should raise ValueError with fewer than 2 temperature points."""
    with pytest.raises(ValueError, match='at least 2'):
        calculate_entropy(temperatures=np.array([1.0]), specific_heat=np.array([1.0]))


def test_entropy_invalid_negative_temperature():
    """Should raise ValueError for non-positive temperatures."""
    with pytest.raises(ValueError, match='positive'):
        calculate_entropy(
            temperatures=np.array([-1.0, 1.0]),
            specific_heat=np.array([1.0, 1.0]),
        )


def test_entropy_invalid_shape_mismatch():
    """Should raise ValueError when array lengths differ."""
    with pytest.raises(ValueError, match='same shape'):
        calculate_entropy(
            temperatures=np.array([1.0, 2.0, 3.0]),
            specific_heat=np.array([1.0, 2.0]),
        )


def test_summarize_entropy_observable_multi_replicate_finite_error():
    """Entropy summary should provide finite uncertainty with >=2 replicates."""
    temperatures = np.linspace(1.0, 3.0, 5)
    cv_samples = np.array(
        [
            [1.0, 1.2, 0.9],
            [1.1, 1.3, 1.0],
            [1.2, 1.4, 1.1],
            [1.3, 1.5, 1.2],
            [1.4, 1.6, 1.3],
        ],
        dtype=float,
    )
    summary = summarize_entropy_observable(
        temperatures=temperatures,
        specific_heat_samples=cv_samples,
    )
    assert np.asarray(summary['value']).shape == (5,)
    assert np.asarray(summary['err']).shape == (5,)
    assert np.all(np.isfinite(np.asarray(summary['err'])))


def test_summarize_entropy_observable_single_replicate_has_nan_error():
    """Single-replicate entropy summary should keep uncertainty undefined."""
    temperatures = np.linspace(1.0, 2.0, 4)
    cv_samples = np.array([[1.0], [1.1], [1.2], [1.3]], dtype=float)
    summary = summarize_entropy_observable(
        temperatures=temperatures,
        specific_heat_samples=cv_samples,
    )
    assert np.all(np.isnan(np.asarray(summary['err'])))
    assert np.all(np.isnan(np.asarray(summary['ci_low'])))
    assert np.all(np.isnan(np.asarray(summary['ci_high'])))


def test_summarize_entropy_observable_single_replicate_with_cv_err_is_finite():
    """Single-replicate entropy can propagate finite uncertainty from Cv errors."""
    temperatures = np.linspace(1.0, 2.0, 4)
    cv_samples = np.array([[1.0], [1.1], [1.2], [1.3]], dtype=float)
    cv_err = np.array([0.05, 0.05, 0.05, 0.05], dtype=float)
    summary = summarize_entropy_observable(
        temperatures=temperatures,
        specific_heat_samples=cv_samples,
        specific_heat_err=cv_err,
        method=UNCERTAINTY_METHOD_BOOTSTRAP,
    )
    assert np.all(np.isfinite(np.asarray(summary['err'])))
    assert np.all(np.isfinite(np.asarray(summary['ci_low'])))
    assert np.all(np.isfinite(np.asarray(summary['ci_high'])))


def test_summarize_entropy_observable_bootstrap_returns_finite_intervals():
    """Bootstrap entropy summary should provide finite intervals with >=2 replicates."""
    temperatures = np.linspace(1.0, 3.0, 8)
    rng = np.random.default_rng(0)
    cv_samples = 1.0 + 0.1 * rng.normal(size=(8, 6))
    summary = summarize_entropy_observable(
        temperatures=temperatures,
        specific_heat_samples=cv_samples,
        method=UNCERTAINTY_METHOD_BOOTSTRAP,
        bootstrap_resamples=200,
        rng_seed=1,
    )
    assert np.all(np.isfinite(np.asarray(summary['ci_low'])))
    assert np.all(np.isfinite(np.asarray(summary['ci_high'])))


def test_summarize_entropy_observable_more_replicates_reduces_mean_err():
    """Average entropy uncertainty should shrink with more replicate curves."""
    temperatures = np.linspace(1.0, 3.0, 12)
    rng = np.random.default_rng(42)
    full_samples = 1.0 + 0.15 * rng.normal(size=(12, 16))

    small = summarize_entropy_observable(
        temperatures=temperatures,
        specific_heat_samples=full_samples[:, :4],
        method=UNCERTAINTY_METHOD_BOOTSTRAP,
        bootstrap_resamples=200,
        rng_seed=7,
    )
    large = summarize_entropy_observable(
        temperatures=temperatures,
        specific_heat_samples=full_samples,
        method=UNCERTAINTY_METHOD_BOOTSTRAP,
        bootstrap_resamples=200,
        rng_seed=7,
    )
    err_small = np.nanmean(np.asarray(small['err']))
    err_large = np.nanmean(np.asarray(large['err']))
    assert err_large < err_small


def test_summarize_asymmetric_replicate_uncertainty_is_asymmetric_for_skewed_data():
    """Asymmetric replicate summary should produce unequal lower and upper errors."""
    samples = np.array([1.0, 1.1, 1.2, 1.5, 3.0], dtype=float)
    summary = summarize_asymmetric_replicate_uncertainty(samples=samples)
    assert summary['ci_high'] > summary['value'] > summary['ci_low']
    assert summary['err_high'] > summary['err_low']


# ---- calculate_autocorr ----


def test_autocorr_normalization():
    """C(0) must be 1.0 for any non-constant input."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    C_t, _ = calculate_autocorr(time_series=x)
    assert pytest.approx(C_t[0], abs=1e-12) == 1.0


def test_autocorr_white_noise_tau():
    """White noise has tau_int close to 0.5 (effective sample count = N)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000)
    _, tau = calculate_autocorr(time_series=x)
    assert 0.3 < tau < 1.5


def test_autocorr_ar1_tau():
    """AR(1) process with rho=0.7 should give tau_int close to rho/(1-rho) = 2.33."""
    rng = np.random.default_rng(42)
    rho = 0.7
    N = 20000
    x = np.empty(N)
    x[0] = 0.0
    noise = rng.standard_normal(N)
    for i in range(1, N):
        x[i] = rho * x[i - 1] + noise[i] * np.sqrt(1 - rho**2)
    _, tau = calculate_autocorr(time_series=x)
    expected = rho / (1.0 - rho)
    assert abs(tau - expected) / expected < 0.3


def test_autocorr_high_correlation_larger_tau():
    """A strongly correlated series must have a larger tau_int than white noise."""
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(3000)
    white = rng.standard_normal(3000)
    rho = 0.9
    corr = np.empty(3000)
    corr[0] = 0.0
    for i in range(1, 3000):
        corr[i] = rho * corr[i - 1] + noise[i] * np.sqrt(1 - rho**2)
    _, tau_white = calculate_autocorr(time_series=white)
    _, tau_corr = calculate_autocorr(time_series=corr)
    assert tau_corr > tau_white


def test_autocorr_invalid_too_few_points():
    """Should raise ValueError with fewer than 3 elements."""
    with pytest.raises(ValueError, match='at least 3'):
        calculate_autocorr(time_series=np.array([1.0, 2.0]))


def test_autocorr_invalid_zero_variance():
    """Should raise the dedicated zero-variance analysis error for constant input."""
    with pytest.raises(ZeroVarianceAutocorrelationError, match='zero variance'):
        calculate_autocorr(time_series=np.ones(100))


# ---- uncertainty helpers ----


def test_estimate_effective_sample_size_ar1_smaller_than_n():
    """Correlated AR(1) samples should have N_eff less than raw N."""
    rng = np.random.default_rng(123)
    rho = 0.8
    n = 4000
    x = np.empty(n)
    x[0] = 0.0
    noise = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + noise[i] * np.sqrt(1.0 - rho**2)

    n_eff = estimate_effective_sample_size(time_series=x)
    assert np.isfinite(n_eff)
    assert 1.0 <= n_eff < float(n)


def test_blocking_error_iid_agrees_with_naive():
    """Blocking stderr should stay close to naive stderr for IID noise."""
    rng = np.random.default_rng(12)
    x = rng.standard_normal(8192)
    block = blocking_error(time_series=x)
    ratio = block['stderr'] / block['stderr_naive']
    assert 0.7 <= ratio <= 1.5


def test_summarize_primary_observable_zero_variance_policy():
    """Constant series should return zero error and undefined tau diagnostics."""
    x = np.ones(256)
    summary = summarize_primary_observable(time_series=x)
    assert summary['value'] == pytest.approx(1.0)
    assert summary['err'] == pytest.approx(0.0)
    assert np.isnan(summary['tau_int'])
    assert np.isnan(summary['n_eff'])


def test_summarize_derived_observable_chi_blocking():
    """chi summary should produce finite value and non-negative uncertainty."""
    rng = np.random.default_rng(21)
    mags = rng.normal(loc=0.0, scale=0.3, size=4096)
    summary = summarize_derived_observable(
        magnetization_series=mags,
        temperature=2.0,
        L=16,
        observable='chi',
    )
    assert summary['value'] >= 0.0
    assert summary['err'] >= 0.0
    assert summary['ci_high'] >= summary['ci_low']


def test_summarize_derived_observable_cv_bootstrap():
    """Cv bootstrap mode should provide finite interval and error."""
    rng = np.random.default_rng(7)
    engs = rng.normal(loc=-1.0, scale=0.2, size=4096)
    summary = summarize_derived_observable(
        energy_series=engs,
        temperature=2.5,
        L=12,
        observable='cv',
        method=UNCERTAINTY_METHOD_BOOTSTRAP,
        bootstrap_resamples=200,
    )
    assert summary['value'] >= 0.0
    assert summary['err'] >= 0.0
    assert summary['ci_high'] >= summary['ci_low']


def test_summarize_replicate_samples_2d_nan_safe():
    """Replicate summary should be NaN-safe and keep output vectorized."""
    samples = np.array(
        [
            [1.0, 1.2, np.nan, 0.8],
            [2.0, np.nan, 2.2, 1.8],
        ]
    )
    summary = summarize_replicate_samples(samples=samples)
    assert isinstance(summary['value'], np.ndarray)
    assert isinstance(summary['ci_low'], np.ndarray)
    assert isinstance(summary['ci_high'], np.ndarray)
    assert summary['samples'] == pytest.approx(4.0)


def test_summarize_seed_ensemble_single_seed_matches_within_seed_error():
    """With one seed, hierarchical error should equal within-seed error."""
    summary = summarize_seed_ensemble(
        values=np.array([1.5]),
        within_seed_errors=np.array([0.2]),
    )
    assert summary['value'] == pytest.approx(1.5)
    assert summary['err'] == pytest.approx(0.2)
    assert summary['samples'] == pytest.approx(1.0)


def test_summarize_seed_ensemble_multi_seed_has_between_component():
    """Multi-seed aggregation should include between-seed spread in total error."""
    summary = summarize_seed_ensemble(
        values=np.array([1.0, 1.5, 2.0]),
        within_seed_errors=np.array([0.1, 0.1, 0.1]),
    )
    assert summary['value'] == pytest.approx(1.5)
    assert summary['between_seed_component'] > 0.0
    assert summary['err'] >= summary['within_seed_component']


# ---- get_averaged_correlation ----
















# ---- radial_average_sk ----








# ---- pair_correlation_x ----








# ---- compute_kinetics_metrics ----






# ---- Observables: helicity, structure factor, energy ----




















# ---- power_fit ----


def test_power_fit():
    """power_fit should extract exponent and prefactor for perfect power law."""
    t = np.array([10, 100, 1000], dtype=float)
    y = 2.0 * t**0.5
    mask = np.ones_like(t, dtype=bool)
    exp, pre = power_fit(t_arr=t, y_arr=y, mask=mask)
    assert exp == pytest.approx(0.5)
    assert pre == pytest.approx(2.0)


def test_power_fit_none_on_insufficient_data():
    """power_fit should return None if not enough valid data points."""
    t = np.array([1, 2], dtype=float)
    y = np.array([1, 2], dtype=float)
    mask = np.ones_like(t, dtype=bool)
    exp, pre = power_fit(t_arr=t, y_arr=y, mask=mask)
    assert exp is None
    assert pre is None


def test_estimate_relaxation_time_two_start_basic():
    """Verify relaxation time estimation on a synthetic converging sequence."""
    n = 300
    # Ordered starts at 1, decays to 0.5
    trace_ordered = 0.5 + 0.5 * np.exp(-np.arange(n) / 20.0)
    # Random starts at 0, grows to 0.5
    trace_random = 0.5 * (1.0 - np.exp(-np.arange(n) / 20.0))

    # Add some noise
    rng = np.random.default_rng(42)
    trace_ordered += rng.normal(0, 0.01, n)
    trace_random += rng.normal(0, 0.01, n)

    tau = estimate_relaxation_time_two_start(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        smooth_window=10,
        dwell_window=10,
    )

    # Should converge well before the end
    assert 0 < tau < n
    # For these params, should be roughly a few decay constants
    assert 20 < tau < 250


def test_estimate_relaxation_time_two_start_short():
    """Should return 0 for very short traces."""
    r = np.array([0.1, 0.2])
    o = np.array([0.9, 0.8])
    assert estimate_relaxation_time_two_start(r, o) == 0


def test_estimate_relaxation_time_two_start_no_convergence():
    """Should return trace length if the two traces never meet."""
    n = 50
    r = np.zeros(n)
    o = np.ones(n)
    # Use small k to ensure they don't accidentally converge
    tau = estimate_relaxation_time_two_start(r, o, k=0.1)
    assert tau == n


def test_estimate_relaxation_time_two_start_sigma_floor():
    """Sigma floor must prevent false positives when one trace is nearly flat.

    The old pooled-band criterion would declare convergence here because the
    near-zero variance of the ordered trace widens the pooled band enough to
    swallow the random trace.  The mutual cross-band criterion should not fire
    because the random trace (tail mean ~0.0) is far from the ordered tail
    mean (~1.0), and sigma_floor keeps the ordered band narrow.
    """
    n = 200
    rng = np.random.default_rng(99)
    # Ordered trace: stays near 1.0 with tiny noise (nearly flat, very low variance)
    trace_ordered = np.ones(n) + rng.normal(0, 1e-4, n)
    # Random trace: stays near 0.0 with small noise
    trace_random = np.zeros(n) + rng.normal(0, 0.05, n)

    tau = estimate_relaxation_time_two_start(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        k=2.0,
        smooth_window=10,
        dwell_window=10,
        sigma_floor=0.02,
    )
    # These traces never meet; should return full trace length
    assert tau == n


def test_detect_quasi_steady_stuck_true_for_separated_plateaus():
    """Should flag stuck when both traces are flat but separated."""


    n = 240
    rng = np.random.default_rng(7)
    trace_ordered = np.ones(n) + rng.normal(0, 1e-3, n)
    trace_random = np.full(n, 0.55) + rng.normal(0, 1e-3, n)

    assert _detect_quasi_steady_stuck(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        k=1.0,
        smooth_window=20,
        qs_sigma_threshold=0.02,
        sigma_floor=0.02,
    )


def test_detect_quasi_steady_stuck_false_for_converged_plateaus():
    """Should not flag stuck when flat traces are already mutually inside bands."""


    n = 240
    rng = np.random.default_rng(13)
    trace_ordered = np.full(n, 0.82) + rng.normal(0, 1e-3, n)
    trace_random = np.full(n, 0.83) + rng.normal(0, 1e-3, n)

    assert not _detect_quasi_steady_stuck(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        k=1.0,
        smooth_window=20,
        qs_sigma_threshold=0.02,
        sigma_floor=0.02,
    )


def test_detect_quasi_steady_stuck_false_for_high_variance_traces():
    """Should not flag stuck when both traces have converged with large critical fluctuations.

    At or near Tc both starts settle to the same equilibrium mean with large
    thermal fluctuations.  When means agree the cross-band check always passes
    regardless of variance, so stuck is correctly not declared.
    """


    n = 240
    rng = np.random.default_rng(19)
    # Both traces at the same equilibrium mean - models two-start convergence near Tc
    trace_ordered = 0.65 + rng.normal(0, 0.15, n)
    trace_random  = 0.65 + rng.normal(0, 0.15, n)

    assert not _detect_quasi_steady_stuck(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        k=1.0,
        smooth_window=20,
        qs_sigma_threshold=0.02,
        sigma_floor=0.02,
    )


def test_detect_quasi_steady_stuck_large_L_domain_fluctuations():
    """Should detect stuck even when random-trace variance exceeds flat threshold.

    At large L the random-start trace in a multi-domain metastable state has
    domain-wall fluctuations sig_r ~ 0.07, which would exceed the default flat
    threshold=0.05 and block detection under the old both-traces guard.
    The new ordered-trace-only guard must still fire because sig_o ~ 0.
    """


    n = 300
    rng = np.random.default_rng(41)
    # Ordered trace: deep ordered phase, near-zero variance
    trace_ordered = np.ones(n) + rng.normal(0, 1e-4, n)
    # Random trace: stuck in multi-domain state with domain-wall fluctuations
    # sig_r ~ 0.07, well above the flat threshold of 0.05
    trace_random = np.full(n, 0.15) + rng.normal(0, 0.07, n)

    assert _detect_quasi_steady_stuck(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        k=1.0,
        smooth_window=20,
        qs_sigma_threshold=0.05,
        sigma_floor=0.02,
    )


def test_detect_quasi_steady_stuck_l_scaling_prevents_false_positive():
    """L-scaling must maintain safety margin between threshold and thermal variance.

    At large L (e.g. L=128) the ordered trace in the disordered phase has
    thermal variance sig_o ~ c/L, which may fall below the flat threshold=0.05.
    With L-scaling the effective threshold shrinks proportionally, keeping a
    constant safety margin.  The cross-band check then correctly reports
    not-stuck because both traces share the same equilibrium mean.
    """


    n = 300
    rng = np.random.default_rng(53)
    L = 128
    # Thermalized disordered phase at large L: both traces near same low mean
    sigma_thermal = 0.07 * 64 / L  # ~ 0.035
    mu_eq = 0.045
    trace_ordered = mu_eq + rng.normal(0, sigma_thermal, n)
    trace_random  = mu_eq + rng.normal(0, sigma_thermal, n)

    # Without L-scaling: cross-band check still finds gap ~ 0, returns not-stuck.
    assert not _detect_quasi_steady_stuck(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        k=1.0,
        smooth_window=20,
        qs_sigma_threshold=0.05,
        sigma_floor=0.02,
    )

    # With L-scaling: same result, gap still zero.
    assert not _detect_quasi_steady_stuck(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        k=1.0,
        smooth_window=20,
        qs_sigma_threshold=0.05,
        sigma_floor=0.02,
        lattice_size=L,
    )


def test_detect_quasi_steady_stuck_l_scaling_true_for_low_t_stuck():
    """L-scaled threshold must still detect deep-ordered-phase stuck state.

    Even with L-scaling reducing the effective threshold, the ordered trace at
    low T has sig_o ~ 0, which always satisfies the guard.
    The large gap then triggers the cross-band failure.
    """


    n = 300
    rng = np.random.default_rng(67)
    L = 64
    trace_ordered = np.ones(n) + rng.normal(0, 1e-4, n)
    # Random trace stuck with domain-wall fluctuations sig_r > flat threshold
    trace_random = np.full(n, 0.12) + rng.normal(0, 0.07, n)

    assert _detect_quasi_steady_stuck(
        trace_random=trace_random,
        trace_ordered=trace_ordered,
        k=1.0,
        smooth_window=20,
        qs_sigma_threshold=0.05,
        sigma_floor=0.02,
        lattice_size=L,
    )


class TestPhysicsHelpersValidation:
    """Verify error handling for invalid inputs in physics_helpers.py."""

    def test_validate_confidence_errors(self) -> None:
        """Should raise ValueError for confidence outside (0, 1)."""

        with pytest.raises(ValueError, match='confidence must satisfy'):
            _validate_confidence(confidence=0.0)
        with pytest.raises(ValueError, match='confidence must satisfy'):
            _validate_confidence(confidence=1.0)
        with pytest.raises(ValueError, match='confidence must satisfy'):
            _validate_confidence(confidence=1.5)

    def test_as_1d_float_array_errors(self) -> None:
        """Should raise ValueError for non-1D or too small arrays."""

        # 2D array
        with pytest.raises(ValueError, match='must be 1-D'):
            _as_1d_float_array(time_series=np.zeros((2, 2)), name='test')
        # Too small
        with pytest.raises(ValueError, match='at least 2 elements'):
            _as_1d_float_array(time_series=np.array([1.0]), name='test')

    def test_blocking_error_zero_variance(self) -> None:
        """blocking_error should handle zero-variance input by returning 0 error."""
        x = np.ones(100)
        res = blocking_error(time_series=x)
        assert res['stderr'] == 0.0
        assert np.isnan(res['tau_int_from_blocking'])

    def test_summarize_primary_observable_empty_or_nan(self) -> None:
        """summarize_primary_observable should handle all-NaN series."""
        x = np.array([np.nan, np.nan, np.nan])
        # It calls _as_1d_float_array which is fine, but np.nanmean etc will return NaN
        summary = summarize_primary_observable(time_series=x)
        assert np.isnan(summary['value'])
        assert np.isnan(summary['err'])
