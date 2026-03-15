# mypy: disable-error-code=no-untyped-def

"""
Unit tests for physics-related utility functions in utils/physics_helpers.py.
Covers thermodynamic averages, entropy, autocorrelation, spatial diagnostics,
kinetics metrics, and power-law fitting.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation
from utils.physics_helpers import (
    calculate_autocorr,
    calculate_entropy,
    calculate_thermodynamics,
    compute_kinetics_metrics,
    get_averaged_correlation,
    pair_correlation_x,
    power_fit,
    radial_average_sk,
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
    """Should raise ValueError for constant input."""
    with pytest.raises(ValueError, match='zero variance'):
        calculate_autocorr(time_series=np.ones(100))


# ---- get_averaged_correlation ----


def test_get_averaged_correlation():
    """Verify correlation averaging over multiple steps."""
    size = 10
    sim = IsingSimulation(size=size, temp=1.0, seed=42)
    r, g_avg = get_averaged_correlation(
        sim=sim, total_steps=10, sample_interval=5
    )
    assert len(r) == size // 2
    assert len(g_avg) == size // 2
    assert g_avg[0] == pytest.approx(1.0)
    assert sim.steps == 10


def test_get_averaged_correlation_returns_two_arrays(ising_sim):
    """Should return a tuple of two numpy arrays."""
    r, G_r = get_averaged_correlation(sim=ising_sim, total_steps=20, sample_interval=5)
    assert isinstance(r, np.ndarray)
    assert isinstance(G_r, np.ndarray)


def test_get_averaged_correlation_output_lengths_match(ising_sim):
    """r and G_r must have the same length."""
    r, G_r = get_averaged_correlation(sim=ising_sim, total_steps=20, sample_interval=5)
    assert len(r) == len(G_r)


def test_normalization_at_zero(ising_sim):
    """G(0) should be 1 (normalized by definition)."""
    r, G_r = get_averaged_correlation(sim=ising_sim, total_steps=20, sample_interval=5)
    assert pytest.approx(G_r[0], abs=1e-5) == 1.0


def test_output_length_is_half_lattice(ising_sim):
    """Length of r should be size // 2 (radial profile up to half the box)."""
    r, G_r = get_averaged_correlation(sim=ising_sim, total_steps=10, sample_interval=5)
    assert len(r) == ising_sim.size // 2


def test_get_averaged_correlation_invalid_params():
    """Verify validation for get_averaged_correlation parameters."""
    sim = IsingSimulation(size=10, temp=1.0)
    with pytest.raises(ValueError, match="sample_interval must be >= 1"):
        get_averaged_correlation(sim=sim, total_steps=10, sample_interval=0)
    with pytest.raises(ValueError, match="total_steps must be non-negative"):
        get_averaged_correlation(sim=sim, total_steps=-1, sample_interval=1)


def test_get_averaged_correlation_invalid_inputs(ising_sim):
    """get_averaged_correlation should raise ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        get_averaged_correlation(sim=ising_sim, total_steps=-1, sample_interval=1)
    with pytest.raises(ValueError):
        get_averaged_correlation(sim=ising_sim, total_steps=10, sample_interval=0)


# ---- radial_average_sk ----


def test_radial_average_sk_ising():
    """Verify radial structure factor for an Ising lattice."""
    size = 16
    spins = np.ones((size, size))
    k, sk = radial_average_sk(spins=spins)
    assert len(k) == size // 2 + 1
    assert sk[0] == pytest.approx(float(size * size))
    assert np.all(sk[1:] < 1e-10)


def test_radial_average_sk_vector():
    """Verify radial structure factor for vector spins (XY/Clock)."""
    size = 16
    spins = np.zeros((size, size, 2))
    spins[..., 0] = 1.0
    k, sk = radial_average_sk(spins=spins)
    assert sk[0] == pytest.approx(float(size * size))
    assert np.all(sk[1:] < 1e-10)


def test_radial_average_sk_returns_arrays():
    """radial_average_sk should return matching-length arrays."""
    spins = np.ones((32, 32))
    k, sk = radial_average_sk(spins=spins)
    assert isinstance(k, np.ndarray)
    assert isinstance(sk, np.ndarray)
    assert len(k) == len(sk)
    assert len(k) > 0


# ---- pair_correlation_x ----


def test_pair_correlation_x():
    """Uniform -> G=1, checkerboard -> G=(-1)^r."""
    size = 16
    spins_uniform = np.ones((size, size))
    r, g = pair_correlation_x(spins=spins_uniform)
    np.testing.assert_allclose(g, 1.0)

    iy, ix = np.indices((size, size))
    spins_checker = np.where((ix + iy) % 2 == 0, 1, -1).astype(np.float64)
    r_c, g_c = pair_correlation_x(spins=spins_checker)
    expected = np.array([(-1) ** rr for rr in range(len(g_c))], dtype=np.float64)
    np.testing.assert_allclose(g_c, expected, atol=1e-10)


def test_pair_correlation_x_ising():
    """pair_correlation_x should work for scalar spins."""
    spins = np.ones((32, 32))
    r, g = pair_correlation_x(spins=spins)
    assert len(r) == len(g)
    assert g[0] == pytest.approx(1.0)


def test_pair_correlation_x_xy():
    """pair_correlation_x should work for vector spins."""
    spins = np.ones((32, 32, 2))
    r, g = pair_correlation_x(spins=spins)
    assert len(r) == len(g)
    assert g[0] == pytest.approx(1.0)


# ---- compute_kinetics_metrics ----


def test_compute_kinetics_metrics():
    """Returns R_sk and xi; xi = L/2 for uniform state (maximum)."""
    size = 16
    sim = IsingSimulation(size=size, temp=1.0)
    sim.spins = np.ones((size, size), dtype=np.int8)
    metrics = compute_kinetics_metrics(sim=sim)
    assert 'R_sk' in metrics
    assert 'xi' in metrics
    assert isinstance(metrics['R_sk'], float)
    assert isinstance(metrics['xi'], float)


def test_compute_kinetics_metrics_xy():
    """compute_kinetics_metrics should return R_sk and xi for XY."""
    sim = XYSimulation(size=16, temp=1.0)
    metrics = compute_kinetics_metrics(sim=sim)
    assert 'R_sk' in metrics
    assert 'xi' in metrics


# ---- Observables: helicity, structure factor, energy ----


def test_xy_helicity_data():
    """Verify helicity data calculation for XY model."""
    size = 10
    sim = XYSimulation(size=size, temp=1.0)
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0
    cos_sum, sin_sum = sim._get_helicity_data()
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)


def test_clock_helicity_data():
    """Verify helicity data calculation for Clock model."""
    size = 10
    sim = ClockSimulation(size=size, temp=0.5, q=6)
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0
    cos_sum, sin_sum = sim._get_helicity_data()
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)


def test_discrete_clock_helicity_data():
    """Verify helicity data calculation for DiscreteClock model."""
    size = 10
    sim = DiscreteClockSimulation(size=size, temp=0.5, q=6)
    sim.spins = np.zeros((size, size), dtype=np.int32)
    cos_sum, sin_sum = sim._get_helicity_data()
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)


def test_discrete_clock_vorticity_detection():
    """Verify vorticity detection via vector conversion for DiscreteClock."""
    size = 4
    q = 4
    sim = DiscreteClockSimulation(size=size, temp=1.0, q=q)
    sim.spins = np.zeros((size, size), dtype=np.int32)
    sim.spins[0, 0] = 0
    sim.spins[0, 1] = 1
    sim.spins[1, 1] = 2
    sim.spins[1, 0] = 3
    vort = sim._calculate_vorticity()
    assert vort[0, 0] == 1.0


def test_base_structure_factor():
    """Verify structure factor via base class method."""
    size = 8
    sim = IsingSimulation(size=size, temp=1.0)
    sim.spins = np.ones((size, size), dtype=np.int8)
    sf = sim._calculate_structure_factor()
    assert sf.shape == (size, size)
    center = size // 2
    assert sf[center, center] == pytest.approx(float(size * size))


def test_xy_structure_factor_squared():
    """Verify XY structure factor squared (unshifted)."""
    size = 8
    sim = XYSimulation(size=size, temp=1.0)
    sf = sim._get_structure_factor_squared_unshifted()
    assert sf.shape == (size, size)


def test_clock_structure_factor_squared():
    """Verify Clock structure factor squared (unshifted)."""
    size = 8
    sim = ClockSimulation(size=size, temp=1.0, q=6)
    sf = sim._get_structure_factor_squared_unshifted()
    assert sf.shape == (size, size)


def test_discrete_clock_structure_factor_squared():
    """Verify DiscreteClock structure factor squared (unshifted)."""
    size = 8
    sim = DiscreteClockSimulation(size=size, temp=1.0, q=6)
    sf = sim._get_structure_factor_squared_unshifted()
    assert sf.shape == (size, size)


def test_clock_energy_calc():
    """Verify Clock energy returns a float."""
    size = 8
    sim = ClockSimulation(size=size, temp=1.0, q=6)
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0
    e = sim._get_energy()
    assert isinstance(e, float)


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
