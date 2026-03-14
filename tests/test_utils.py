# mypy: disable-error-code=no-untyped-def

"""
Unit tests for utility functions in utils/physics_helpers.py and utils/system_helpers.py.
"""
from __future__ import annotations

import os
import shutil

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')  # Non-interactive backend - no display required
import matplotlib.pyplot as plt

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
from utils.system_helpers import (
    adaptive_equilibrate,
    ensure_results_dir,
    parallel_sweep,
    plot_ordering_evolution,
    plot_ordering_kinetics,
    plot_temperature_sweep,
    save_plot,
    setup_logging,
)


def _square_worker(x: int) -> int:
    """Module-level worker function for parallel_sweep test."""
    return x * x


@pytest.fixture
def test_results_dir():
    """Fixture to manage a clean test results directory."""
    test_dir = 'test_results'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture
def ising_sim():
    """Fixture for a small Ising simulation."""
    return IsingSimulation(size=10, temp=2.0)


# Tests for calculate_thermodynamics
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
    """Susceptibility (chi = N * Var(M) / T) should scale with lattice size N = L^2."""
    mags = np.array([0.0, 1.0])  # variance = 0.25
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


# Tests for get_averaged_correlation
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


def test_get_averaged_correlation_invalid_inputs(ising_sim):
    """get_averaged_correlation should raise ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        get_averaged_correlation(sim=ising_sim, total_steps=-1, sample_interval=1)
    with pytest.raises(ValueError):
        get_averaged_correlation(sim=ising_sim, total_steps=10, sample_interval=0)


# New Tests for Kinetics Analysis
def test_radial_average_sk_ising():
    """radial_average_sk should work for scalar spins."""
    spins = np.ones((32, 32))
    k, sk = radial_average_sk(spins=spins)
    assert isinstance(k, np.ndarray)
    assert isinstance(sk, np.ndarray)
    assert len(k) == len(sk)
    assert len(k) > 0


def test_radial_average_sk_xy():
    """radial_average_sk should work for vector spins."""
    spins = np.ones((32, 32, 2))
    k, sk = radial_average_sk(spins=spins)
    assert len(k) == len(sk)
    assert len(k) > 0


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


def test_compute_kinetics_metrics_ising():
    """compute_kinetics_metrics should return R_sk and xi for Ising."""
    sim = IsingSimulation(size=16, temp=1.0)
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


# Tests for system utility functions
def test_setup_logging():
    """setup_logging should return a logger instance."""
    logger = setup_logging()
    assert logger.name == 'vibespin'


def test_ensure_results_dir(test_results_dir):
    """ensure_results_dir should create a directory if it does not exist."""
    sub_dir = os.path.join(test_results_dir, 'subfolder')
    path = ensure_results_dir(directory=sub_dir)
    assert path == sub_dir
    assert os.path.isdir(sub_dir)


def test_save_plot(test_results_dir):
    """save_plot should create a PNG file in the specified directory."""
    plt.figure()
    plt.plot([1, 2], [1, 2])
    filename = 'test_plot.png'
    save_plot(filename=filename, directory=test_results_dir)
    assert os.path.isfile(os.path.join(test_results_dir, filename))
    plt.close()


def test_parallel_sweep():
    """parallel_sweep should correctly execute a worker function across parameters."""
    params = [1, 2, 3, 4, 5]
    results = parallel_sweep(worker_func=_square_worker, params=params, num_processes=2)
    assert results == [1, 4, 9, 16, 25]


# Tests for high-level plotting functions
def test_plot_temperature_sweep_runs():
    """plot_temperature_sweep smoke test."""
    temps = np.array([1.0, 2.0])
    data = np.array([0.5, 0.5])
    data_seq = data.tolist()
    plot_temperature_sweep(
        temperatures=temps,
        avg_m=data_seq,
        avg_e=data_seq,
        susc=data_seq,
        spec_h=data_seq,
        title='Test',
        filename='_ts.png',
        directory='test_results',
    )
    plt.close('all')


def test_plot_ordering_kinetics_runs():
    """plot_ordering_kinetics smoke test."""
    t = np.array([1, 10, 100])
    r = np.array([1, 2, 3])
    exponents: dict[str, float | None] = {'R_sk': 0.5, 'xi': 0.5, 'third': -1.0}
    prefactors: dict[str, float | None] = {'R_sk': 1.0, 'xi': 1.0, 'third': 1.0}
    mask = np.ones_like(t, dtype=bool)
    plot_ordering_kinetics(
        t=t,
        R_sk=r,
        R_xi=r,
        third_metric=r,
        third_metric_label='Test Metric',
        exponents=exponents,
        prefactors=prefactors,
        fit_mask=mask,
        title='Title',
        filename='_kin.png',
        directory='test_results',
    )
    plt.close('all')


def test_plot_ordering_evolution_runs():
    """plot_ordering_evolution smoke test."""
    targets = [1, 10]
    snapshots = [np.ones((16, 16)), np.ones((16, 16))]
    gr_data = [(np.arange(8), np.ones(8)), (np.arange(8), np.ones(8))]
    plot_ordering_evolution(
        targets=targets,
        snapshots=snapshots,
        gr_data=gr_data,
        vorticity_data=None,
        title='Title',
        filename='_evol.png',
        directory='test_results',
        is_vector=False,
    )
    plt.close('all')


# ---------- Tests for calculate_entropy ----------

def test_entropy_constant_cv():
    """For constant Cv, total entropy change equals Cv * ln(T_max / T_min)."""
    T = np.linspace(1.0, 10.0, 200)
    Cv = np.full_like(T, 3.0)
    S = calculate_entropy(temperatures=T, specific_heat=Cv)
    # S(T_max) = 0.0 (default s_ref), S(T_min) = -Cv * ln(T_max/T_min)
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
    Cv = np.abs(np.sin(T)) + 0.1  # strictly positive
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


# ---------- Tests for calculate_autocorr ----------

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
    expected = rho / (1.0 - rho)  # ~2.33
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


# ---------------------------------------------------------------------------
# adaptive_equilibrate tests
# ---------------------------------------------------------------------------


class _StubSim:
    """Minimal simulation stub whose run() always returns a constant magnetization."""

    def equilibrate(self, *, n_steps: int) -> None:
        pass

    def run(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        # Constant array triggers zero-variance ValueError in calculate_autocorr.
        return np.ones(n_steps), np.zeros(n_steps)


def test_adaptive_equilibrate_ordered_phase():
    """Zero-variance probe (ordered phase) should return without crashing."""
    sim = _StubSim()
    total = adaptive_equilibrate(sim, min_steps=100, probe_steps=50)
    assert total >= 100


def test_adaptive_equilibrate_high_temp():
    """At T=10 (far above Tc) tau_int ~ 0.5, so one probe_steps=500 satisfies factor=5."""
    sim = IsingSimulation(size=8, temp=10.0)
    total = adaptive_equilibrate(sim, min_steps=200, probe_steps=500, factor=5.0)
    # One probe suffices: 500 >> 5 * 0.5 = 2.5, so total = min_steps + probe_steps.
    assert total == 200 + 500
