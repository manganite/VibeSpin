from __future__ import annotations
"""
Unit tests for utility functions in utils/physics_helpers.py and utils/system_helpers.py.
"""

import os
import shutil

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')  # Non-interactive backend — no display required
import matplotlib.pyplot as plt

from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation
from utils.physics_helpers import (
    calculate_thermodynamics,
    compute_kinetics_metrics,
    get_averaged_correlation,
    pair_correlation_x,
    power_fit,
    radial_average_sk,
)
from utils.system_helpers import (
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
    plot_temperature_sweep(
        temperatures=temps,
        avg_m=data,
        avg_e=data,
        susc=data,
        spec_h=data,
        title='Test',
        filename='_ts.png',
        directory='test_results',
    )
    plt.close('all')


def test_plot_ordering_kinetics_runs():
    """plot_ordering_kinetics smoke test."""
    t = np.array([1, 10, 100])
    r = np.array([1, 2, 3])
    exponents = {'R_sk': 0.5, 'xi': 0.5, 'third': -1.0}
    prefactors = {'R_sk': 1.0, 'xi': 1.0, 'third': 1.0}
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
