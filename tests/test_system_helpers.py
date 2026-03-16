# mypy: disable-error-code=no-untyped-def

"""
Unit tests for system-related utility functions in utils/system_helpers.py.
Covers logging, directory management, plotting, parallel execution,
and adaptive equilibration.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.ising_model import IsingSimulation
from utils.exceptions import ZeroVarianceAutocorrelationError
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
def temp_dir():
    """Fixture to create and cleanup a temporary directory."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


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


# ---- Directory and logging ----


def test_ensure_results_dir(temp_dir):
    """Verify directory creation."""
    target = os.path.join(temp_dir, 'test_results')
    path = ensure_results_dir(directory=target)
    assert os.path.exists(target)
    assert path == target


def test_ensure_results_dir_subdir(test_results_dir):
    """ensure_results_dir should create a subdirectory if it does not exist."""
    sub_dir = os.path.join(test_results_dir, 'subfolder')
    path = ensure_results_dir(directory=sub_dir)
    assert path == sub_dir
    assert os.path.isdir(sub_dir)


def test_setup_logging():
    """Verify logger configuration."""
    logger = setup_logging()
    assert logger.name == 'vibespin'
    assert len(logger.handlers) >= 1


# ---- File I/O ----


def test_save_plot(temp_dir):
    """Verify saving a matplotlib plot."""
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filename = 'test_plot.png'
    save_plot(filename=filename, directory=temp_dir)
    assert os.path.exists(os.path.join(temp_dir, filename))
    plt.close()


def test_save_plot_creates_file(test_results_dir):
    """save_plot should create a PNG file in the specified directory."""
    plt.figure()
    plt.plot([1, 2], [1, 2])
    filename = 'test_plot.png'
    save_plot(filename=filename, directory=test_results_dir)
    assert os.path.isfile(os.path.join(test_results_dir, filename))
    plt.close()


# ---- Parallel execution ----


def test_parallel_sweep_mocked():
    """Verify parallel execution with mocked Pool."""
    def worker(x):
        return x * 2

    params = [1, 2, 3]
    with patch('utils.system_helpers.Pool') as mock_pool:
        mock_instance = mock_pool.return_value.__enter__.return_value
        mock_instance.imap.return_value = [2, 4, 6]
        results = parallel_sweep(worker_func=worker, params=params, num_processes=1)
        assert results == [2, 4, 6]
        mock_instance.imap.assert_called_once()


def test_parallel_sweep():
    """parallel_sweep should correctly execute a worker function across parameters."""
    params = [1, 2, 3, 4, 5]
    results = parallel_sweep(worker_func=_square_worker, params=params, num_processes=2)
    assert results == [1, 4, 9, 16, 25]


# ---- Plotting smoke tests ----


def test_plot_temperature_sweep(temp_dir):
    """Verify temperature sweep plotting runs without error."""
    temps = np.array([1.0, 2.0])
    data = [0.5, 0.6]

    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            title='Test',
            filename='test.png',
            directory=temp_dir,
        )
        mock_save.assert_called_once_with(filename='test.png', directory=temp_dir)
    plt.close('all')


def test_plot_temperature_sweep_runs():
    """plot_temperature_sweep smoke test with real save."""
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


def test_plot_ordering_kinetics(temp_dir):
    """Verify ordering kinetics plotting runs without error."""
    t = np.array([1, 10])
    R = np.array([1, 2])
    exponents = {'R_sk': 0.5, 'xi': 0.5}
    prefactors = {'R_sk': 1.0, 'xi': 1.0}
    mask = np.ones_like(t, dtype=bool)

    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_ordering_kinetics(
            t=t,
            R_sk=R,
            R_xi=R,
            third_metric=R,
            third_metric_label='Test',
            exponents=exponents,
            prefactors=prefactors,
            fit_mask=mask,
            title='Test',
            filename='test_kin.png',
            directory=temp_dir,
        )
        mock_save.assert_called_once()
    plt.close('all')


def test_plot_ordering_kinetics_runs():
    """plot_ordering_kinetics smoke test with real save."""
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


def test_plot_ordering_evolution(temp_dir):
    """Verify ordering evolution plotting runs without error."""
    targets = [1, 10]
    snapshots = [np.ones((16, 16)), np.ones((16, 16))]
    gr_data = [(np.arange(8), np.ones(8)), (np.arange(8), np.ones(8))]

    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_ordering_evolution(
            targets=targets,
            snapshots=snapshots,
            gr_data=gr_data,
            vorticity_data=None,
            title='Test',
            filename='test_evol.png',
            directory=temp_dir,
            is_vector=False,
        )
        mock_save.assert_called_once()
    plt.close('all')


def test_plot_ordering_evolution_runs():
    """plot_ordering_evolution smoke test with real save."""
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


# ---- adaptive_equilibrate ----


class _StubSim:
    """Minimal simulation stub whose run() always returns a constant magnetization."""

    def equilibrate(self, *, n_steps: int) -> None:
        pass

    def run(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
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
    assert total == 200 + 500


def test_adaptive_equilibrate_invalid_probe_steps():
    """Probe windows shorter than the autocorrelation precondition must fail fast."""
    sim = _StubSim()
    with pytest.raises(ValueError, match='probe_steps must be >= 3'):
        adaptive_equilibrate(sim, min_steps=10, probe_steps=2)


def test_adaptive_equilibrate_only_swallows_zero_variance():
    """Invalid autocorrelation inputs must still surface to the caller."""

    class _ShortRunSim:
        def equilibrate(self, *, n_steps: int) -> None:
            pass

        def run(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
            return np.ones(2), np.zeros(2)

    sim = _ShortRunSim()
    with pytest.raises(ValueError, match='at least 3'):
        adaptive_equilibrate(sim, min_steps=5, probe_steps=3)


def test_zero_variance_error_is_runtime_error():
    """Zero-variance analysis failures should not be treated as argument validation errors."""
    error = ZeroVarianceAutocorrelationError('zero variance')
    assert isinstance(error, RuntimeError)
