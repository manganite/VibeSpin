"""
Unit tests for system-related utility functions in utils/system_helpers.py.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from utils.system_helpers import (
    ensure_results_dir,
    parallel_sweep,
    plot_ordering_evolution,
    plot_ordering_kinetics,
    plot_temperature_sweep,
    save_plot,
    setup_logging,
)


@pytest.fixture
def temp_dir():
    """Fixture to create and cleanup a temporary directory."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_ensure_results_dir(temp_dir):
    """Verify directory creation."""
    target = os.path.join(temp_dir, 'test_results')
    path = ensure_results_dir(directory=target)
    assert os.path.exists(target)
    assert path == target


def test_setup_logging():
    """Verify logger configuration."""
    logger = setup_logging()
    assert logger.name == 'vibespin'
    assert len(logger.handlers) >= 1


def test_save_plot(temp_dir):
    """Verify saving a matplotlib plot."""
    plt.figure()
    plt.plot([0, 1], [0, 1])
    filename = 'test_plot.png'
    save_plot(filename=filename, directory=temp_dir)
    assert os.path.exists(os.path.join(temp_dir, filename))
    plt.close()


def test_parallel_sweep():
    """Verify parallel execution (mocked)."""
    def worker(x):
        return x * 2

    params = [1, 2, 3]
    # Mocking Pool to avoid actual multiprocessing during tests
    with patch('utils.system_helpers.Pool') as mock_pool:
        mock_instance = mock_pool.return_value.__enter__.return_value
        mock_instance.imap.return_value = [2, 4, 6]

        results = parallel_sweep(worker_func=worker, params=params, num_processes=1)

        assert results == [2, 4, 6]
        mock_instance.imap.assert_called_once()


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
            directory=temp_dir
        )
        mock_save.assert_called_once_with(filename='test.png', directory=temp_dir)
    plt.close('all')


def test_plot_ordering_kinetics(temp_dir):
    """Verify ordering kinetics plotting runs without error."""
    t = np.array([1, 10])
    R = np.array([1, 2])
    exponents = {'R_sk': 0.5, 'xi': 0.5}
    prefactors = {'R_sk': 1.0, 'xi': 1.0}
    fit_mask = np.array([True, True])

    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_ordering_kinetics(
            t=t, R_sk=R, R_xi=R, third_metric=None, third_metric_label=None,
            exponents=exponents, prefactors=prefactors, fit_mask=fit_mask,
            title='Test', filename='test.png', directory=temp_dir
        )
        mock_save.assert_called_once()
    plt.close('all')


def test_plot_ordering_evolution(temp_dir):
    """Verify ordering evolution plotting runs without error."""
    targets = [0, 10]
    snapshots = [np.ones((10, 10)), np.ones((10, 10))]
    gr_data = [(np.arange(5), np.ones(5)), (np.arange(5), np.ones(5))]

    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_ordering_evolution(
            targets=targets, snapshots=snapshots, gr_data=gr_data,
            vorticity_data=None, title='Test', filename='test.png',
            directory=temp_dir, is_vector=False
        )
        mock_save.assert_called_once()
    plt.close('all')
