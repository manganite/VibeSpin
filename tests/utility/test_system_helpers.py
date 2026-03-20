# mypy: disable-error-code=no-untyped-def

"""
Unit tests for system-related utility functions in utils/system_helpers.py.
Covers logging, directory management, plotting, parallel execution,
and adaptive equilibration.
"""
from __future__ import annotations

import logging
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


def test_setup_logging_with_file(temp_dir):
    """setup_logging should create the parent directory and a file handler when requested."""
    log_path = os.path.join(temp_dir, 'logs', 'vibespin.log')
    logger = logging.getLogger('vibespin')
    original_handlers = list(logger.handlers)
    for handler in original_handlers:
        logger.removeHandler(handler)

    try:
        logger = setup_logging(log_file=log_path)

        file_handlers = [h for h in logger.handlers if h.__class__.__name__ == 'FileHandler']
        assert os.path.isdir(os.path.dirname(log_path))
        assert file_handlers
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            logger.addHandler(handler)


def test_ensure_results_dir_empty_string():
    """ensure_results_dir should be a no-op for empty directory strings."""
    with patch('utils.system_helpers.os.makedirs') as mock_makedirs:
        path = ensure_results_dir(directory='')
    assert path == ''
    mock_makedirs.assert_not_called()


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


def test_adaptive_equilibrate_invalid_min_steps():
    """Negative min_steps should raise ValueError."""
    sim = _StubSim()
    with pytest.raises(ValueError, match='min_steps must be non-negative'):
        adaptive_equilibrate(sim, min_steps=-1, probe_steps=10)


def test_adaptive_equilibrate_invalid_factor():
    """Non-positive factor should raise ValueError."""
    sim = _StubSim()
    with pytest.raises(ValueError, match='factor must be positive'):
        adaptive_equilibrate(sim, min_steps=10, probe_steps=10, factor=0.0)


def test_adaptive_equilibrate_invalid_max_steps():
    """max_steps smaller than min_steps should raise ValueError."""
    sim = _StubSim()
    with pytest.raises(ValueError, match='max_steps must be >= min_steps'):
        adaptive_equilibrate(sim, min_steps=20, probe_steps=10, max_steps=10)


def test_adaptive_equilibrate_warns_on_max_steps(caplog):
    """If the criterion is never met before max_steps, the function should warn and return."""

    class _RunSim:
        def equilibrate(self, *, n_steps: int) -> None:
            pass

        def run(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
            return np.linspace(0.0, 1.0, n_steps), np.zeros(n_steps)

    caplog.set_level('WARNING', logger='vibespin')
    with patch('utils.physics_helpers.calculate_autocorr', return_value=(np.array([0.0]), 1e9)):
        total = adaptive_equilibrate(
            _RunSim(),
            min_steps=10,
            probe_steps=5,
            factor=2.0,
            max_steps=15,
        )

    assert total == 15
    assert 'reached max_steps=15' in caplog.text


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


def test_convergence_equilibrate_warns_on_max_steps(caplog):
    """Two-start equilibration should warn when no convergence is detected before max_steps."""

    class _ConvergenceStub:
        def run(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
            return np.ones(n_steps), np.zeros(n_steps)

    caplog.set_level('WARNING', logger='vibespin')
    with patch('utils.physics_helpers.estimate_relaxation_time_two_start', return_value=1000):
        from utils.system_helpers import convergence_equilibrate

        total = convergence_equilibrate(
            _ConvergenceStub(),
            _ConvergenceStub(),
            chunk_size=50,
            max_steps=100,
        )

    assert total == 100
    assert 'without convergence' in caplog.text


def test_plot_temperature_sweep_with_entropy_only(temp_dir):
    """Optional panel layout should handle entropy-only inputs."""
    temps = np.array([1.0, 2.0])
    data = np.array([0.5, 0.6])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            entropy=np.array([0.1, 0.2]),
            tau_int=None,
            title='Entropy only',
            filename='entropy_only.png',
            directory=temp_dir,
        )
        assert mock_save.call_count == 2
        mock_save.assert_any_call(filename='entropy_only.png', directory=temp_dir)
        mock_save.assert_any_call(filename='entropy_only_diagnostics.png', directory=temp_dir)
    plt.close('all')


def test_plot_temperature_sweep_with_tau_only(temp_dir):
    """Optional panel layout should handle tau-only inputs."""
    temps = np.array([1.0, 2.0])
    data = np.array([0.5, 0.6])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            entropy=None,
            tau_int=np.array([2.0, 3.0]),
            title='Tau only',
            filename='tau_only.png',
            directory=temp_dir,
        )
        assert mock_save.call_count == 2
        mock_save.assert_any_call(filename='tau_only.png', directory=temp_dir)
        mock_save.assert_any_call(filename='tau_only_diagnostics.png', directory=temp_dir)
    plt.close('all')


def test_plot_temperature_sweep_with_tau_invalid_band(temp_dir):
    """Tau panel should tolerate and mark invalid CI points without crashing."""
    temps = np.array([1.0, 2.0, 3.0])
    data = np.array([0.5, 0.6, 0.7])
    tau = np.array([2.0, 2.5, 3.0])
    tau_lo = np.array([1.5, np.nan, 2.4])
    tau_hi = np.array([2.7, np.nan, 3.7])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            entropy=None,
            tau_int=tau,
            tau_int_ci_low=tau_lo,
            tau_int_ci_high=tau_hi,
            title='Tau invalid band',
            filename='tau_invalid_band.png',
            directory=temp_dir,
            mark_invalid_uncertainty=True,
        )
        assert mock_save.call_count == 2
        mock_save.assert_any_call(filename='tau_invalid_band.png', directory=temp_dir)
        mock_save.assert_any_call(
            filename='tau_invalid_band_diagnostics.png',
            directory=temp_dir,
        )
    plt.close('all')


def test_plot_temperature_sweep_with_entropy_band_saves_diagnostics(temp_dir):
    """Entropy confidence bands should be rendered in the companion diagnostics figure."""
    temps = np.array([1.0, 2.0, 3.0])
    data = np.array([0.5, 0.6, 0.7])
    entropy = np.array([0.1, 0.2, 0.35])
    entropy_lo = np.array([0.08, 0.16, 0.3])
    entropy_hi = np.array([0.13, 0.24, 0.4])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            entropy=entropy,
            entropy_ci_low=entropy_lo,
            entropy_ci_high=entropy_hi,
            title='Entropy band',
            filename='entropy_band.png',
            directory=temp_dir,
        )
        assert mock_save.call_count == 2
        mock_save.assert_any_call(filename='entropy_band.png', directory=temp_dir)
        mock_save.assert_any_call(filename='entropy_band_diagnostics.png', directory=temp_dir)
    plt.close('all')


def test_plot_temperature_sweep_with_transition_guides(temp_dir):
    """Transition markers and window overlays should render without affecting output contract."""
    temps = np.array([1.0, 1.5, 2.0, 2.5])
    data = np.array([0.2, 0.5, 0.9, 0.6])
    tau = np.array([1.0, 1.3, 2.5, 1.8])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            tau_int=tau,
            transition_temperatures={r'$T_{\\chi}$': 2.0, r'$T_{C_v}$': 1.9},
            transition_window=(1.8, 2.1),
            annotate_peaks=True,
            title='Transition guides',
            filename='transition_guides.png',
            directory=temp_dir,
        )
        assert mock_save.call_count == 2
        mock_save.assert_any_call(filename='transition_guides.png', directory=temp_dir)
        mock_save.assert_any_call(
            filename='transition_guides_diagnostics.png',
            directory=temp_dir,
        )
    plt.close('all')


def test_plot_temperature_sweep_with_low_effective_sample_flags(temp_dir):
    """Low-effective-sample flags should be rendered as reliability markers."""
    temps = np.array([1.0, 1.5, 2.0, 2.5])
    data = np.array([0.2, 0.4, 0.8, 0.5])
    low_eff = np.array([0.0, 1.0, 0.0, 1.0])
    tau = np.array([1.0, 1.2, 1.8, 1.4])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            tau_int=tau,
            low_effective_sample_flag=low_eff,
            title='Low effective samples',
            filename='low_effective_samples.png',
            directory=temp_dir,
        )
        assert mock_save.call_count == 2
        mock_save.assert_any_call(filename='low_effective_samples.png', directory=temp_dir)
        mock_save.assert_any_call(
            filename='low_effective_samples_diagnostics.png',
            directory=temp_dir,
        )
    plt.close('all')


def test_plot_temperature_sweep_with_metadata_and_quality_summary(temp_dir):
    """Diagnostics header metadata and quality summary should render safely."""
    temps = np.array([1.0, 1.5, 2.0, 2.5])
    data = np.array([0.2, 0.4, 0.8, 0.5])
    tau = np.array([1.0, 1.2, 1.8, 1.4])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            tau_int=tau,
            diagnostics_note='diag note',
            run_metadata_note='L=32, n_seeds=8, conf=0.68',
            quality_summary={
                'total_points': 4,
                'well_conditioned_count': 2,
                'low_effective_count': 1,
                'unstable_interval_count': 1,
                'undefined_count': 0,
            },
            title='Metadata summary',
            filename='metadata_summary.png',
            directory=temp_dir,
        )
        assert mock_save.call_count == 2
        mock_save.assert_any_call(filename='metadata_summary.png', directory=temp_dir)
        mock_save.assert_any_call(
            filename='metadata_summary_diagnostics.png',
            directory=temp_dir,
        )
    plt.close('all')


def test_plot_temperature_sweep_with_entropy_reference(temp_dir):
    """Entropy reference lines should render in diagnostics entropy panel."""
    temps = np.array([1.0, 1.5, 2.0, 2.5])
    data = np.array([0.2, 0.4, 0.8, 0.5])
    entropy = np.array([0.1, 0.2, 0.35, 0.45])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_temperature_sweep(
            temperatures=temps,
            avg_m=data,
            avg_e=data,
            susc=data,
            spec_h=data,
            entropy=entropy,
            entropy_reference=(r'$S_{\\mathrm{ref}}=\\ln q$', float(np.log(6.0))),
            title='Entropy reference',
            filename='entropy_reference.png',
            directory=temp_dir,
        )
        assert mock_save.call_count == 2
        mock_save.assert_any_call(filename='entropy_reference.png', directory=temp_dir)
        mock_save.assert_any_call(
            filename='entropy_reference_diagnostics.png',
            directory=temp_dir,
        )
    plt.close('all')


def test_plot_ordering_kinetics_without_third_metric(temp_dir):
    """Second panel should be disabled cleanly when no third metric is provided."""
    t = np.array([1, 10])
    R = np.array([1.0, 2.0])
    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_ordering_kinetics(
            t=t,
            R_sk=R,
            R_xi=R,
            third_metric=None,
            third_metric_label=None,
            exponents={'R_sk': 0.5, 'xi': 0.5, 'third': None},
            prefactors={'R_sk': 1.0, 'xi': 1.0, 'third': None},
            fit_mask=np.ones_like(t, dtype=bool),
            title='No third metric',
            filename='no_third_metric.png',
            directory=temp_dir,
        )
        mock_save.assert_called_once()
    plt.close('all')


def test_plot_ordering_evolution_vector_vorticity_with_xi_marker(temp_dir):
    """Vector + vorticity branch should render and annotate xi when G(r) crosses 1/e."""
    targets = [1, 10]
    spins = np.ones((16, 16, 2))
    snapshots = [spins, spins]
    vort = np.zeros((16, 16))
    vort[0, 0] = 1.0
    vorticity_data = [vort, vort]
    r = np.arange(8)
    g = np.array([1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.08, 0.05])
    gr_data = [(r, g), (r, g)]

    with patch('utils.system_helpers.save_plot') as mock_save:
        plot_ordering_evolution(
            targets=targets,
            snapshots=snapshots,
            gr_data=gr_data,
            vorticity_data=vorticity_data,
            title='Vector with vorticity',
            filename='vector_vorticity.png',
            directory=temp_dir,
            is_vector=True,
        )
        mock_save.assert_called_once()
    plt.close('all')


def test_zero_variance_error_is_runtime_error():
    """Zero-variance analysis failures should not be treated as argument validation errors."""
    error = ZeroVarianceAutocorrelationError('zero variance')
    assert isinstance(error, RuntimeError)
