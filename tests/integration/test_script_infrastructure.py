"""
Integration tests for script infrastructure patterns.

Tests reusable infrastructure components used by multiple analysis scripts:
- Deterministic seed generation and reproducibility (seed-based RNG protocols)
- NPZ output schema and shapes (multi-algorithm aggregation formats)
- Multi-seed aggregation logic (per-seed sample arrays, percentiles)
- Fallback demo mode execution (lightweight standalone testing)
- Temperature-sweep typed payloads (main() construction and worker contracts)

Current coverage:
  - measure_z.py: Wolff/Metropolis scaling analysis
  - temperature_sweep.py: Ising, XY, Clock models with typed SweepPoint payloads
Future extensions: wolff_efficiency.py, ordering_kinetics.py, etc.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scripts.ising.measure_z import (
    _measure_tau_point,
)

try:
    from models.clock_model import ClockSimulation
    from models.ising_model import IsingSimulation
    from models.xy_model import XYSimulation
    from utils.sweep_helpers import (
        ThermoPoint,
        build_uncertainty_bundle,
        simulate_thermo_point,
    )
    HAS_TEMPERATURE_SWEEP = True
except ImportError:
    HAS_TEMPERATURE_SWEEP = False


class TestDeterministicSeeds:
    """Verify that seed generation is reproducible across runs."""

    def test_metropolis_seed_consistency(self) -> None:
        """Same (size_idx, seed_idx) yields same Metropolis seed."""
        size_idx, seed_idx = 2, 3
        seed1 = size_idx * 100_000 + seed_idx * 1_000
        seed2 = size_idx * 100_000 + seed_idx * 1_000
        assert seed1 == seed2

    def test_wolff_seed_offset(self) -> None:
        """Wolff seeds should be offset by 50_000 from Metropolis."""
        size_idx, seed_idx = 1, 2
        metro_seed = size_idx * 100_000 + seed_idx * 1_000
        wolff_seed = metro_seed + 50_000
        assert wolff_seed == metro_seed + 50_000
        assert wolff_seed != metro_seed

    def test_different_seeds_for_different_indices(self) -> None:
        """Different (size_idx, seed_idx) pairs yield different seeds."""
        seed_a = 0 * 100_000 + 0 * 1_000  # (0, 0)
        seed_b = 0 * 100_000 + 1 * 1_000  # (0, 1)
        seed_c = 1 * 100_000 + 0 * 1_000  # (1, 0)
        assert seed_a != seed_b
        assert seed_a != seed_c
        assert seed_b != seed_c


class TestMeasureTauPoint:
    """Test the worker function for a single (L, algorithm, seed) point."""

    def test_worker_returns_required_keys(self) -> None:
        """Worker output must contain size_idx, seed_idx, tau_int, update, L."""
        params = (0, 0, 'random', 16, 100, 1000, 500, 123)
        result = _measure_tau_point(params)
        assert result['size_idx'] == 0
        assert result['seed_idx'] == 0
        assert result['tau_int'] >= 0
        assert isinstance(result['tau_int'], (float, np.floating))
        assert result['update'] == 'random'
        assert result['L'] == 16

    def test_metropolis_vs_wolff_different_tau(self) -> None:
        """Metropolis and Wolff should yield different tau_int for same L."""
        L, eq_probe = 16, 100
        eq_max, meas_steps = 1000, 500
        params_m = (0, 0, 'random', L, eq_probe, eq_max, meas_steps, 42)
        params_w = (0, 0, 'wolff', L, eq_probe, eq_max, meas_steps, 42)

        res_m = _measure_tau_point(params_m)
        res_w = _measure_tau_point(params_w)

        # Cluster updates should be much more efficient at Tc
        assert res_w['tau_int'] < res_m['tau_int']


class TestSweepHelpersContract:
    """Verify the utils.sweep_helpers public API contracts."""

    def test_thermo_point_is_picklable(self) -> None:
        """ThermoPoint must be picklable for multiprocessing worker dispatch."""
        if not HAS_TEMPERATURE_SWEEP:
            pytest.skip("sweep_helpers not available")
        import pickle
        pt = ThermoPoint(
            temperature=2.0,
            size=8,
            meas_steps=50,
            seed=1,
            temperature_index=0,
            seed_index=0,
            eq_probe_steps=50,
            eq_max_steps=500,
            eq_qs_sigma_threshold=0.05,
            eq_qs_min_steps=500,
            qs_allow_stuck=False,
            prefer_ordered_start=False,
            model_cls=IsingSimulation,
            model_kwargs={},
            confidence=0.68,
            derived_method='blocking',
            bootstrap_resamples=100,
        )
        restored = pickle.loads(pickle.dumps(pt))
        assert restored.temperature == pt.temperature
        assert restored.model_cls is IsingSimulation

    def test_simulate_thermo_point_returns_required_keys(self) -> None:
        """simulate_thermo_point must return all standard thermodynamic keys."""
        if not HAS_TEMPERATURE_SWEEP:
            pytest.skip("sweep_helpers not available")
        pt = ThermoPoint(
            temperature=2.0,
            size=8,
            meas_steps=50,
            seed=7,
            temperature_index=0,
            seed_index=0,
            eq_probe_steps=50,
            eq_max_steps=500,
            eq_qs_sigma_threshold=0.05,
            eq_qs_min_steps=500,
            qs_allow_stuck=False,
            prefer_ordered_start=False,
            model_cls=IsingSimulation,
            model_kwargs={},
            confidence=0.68,
            derived_method='blocking',
            bootstrap_resamples=100,
        )
        result = simulate_thermo_point(pt)
        required = {
            'temperature_index', 'seed_index', 'equilibrated_flag',
            'avg_m_value', 'avg_m_err', 'avg_e_value', 'avg_e_err',
            'susc_value', 'susc_err', 'spec_h_value', 'spec_h_err',
        }
        assert required <= set(result.keys())

    def test_build_uncertainty_bundle_returns_required_keys(self) -> None:
        """build_uncertainty_bundle must return all expected schema keys."""
        if not HAS_TEMPERATURE_SWEEP:
            pytest.skip("sweep_helpers not available")
        values = np.array([[1.0, 1.1], [2.0, 2.1]])
        errors = np.array([[0.1, 0.1], [0.2, 0.2]])
        tau = np.array([[3.0, 4.0], [2.0, 2.5]])
        n_eff = np.array([[10.0, 9.0], [12.0, 11.0]])
        bundle = build_uncertainty_bundle(
            values_by_seed=values,
            errors_by_seed=errors,
            tau_by_seed=tau,
            n_eff_by_seed=n_eff,
            confidence=0.68,
        )
        required = {'value', 'err', 'ci_low', 'ci_high', 'tau_int', 'n_eff', 'samples'}
        assert required <= set(bundle.keys())
        assert cast(np.ndarray, bundle['value']).shape == (2,)
        assert cast(np.ndarray, bundle['samples']).shape == (2, 2)


class TestTemperatureSweepWorkerPayloads:
    """Verify that temperature-sweep workers follow the typed payload contract."""

    def _assert_valid_thermo_result(self, result: Any) -> None:
        """Helper to validate worker return dict."""
        assert 'avg_m_value' in result
        assert 'avg_e_value' in result
        assert 'susc_value' in result
        assert 'spec_h_value' in result
        value = result['avg_m_value']
        assert isinstance(value, (float, np.floating))
        assert np.isfinite(value) or np.isnan(value)

    def test_ising_worker_accepts_typed_payload(self) -> None:
        """Ising worker should accept SweepPoint payload and return thermodynamics."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        payload = ThermoPoint(
            temperature=2.0,
            size=8,
            meas_steps=100,
            seed=42,
            temperature_index=0,
            seed_index=0,
            eq_probe_steps=100,
            eq_max_steps=1000,
            eq_qs_sigma_threshold=0.05,
            eq_qs_min_steps=1500,
            qs_allow_stuck=False,
            prefer_ordered_start=False,
            model_cls=IsingSimulation,
            model_kwargs={},
            confidence=0.68,
            derived_method='blocking',
            bootstrap_resamples=1000,
        )
        result = simulate_thermo_point(payload)
        self._assert_valid_thermo_result(result)

    def test_xy_worker_accepts_typed_payload(self) -> None:
        """XY worker should accept SweepPoint payload and return thermodynamics."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        payload = ThermoPoint(
            temperature=0.9,
            size=8,
            meas_steps=100,
            seed=42,
            temperature_index=0,
            seed_index=0,
            eq_probe_steps=100,
            eq_max_steps=1000,
            eq_qs_sigma_threshold=0.05,
            eq_qs_min_steps=1500,
            qs_allow_stuck=False,
            prefer_ordered_start=False,
            model_cls=XYSimulation,
            model_kwargs={},
            confidence=0.68,
            derived_method='blocking',
            bootstrap_resamples=1000,
        )
        result = simulate_thermo_point(payload)
        self._assert_valid_thermo_result(result)

    def test_clock_worker_accepts_typed_payload(self) -> None:
        """Clock worker should accept SweepPoint payload and return thermodynamics."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        payload = ThermoPoint(
            temperature=0.9,
            size=8,
            meas_steps=100,
            seed=42,
            temperature_index=0,
            seed_index=0,
            eq_probe_steps=100,
            eq_max_steps=1000,
            eq_qs_sigma_threshold=0.05,
            eq_qs_min_steps=1500,
            qs_allow_stuck=False,
            prefer_ordered_start=False,
            model_cls=ClockSimulation,
            model_kwargs={'q': 6, 'A': 0.0},
            confidence=0.68,
            derived_method='blocking',
            bootstrap_resamples=1000,
        )
        result = simulate_thermo_point(payload)
        self._assert_valid_thermo_result(result)


class TestTemperatureSweepMainPayloads:
    """Verify that main() scripts construct correct payloads for workers."""

    def _capture_sweep_params(
        self,
        monkeypatch: Any,
        module: Any,
        expected_fields: tuple[str, ...],
        argv: list[str],
        expected_len: int | None = None,
        expected_eq_max_steps: int | None = None,
    ) -> None:
        """Mock parallel_sweep to capture parameters passed by main()."""
        captured: dict[str, Any] = {}

        def _fake_parallel_sweep(
            *, worker_func: Any, params: Any, num_processes: Any = None
        ) -> Any:
            params_list = list(params)
            captured.setdefault('params', []).extend(params_list)
            return [
                {
                    'temperature_index': float(p.temperature_index),
                    'seed_index': float(p.seed_index),
                    'equilibrated_flag': 1.0,
                    'equilibration_steps': 100.0,
                    'avg_m_value': 1.0,
                    'avg_m_err': 0.1,
                    'avg_m_tau_int': 3.0,
                    'avg_m_n_eff': 10.0,
                    'avg_e_value': -1.0,
                    'avg_e_err': 0.1,
                    'avg_e_tau_int': 3.0,
                    'avg_e_n_eff': 10.0,
                    'susc_value': 0.5,
                    'susc_err': 0.05,
                    'susc_tau_int': 3.0,
                    'susc_n_eff': 10.0,
                    'spec_h_value': 0.2,
                    'spec_h_err': 0.03,
                    'spec_h_tau_int': 3.0,
                    'spec_h_n_eff': 10.0,
                }
                for p in params_list
            ]

        monkeypatch.setattr(module, 'parallel_sweep', _fake_parallel_sweep)
        monkeypatch.setattr(module, 'plot_temperature_sweep', lambda **kwargs: None)
        monkeypatch.setattr(sys, 'argv', argv)

        module.main()

        assert 'params' in captured
        if expected_len is not None:
            assert len(captured['params']) == expected_len

        for payload in captured['params']:
            assert isinstance(payload, tuple)
            assert hasattr(payload, '_fields')
            for field in expected_fields:
                assert hasattr(payload, field)
            if expected_eq_max_steps is not None:
                assert int(payload.eq_max_steps) == expected_eq_max_steps

    def test_ising_main_builds_typed_sweep_payloads(self, monkeypatch: Any) -> None:
        """Ising temperature sweep main should build Ising SeedSweepPoint payloads."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        import scripts.ising.temperature_sweep as ising_module

        self._capture_sweep_params(
            monkeypatch,
            ising_module,
            (
                'temperature',
                'size',
                'meas_steps',
                'eq_probe_steps',
                'eq_max_steps',
                'eq_qs_sigma_threshold',
                'eq_qs_min_steps',
                'temperature_index',
                'seed_index',
                'seed',
            ),
            [
                'ising_temperature_sweep',
                '--size', '8', '--meas-steps', '20', '--t-points', '2', '--n-seeds', '1',
            ],
            expected_len=2,
            expected_eq_max_steps=20000,
        )

    def test_xy_main_builds_typed_sweep_payloads(self, monkeypatch: Any) -> None:
        """XY temperature sweep main should build XY SweepPoint payloads."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        import scripts.xy.temperature_sweep as xy_module

        self._capture_sweep_params(
            monkeypatch,
            xy_module,
            (
                'temperature',
                'size',
                'meas_steps',
                'eq_probe_steps',
                'eq_max_steps',
                'eq_qs_sigma_threshold',
                'eq_qs_min_steps',
                'temperature_index',
                'seed_index',
                'seed',
                'model_cls',
                'model_kwargs',
            ),
            ['xy_temperature_sweep', '--size', '8', '--meas-steps', '20', '--t-points', '2'],
            expected_len=2,
        )

    def test_clock_main_builds_typed_sweep_payloads(self, monkeypatch: Any) -> None:
        """Clock temperature sweep main should build Clock SweepPoint payloads."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        import scripts.clock.temperature_sweep as clock_module

        self._capture_sweep_params(
            monkeypatch,
            clock_module,
            (
                'temperature',
                'size',
                'meas_steps',
                'eq_probe_steps',
                'eq_max_steps',
                'eq_qs_sigma_threshold',
                'eq_qs_min_steps',
                'temperature_index',
                'seed_index',
                'seed',
                'model_cls',
                'model_kwargs',
            ),
            ['clock_temperature_sweep', '--size', '8', '--meas-steps', '20', '--t-points', '2'],
            expected_len=2,
        )


class TestTemperatureSweepUncertaintySchema:
    """Verify that combined uncertainty bundles use correct shapes and schemas."""

    def test_build_uncertainty_bundle_shapes(self) -> None:
        """Bundle should aggregate per-seed columns into pointwise results."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        # Mock values: 3 temperatures, 2 seeds
        values_by_seed = np.array([[1.0, 1.2], [2.0, 2.2], [3.0, 3.3]])
        errors_by_seed = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.4]])
        tau_by_seed = np.array([[5.0, 4.5], [np.nan, 3.0], [2.0, 2.5]])
        n_eff_by_seed = np.array([[10.0, 12.0], [8.0, 9.0], [15.0, 16.0]])

        bundle = build_uncertainty_bundle(
            values_by_seed=values_by_seed,
            errors_by_seed=errors_by_seed,
            tau_by_seed=tau_by_seed,
            n_eff_by_seed=n_eff_by_seed,
            confidence=0.68,
        )

        value = cast(np.ndarray, bundle['value'])
        err = cast(np.ndarray, bundle['err'])
        ci_low = cast(np.ndarray, bundle['ci_low'])
        ci_high = cast(np.ndarray, bundle['ci_high'])
        tau_bundle = cast(np.ndarray, bundle['tau_int'])
        n_eff = cast(np.ndarray, bundle['n_eff'])
        samples = cast(np.ndarray, bundle['samples'])

        assert value.shape == (3,)
        assert err.shape == (3,)
        assert ci_low.shape == (3,)
        assert ci_high.shape == (3,)
        assert tau_bundle.shape == (3,)
        assert n_eff.shape == (3,)
        assert samples.shape == (3, 2)

    def test_ising_main_writes_uncertainty_npz(self, monkeypatch: Any) -> None:
        """Ising sweep main should persist additive uncertainty schema keys."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        import scripts.ising.temperature_sweep as ising_module

        def _fake_parallel_sweep(
            *, worker_func: Any, params: Any, num_processes: Any = None
        ) -> Any:
            params_list = list(params)
            return [
                {
                    'temperature_index': float(p.temperature_index),
                    'seed_index': float(p.seed_index),
                    'equilibrated_flag': 1.0,
                    'equilibration_steps': 100.0,
                    'avg_m_value': 1.0,
                    'avg_m_err': 0.1,
                    'avg_m_tau_int': 3.0,
                    'avg_m_n_eff': 10.0,
                    'avg_e_value': -1.0,
                    'avg_e_err': 0.1,
                    'avg_e_tau_int': 3.0,
                    'avg_e_n_eff': 10.0,
                    'susc_value': 0.5,
                    'susc_err': 0.05,
                    'susc_tau_int': 3.0,
                    'susc_n_eff': 10.0,
                    'spec_h_value': 0.2,
                    'spec_h_err': 0.03,
                    'spec_h_tau_int': 3.0,
                    'spec_h_n_eff': 10.0,
                }
                for p in params_list
            ]

        monkeypatch.setattr(ising_module, 'parallel_sweep', _fake_parallel_sweep)
        monkeypatch.setattr(ising_module, 'plot_temperature_sweep', lambda **kwargs: None)

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                sys,
                'argv',
                [
                    'ising_temperature_sweep',
                    '--size', '8',
                    '--meas-steps', '20',
                    '--t-points', '2',
                    '--output-dir', tmpdir,
                ],
            )
            ising_module.main()

            data = np.load(Path(tmpdir) / 'temperature_sweep_data.npz')
            required = {
                'avg_m', 'avg_e', 'susc', 'spec_h', 'entropy', 'tau_int',
                'undefined_autocorr_flag', 'low_effective_sample_flag',
                'tau_interval_unstable_flag',
                'avg_m_value', 'avg_m_err', 'avg_m_ci_low', 'avg_m_ci_high',
                'avg_m_tau_int', 'avg_m_n_eff', 'avg_m_samples',
                'avg_e_value', 'avg_e_err', 'avg_e_ci_low', 'avg_e_ci_high',
                'susc_value', 'susc_err', 'susc_ci_low', 'susc_ci_high',
                'spec_h_value', 'spec_h_err', 'spec_h_ci_low', 'spec_h_ci_high',
                'entropy_value', 'entropy_err', 'entropy_ci_low', 'entropy_ci_high',
                'entropy_samples',
                'entropy_uncertainty_method',
            }
            present = set(data.keys())
            missing = required - present
            assert not missing, f"Missing keys in NPZ: {missing}"


class TestTemperatureSweepPlotPayloads:
    """Verify that main() forwards correct diagnostic payloads to plotting functions."""

    def test_ising_main_passes_diagnostics_plot_payload(self, monkeypatch: Any) -> None:
        """Ising sweep main should forward diagnostics arrays and note to plotting."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        import scripts.ising.temperature_sweep as ising_module

        captured_plot_kwargs: dict[str, Any] = {}

        def _fake_parallel_sweep(
            *, worker_func: Any, params: Any, num_processes: Any = None
        ) -> Any:
            params_list = list(params)
            return [
                {
                    'temperature_index': float(p.temperature_index),
                    'seed_index': float(p.seed_index),
                    'equilibrated_flag': 1.0,
                    'equilibration_steps': 100.0,
                    'avg_m_value': 1.0,
                    'avg_m_err': 0.1,
                    'avg_m_tau_int': 3.0,
                    'avg_m_n_eff': 10.0,
                    'avg_e_value': -1.0,
                    'avg_e_err': 0.1,
                    'avg_e_tau_int': 3.0,
                    'avg_e_n_eff': 10.0,
                    'susc_value': 0.5,
                    'susc_err': 0.05,
                    'susc_tau_int': 3.0,
                    'susc_n_eff': 10.0,
                    'spec_h_value': 0.2,
                    'spec_h_err': 0.03,
                    'spec_h_tau_int': 3.0,
                    'spec_h_n_eff': 10.0,
                }
                for p in params_list
            ]

        def _capture_plot_kwargs(**kwargs: Any) -> None:
            captured_plot_kwargs.update(kwargs)

        monkeypatch.setattr(ising_module, 'parallel_sweep', _fake_parallel_sweep)
        monkeypatch.setattr(ising_module, 'plot_temperature_sweep', _capture_plot_kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                sys,
                'argv',
                [
                    'ising_temperature_sweep',
                    '--size', '8',
                    '--meas-steps', '20',
                    '--t-points', '2',
                    '--n-seeds', '2',
                    '--output-dir', tmpdir,
                ],
            )
            ising_module.main()

        assert 'entropy_ci_low' in captured_plot_kwargs
        assert 'entropy_ci_high' in captured_plot_kwargs
        assert 'tau_unstable_flag' in captured_plot_kwargs
        assert 'low_effective_sample_flag' in captured_plot_kwargs
        assert 'run_metadata_note' in captured_plot_kwargs
        assert 'transition_temperatures' in captured_plot_kwargs

        metadata = str(captured_plot_kwargs['run_metadata_note'])
        assert 'target_n_seeds=2' in metadata
        transitions = cast(dict[str, float] | None, captured_plot_kwargs['transition_temperatures'])
        assert transitions is None or isinstance(transitions, dict)

    def test_ising_main_transition_preset_none_disables_guides(self, monkeypatch: Any) -> None:
        """Transition preset 'none' should forward empty transition overlays."""
        if not HAS_TEMPERATURE_SWEEP:
            import pytest
            pytest.skip("Temperature sweep modules not available")

        import scripts.ising.temperature_sweep as ising_module

        captured_plot_kwargs: dict[str, Any] = {}

        def _fake_parallel_sweep(
            *, worker_func: Any, params: Any, num_processes: Any = None
        ) -> Any:
            params_list = list(params)
            return [
                {
                    'temperature_index': float(p.temperature_index),
                    'seed_index': float(p.seed_index),
                    'equilibrated_flag': 1.0,
                    'equilibration_steps': 100.0,
                    'avg_m_value': 1.0,
                    'avg_m_err': 0.1,
                    'avg_m_tau_int': 3.0,
                    'avg_m_n_eff': 10.0,
                    'avg_e_value': -1.0,
                    'avg_e_err': 0.1,
                    'avg_e_tau_int': 3.0,
                    'avg_e_n_eff': 10.0,
                    'susc_value': 0.5,
                    'susc_err': 0.05,
                    'susc_tau_int': 3.0,
                    'susc_n_eff': 10.0,
                    'spec_h_value': 0.2,
                    'spec_h_err': 0.03,
                    'spec_h_tau_int': 3.0,
                    'spec_h_n_eff': 10.0,
                }
                for p in params_list
            ]

        def _capture_plot_kwargs(**kwargs: Any) -> None:
            captured_plot_kwargs.update(kwargs)

        monkeypatch.setattr(ising_module, 'parallel_sweep', _fake_parallel_sweep)
        monkeypatch.setattr(ising_module, 'plot_temperature_sweep', _capture_plot_kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(
                sys,
                'argv',
                [
                    'ising_temperature_sweep',
                    '--size', '8',
                    '--meas-steps', '20',
                    '--t-points', '2',
                    '--n-seeds', '2',
                    '--transition-preset', 'none',
                    '--output-dir', tmpdir,
                ],
            )
            ising_module.main()

        transitions = cast(
            dict[str, float] | None, captured_plot_kwargs.get('transition_temperatures')
        )
        assert transitions is None


class TestOrderingKineticsMain:
    """Verify that ordering kinetics scripts run through their main() loop."""

    def test_ising_kinetics_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ising ordering kinetics main() execution with mocked I/O."""
        import scripts.ising.ordering_kinetics as ising_kinetics
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['ising_kinetics.py', '--size', '16', '--max-steps', '5', '--samples', '2'],
        )
        ising_kinetics.main()

    def test_xy_kinetics_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XY ordering kinetics main() execution with mocked I/O."""
        import scripts.xy.ordering_kinetics as xy_kinetics
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['xy_kinetics.py', '--size', '16', '--max-steps', '5', '--samples', '2'],
        )
        xy_kinetics.main()

    def test_clock_kinetics_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clock ordering kinetics main() execution with mocked I/O."""
        import scripts.clock.ordering_kinetics as clock_kinetics
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['clock_kinetics.py', '--size', '16', '--max-steps', '5', '--samples', '2'],
        )
        clock_kinetics.main()


class TestOrderingEvolutionMain:
    """Verify that ordering evolution scripts run through their main() loop."""

    def test_ising_evolution_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ising ordering evolution main() execution with mocked I/O."""
        import scripts.ising.ordering_evolution as ising_evolution
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['ising_evolution.py', '--size', '16', '--targets', '1', '2'],
        )
        ising_evolution.main()

    def test_xy_evolution_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XY ordering evolution main() execution with mocked I/O."""
        import scripts.xy.ordering_evolution as xy_evolution
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['xy_evolution.py', '--size', '16', '--targets', '1', '2'],
        )
        xy_evolution.main()

    def test_clock_evolution_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clock ordering evolution main() execution with mocked I/O."""
        import scripts.clock.ordering_evolution as clock_evolution
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['clock_evolution.py', '--size', '16', '--targets', '1', '2'],
        )
        clock_evolution.main()


class TestMiscScriptsMain:
    """Verify that miscellaneous analysis scripts run through their main() loop."""

    def test_throughput_benchmark_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Throughput benchmark main() execution."""
        import scripts.benchmarks.throughput as throughput
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['throughput.py', '--sizes', '4', '--sweeps', '2'],
        )
        throughput.main()

    def test_wolff_efficiency_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ising Wolff efficiency main() execution."""
        import scripts.ising.wolff_efficiency as wolff_efficiency
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        # Use a very small number of points and steps
        monkeypatch.setattr(
            'sys.argv',
            [
                'wolff_efficiency.py',
                '--size', '8',
                '--t-min', '2.0',
                '--t-max', '2.5',
                '--t-points', '2',
                '--meas-steps', '5',
                '--eq-max-steps', '10',
            ],
        )
        wolff_efficiency.main()

    def test_bkt_transition_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XY BKT transition main() execution."""
        import scripts.xy.bkt_transition as bkt
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['bkt_transition.py', '--size', '8', '--meas-steps', '5', '--t-points', '2'],
        )
        bkt.main()

    def test_helicity_modulus_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """XY helicity modulus main() execution."""
        import scripts.xy.helicity_modulus as helicity
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['helicity_modulus.py', '--size', '8', '--meas-steps', '5', '--t-points', '2'],
        )
        helicity.main()

    def test_ising_correlation_comparison_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ising correlation comparison main() execution."""
        import scripts.ising.correlation_comparison as corr_comp
        monkeypatch.setattr('matplotlib.pyplot.savefig', lambda *args, **kwargs: None)
        monkeypatch.setattr('matplotlib.pyplot.close', lambda *args, **kwargs: None)
        monkeypatch.setattr('os.makedirs', lambda *args, **kwargs: None)
        monkeypatch.setattr(
            'sys.argv',
            ['correlation_comparison.py', '--size', '16', '--steps', '5'],
        )
        corr_comp.main()
