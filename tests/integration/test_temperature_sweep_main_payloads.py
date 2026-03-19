"""Integration tests for typed payload construction in temperature-sweep main entry points."""
from __future__ import annotations

import sys
from typing import Any

from scripts.clock.temperature_sweep import SweepPoint as ClockSweepPoint
from scripts.ising.temperature_sweep import SweepPoint as IsingSweepPoint
from scripts.xy.temperature_sweep import SweepPoint as XYSweepPoint


def _capture_sweep_params(
    monkeypatch,
    module: Any,
    sweep_point_type: type,
    argv: list[str],
) -> None:
    """Run a sweep main() with patched dependencies and assert typed payload construction."""
    captured: dict[str, list[Any]] = {}

    def _fake_parallel_sweep(*, worker_func, params, num_processes=None):
        params_list = list(params)
        captured['params'] = params_list
        return [(1.0, 2.0, 3.0, 4.0, 5.0)] * len(params_list)

    monkeypatch.setattr(module, 'parallel_sweep', _fake_parallel_sweep)
    monkeypatch.setattr(module, 'plot_temperature_sweep', lambda **kwargs: None)
    monkeypatch.setattr(sys, 'argv', argv)

    module.main()

    assert 'params' in captured
    assert len(captured['params']) == 2
    assert all(isinstance(p, sweep_point_type) for p in captured['params'])


def test_ising_main_builds_typed_sweep_payloads(monkeypatch) -> None:
    """Ising temperature sweep main should build Ising SweepPoint payloads."""
    import scripts.ising.temperature_sweep as ising_module

    _capture_sweep_params(
        monkeypatch,
        ising_module,
        IsingSweepPoint,
        ['ising_temperature_sweep', '--size', '8', '--meas-steps', '20', '--t-points', '2'],
    )


def test_xy_main_builds_typed_sweep_payloads(monkeypatch) -> None:
    """XY temperature sweep main should build XY SweepPoint payloads."""
    import scripts.xy.temperature_sweep as xy_module

    _capture_sweep_params(
        monkeypatch,
        xy_module,
        XYSweepPoint,
        ['xy_temperature_sweep', '--size', '8', '--meas-steps', '20', '--t-points', '2'],
    )


def test_clock_main_builds_typed_sweep_payloads(monkeypatch) -> None:
    """Clock temperature sweep main should build Clock SweepPoint payloads."""
    import scripts.clock.temperature_sweep as clock_module

    _capture_sweep_params(
        monkeypatch,
        clock_module,
        ClockSweepPoint,
        ['clock_temperature_sweep', '--size', '8', '--meas-steps', '20', '--t-points', '2'],
    )
