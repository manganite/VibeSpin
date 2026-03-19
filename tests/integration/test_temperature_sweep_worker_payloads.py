"""Integration tests for typed temperature-sweep worker payload contracts."""
from __future__ import annotations

import numpy as np

from scripts.clock.temperature_sweep import SweepPoint as ClockSweepPoint
from scripts.clock.temperature_sweep import simulate_temperature as simulate_clock_temperature
from scripts.ising.temperature_sweep import SweepPoint as IsingSweepPoint
from scripts.ising.temperature_sweep import simulate_temperature as simulate_ising_temperature
from scripts.xy.temperature_sweep import SweepPoint as XYSweepPoint
from scripts.xy.temperature_sweep import simulate_temperature as simulate_xy_temperature


def _assert_valid_thermo_result(result: tuple[float, float, float, float, float]) -> None:
    """Validate common return shape and finite/NaN-safe numeric outputs."""
    assert len(result) == 5
    for value in result:
        assert isinstance(value, (float, np.floating))
        assert np.isfinite(value) or np.isnan(value)


def test_ising_worker_accepts_typed_payload() -> None:
    """Ising worker should accept SweepPoint payload and return 5-value thermodynamics."""
    payload = IsingSweepPoint(
        temperature=2.0,
        size=8,
        meas_steps=40,
        eq_probe_steps=10,
        eq_max_steps=40,
    )
    result = simulate_ising_temperature(payload)
    _assert_valid_thermo_result(result)


def test_xy_worker_accepts_typed_payload() -> None:
    """XY worker should accept SweepPoint payload and return 5-value thermodynamics."""
    payload = XYSweepPoint(
        temperature=0.9,
        size=8,
        meas_steps=40,
        eq_probe_steps=10,
        eq_max_steps=40,
    )
    result = simulate_xy_temperature(payload)
    _assert_valid_thermo_result(result)


def test_clock_worker_accepts_typed_payload() -> None:
    """Clock worker should accept SweepPoint payload and return 5-value thermodynamics."""
    payload = ClockSweepPoint(
        temperature=0.8,
        size=8,
        q=6,
        aniso=0.1,
        meas_steps=40,
        eq_probe_steps=10,
        eq_max_steps=40,
        discrete=False,
    )
    result = simulate_clock_temperature(payload)
    _assert_valid_thermo_result(result)
