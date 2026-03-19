"""Focused branch tests for XYSimulation control flow."""
from __future__ import annotations

import numpy as np

import models.xy_model as xy_model
from models.xy_model import XYSimulation


def test_ordered_init_state_sets_all_spins_x_axis() -> None:
    """Ordered initialization should create unit vectors aligned with +x."""
    sim = XYSimulation(size=6, temp=1.2, init_state='ordered')
    assert sim.spins is not None
    np.testing.assert_allclose(sim.spins[..., 0], 1.0)
    np.testing.assert_allclose(sim.spins[..., 1], 0.0)


def test_step_increments_even_if_spins_missing() -> None:
    """Defensive branch: step counter should still advance when spins is None."""
    sim = XYSimulation(size=4, temp=1.0)
    sim.spins = None
    start = sim.steps

    sim.step()

    assert sim.steps == start + 1


def test_getters_fallback_when_spins_missing() -> None:
    """Getter fallbacks should return safe defaults when spin storage is unavailable."""
    sim = XYSimulation(size=4, temp=1.0)
    sim.spins = None

    assert sim._get_magnetization() == 0.0
    assert sim._get_energy() == 0.0
    np.testing.assert_array_equal(sim._calculate_vorticity(), np.array([]))
    assert sim._get_vortex_density() == 0.0
    assert sim._get_helicity_data() == (0.0, 0.0)
    np.testing.assert_array_equal(sim._get_structure_factor_squared_unshifted(), np.array([]))


def test_step_uses_random_kernel(monkeypatch) -> None:
    """Random update mode should dispatch to the random Metropolis kernel."""
    sim = XYSimulation(size=4, temp=1.0, update='random')
    calls: list[str] = []

    def _fake_random(**kwargs):
        calls.append('random')
        return kwargs['spins']

    monkeypatch.setattr(xy_model, 'xy_step_random_numba', _fake_random)

    sim.step()

    assert calls == ['random']
    assert sim.steps == 1


def test_step_uses_wolff_kernel(monkeypatch) -> None:
    """Wolff mode should dispatch to the XY Wolff cluster kernel."""
    sim = XYSimulation(size=4, temp=1.0, update='wolff')
    calls: list[str] = []

    def _fake_wolff(**kwargs):
        calls.append('wolff')
        return kwargs['spins']

    monkeypatch.setattr(xy_model, 'xy_wolff_step_numba', _fake_wolff)

    sim.step()

    assert calls == ['wolff']
    assert sim.steps == 1


def test_step_uses_parallel_checkerboard_kernel(monkeypatch) -> None:
    """Checkerboard mode with parallel=True should dispatch to the parallel kernel."""
    sim = XYSimulation(size=4, temp=1.0, update='checkerboard', parallel=True)
    calls: list[str] = []

    def _fake_parallel(**kwargs):
        calls.append('parallel')
        return kwargs['spins']

    monkeypatch.setattr(xy_model, 'xy_step_parallel_numba', _fake_parallel)

    sim.step()

    assert calls == ['parallel']
    assert sim.steps == 1


def test_step_uses_serial_checkerboard_kernel(monkeypatch) -> None:
    """Checkerboard mode with parallel=False should dispatch to serial kernel."""
    sim = XYSimulation(size=4, temp=1.0, update='checkerboard', parallel=False)
    calls: list[str] = []

    def _fake_serial(**kwargs):
        calls.append('serial')
        return kwargs['spins']

    monkeypatch.setattr(xy_model, 'xy_step_numba', _fake_serial)

    sim.step()

    assert calls == ['serial']
    assert sim.steps == 1


def test_seeded_step_seeds_numba_rng(monkeypatch) -> None:
    """When seed is provided, each step should reseed Numba RNG with seed + step."""
    sim = XYSimulation(size=4, temp=1.0, seed=21)
    seeded: list[int] = []

    def _fake_seed_numba(*, seed: int) -> None:
        seeded.append(seed)

    def _fake_serial(**kwargs):
        return kwargs['spins']

    monkeypatch.setattr('models.simulation_base._seed_numba', _fake_seed_numba)
    monkeypatch.setattr(xy_model, 'xy_step_numba', _fake_serial)

    sim.step()
    sim.step()

    assert seeded == [21, 22]
