"""Focused branch tests for IsingSimulation control flow."""
from __future__ import annotations

import numpy as np

import models.ising_model as ising_model
from models.ising_model import IsingSimulation


def test_ordered_init_state_sets_all_spins_up() -> None:
    """Ordered initialization should create an all +1 lattice."""
    sim = IsingSimulation(size=6, temp=1.5, init_state='ordered')
    assert sim.spins is not None
    assert np.all(sim.spins == 1)


def test_step_increments_even_if_spins_missing() -> None:
    """Defensive branch: step counter should still increment when spins is None."""
    sim = IsingSimulation(size=4, temp=2.0)
    sim.spins = None
    start = sim.steps

    sim.step()

    assert sim.steps == start + 1


def test_getters_fallback_when_spins_missing() -> None:
    """Getter fallbacks should return safe defaults when spin storage is unavailable."""
    sim = IsingSimulation(size=4, temp=2.0)
    sim.spins = None

    assert sim._get_magnetization() == 0.0
    assert sim._get_energy() == 0.0
    np.testing.assert_array_equal(sim._get_structure_factor_squared_unshifted(), np.array([]))


def test_step_uses_random_kernel(monkeypatch) -> None:
    """Random update mode should dispatch to the random Metropolis kernel."""
    sim = IsingSimulation(size=4, temp=2.0, update='random')
    calls: list[str] = []

    def _fake_random(**kwargs):
        calls.append('random')
        return kwargs['spins']

    monkeypatch.setattr(ising_model, 'ising_step_random_numba', _fake_random)

    sim.step()

    assert calls == ['random']
    assert sim.steps == 1


def test_step_uses_wolff_kernel_and_tracks_cluster_size(monkeypatch) -> None:
    """Wolff mode should dispatch to cluster kernel and persist returned cluster size."""
    sim = IsingSimulation(size=4, temp=2.0, update='wolff')

    def _fake_wolff(**kwargs):
        return kwargs['spins'], 7

    monkeypatch.setattr(ising_model, 'ising_wolff_step_numba', _fake_wolff)

    sim.step()

    assert sim.last_cluster_size == 7
    assert sim.steps == 1


def test_step_uses_parallel_checkerboard_kernel(monkeypatch) -> None:
    """Checkerboard mode with parallel=True should dispatch to the parallel kernel."""
    sim = IsingSimulation(size=4, temp=2.0, update='checkerboard', parallel=True)
    calls: list[str] = []

    def _fake_parallel(**kwargs):
        calls.append('parallel')
        return kwargs['spins']

    monkeypatch.setattr(ising_model, 'ising_step_parallel_numba', _fake_parallel)

    sim.step()

    assert calls == ['parallel']
    assert sim.steps == 1


def test_step_uses_serial_checkerboard_kernel(monkeypatch) -> None:
    """Checkerboard mode with parallel=False should dispatch to serial kernel."""
    sim = IsingSimulation(size=4, temp=2.0, update='checkerboard', parallel=False)
    calls: list[str] = []

    def _fake_serial(**kwargs):
        calls.append('serial')
        return kwargs['spins']

    monkeypatch.setattr(ising_model, 'ising_step_numba', _fake_serial)

    sim.step()

    assert calls == ['serial']
    assert sim.steps == 1


def test_seeded_step_seeds_numba_rng(monkeypatch) -> None:
    """When seed is provided, each step should reseed Numba RNG with seed + step."""
    sim = IsingSimulation(size=4, temp=2.0, seed=10)
    seeded: list[int] = []

    def _fake_seed_numba(*, seed: int) -> None:
        seeded.append(seed)

    def _fake_serial(**kwargs):
        return kwargs['spins']

    monkeypatch.setattr('models.simulation_base._seed_numba', _fake_seed_numba)
    monkeypatch.setattr(ising_model, 'ising_step_numba', _fake_serial)

    sim.step()
    sim.step()

    assert seeded == [10, 11]
