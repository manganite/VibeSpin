"""Focused branch tests for Ising, XY, and Clock model control flow."""
from __future__ import annotations

import numpy as np

import models.clock_model as clock_model
import models.ising_model as ising_model
import models.xy_model as xy_model
from models.clock_model import ClockSimulation, DiscreteClockSimulation
from models.ising_model import IsingSimulation
from models.simulation_base import _derive_step_seed
from models.xy_model import XYSimulation

# ---------------------------------------------------------------------------
# IsingSimulation
# ---------------------------------------------------------------------------


def test_ising_ordered_init_state_sets_all_spins_up() -> None:
    """Ordered initialization should create an all +1 lattice."""
    sim = IsingSimulation(size=6, temp=1.5, init_state='ordered')
    assert sim.spins is not None
    assert np.all(sim.spins == 1)


def test_ising_step_increments_even_if_spins_missing() -> None:
    """Defensive branch: step counter should still increment when spins is None."""
    sim = IsingSimulation(size=4, temp=2.0)
    sim.spins = None
    start = sim.steps

    sim.step()

    assert sim.steps == start + 1


def test_ising_getters_fallback_when_spins_missing() -> None:
    """Getter fallbacks should return safe defaults when spin storage is unavailable."""
    sim = IsingSimulation(size=4, temp=2.0)
    sim.spins = None

    assert sim._get_magnetization() == 0.0
    assert sim._get_energy() == 0.0
    np.testing.assert_array_equal(sim._get_structure_factor_squared_unshifted(), np.array([]))


def test_ising_step_uses_random_kernel(monkeypatch) -> None:
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


def test_ising_step_uses_wolff_kernel_and_tracks_cluster_size(monkeypatch) -> None:
    """Wolff mode should dispatch to cluster kernel and persist returned cluster size."""
    sim = IsingSimulation(size=4, temp=2.0, update='wolff')

    def _fake_wolff(**kwargs):
        return kwargs['spins'], 7

    monkeypatch.setattr(ising_model, 'ising_wolff_step_numba', _fake_wolff)

    sim.step()

    assert sim.last_cluster_size == 7
    assert sim.steps == 1


def test_ising_step_uses_parallel_checkerboard_kernel(monkeypatch) -> None:
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


def test_ising_step_uses_serial_checkerboard_kernel(monkeypatch) -> None:
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


def test_ising_seeded_step_seeds_numba_rng(monkeypatch) -> None:
    """When seed is provided, each step should reseed Numba RNG with the mixed step seed."""
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

    assert seeded == [
        _derive_step_seed(seed=10, step=0),
        _derive_step_seed(seed=10, step=1),
    ]


# ---------------------------------------------------------------------------
# XYSimulation
# ---------------------------------------------------------------------------


def test_xy_ordered_init_state_sets_all_spins_x_axis() -> None:
    """Ordered initialization should create unit vectors aligned with +x."""
    sim = XYSimulation(size=6, temp=1.2, init_state='ordered')
    assert sim.spins is not None
    np.testing.assert_allclose(sim.spins[..., 0], 1.0)
    np.testing.assert_allclose(sim.spins[..., 1], 0.0)


def test_xy_step_increments_even_if_spins_missing() -> None:
    """Defensive branch: step counter should still advance when spins is None."""
    sim = XYSimulation(size=4, temp=1.0)
    sim.spins = None
    start = sim.steps

    sim.step()

    assert sim.steps == start + 1


def test_xy_getters_fallback_when_spins_missing() -> None:
    """Getter fallbacks should return safe defaults when spin storage is unavailable."""
    sim = XYSimulation(size=4, temp=1.0)
    sim.spins = None

    assert sim._get_magnetization() == 0.0
    assert sim._get_energy() == 0.0
    np.testing.assert_array_equal(sim._calculate_vorticity(), np.array([]))
    assert sim._get_vortex_density() == 0.0
    assert sim._get_helicity_data() == (0.0, 0.0)
    np.testing.assert_array_equal(sim._get_structure_factor_squared_unshifted(), np.array([]))


def test_xy_step_uses_random_kernel(monkeypatch) -> None:
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


def test_xy_step_uses_wolff_kernel(monkeypatch) -> None:
    """Wolff mode should dispatch to the XY Wolff cluster kernel."""
    sim = XYSimulation(size=4, temp=1.0, update='wolff')
    calls: list[str] = []

    def _fake_wolff(**kwargs):
        calls.append('wolff')
        return kwargs['spins'], 5

    monkeypatch.setattr(xy_model, 'o2_wolff_step_numba', _fake_wolff)

    sim.step()

    assert calls == ['wolff']
    assert sim.steps == 1


def test_xy_step_uses_parallel_checkerboard_kernel(monkeypatch) -> None:
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


def test_xy_step_uses_serial_checkerboard_kernel(monkeypatch) -> None:
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


def test_xy_seeded_step_seeds_numba_rng(monkeypatch) -> None:
    """When seed is provided, each step should reseed Numba RNG with the mixed step seed."""
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

    assert seeded == [
        _derive_step_seed(seed=21, step=0),
        _derive_step_seed(seed=21, step=1),
    ]


# ---------------------------------------------------------------------------
# ClockSimulation
# ---------------------------------------------------------------------------


def test_clock_ordered_init_state_sets_all_spins_x_axis() -> None:
    """Ordered clock initialization should create unit vectors aligned with +x."""
    sim = ClockSimulation(size=6, temp=1.2, init_state='ordered')
    assert sim.spins is not None
    np.testing.assert_allclose(sim.spins[..., 0], 1.0)
    np.testing.assert_allclose(sim.spins[..., 1], 0.0)


def test_clock_step_increments_even_if_spins_missing() -> None:
    """Defensive branch: Clock step counter should still advance when spins is None."""
    sim = ClockSimulation(size=4, temp=1.0)
    sim.spins = None
    start = sim.steps

    sim.step()

    assert sim.steps == start + 1


def test_clock_getters_fallback_when_spins_missing() -> None:
    """Clock getter fallbacks should return safe defaults when spin storage is unavailable."""
    sim = ClockSimulation(size=4, temp=1.0)
    sim.spins = None

    assert sim._get_magnetization() == 0.0
    assert sim._get_energy() == 0.0
    np.testing.assert_array_equal(sim._calculate_vorticity(), np.array([]))
    assert sim._get_vortex_density() == 0.0
    assert sim._get_helicity_data() == (0.0, 0.0)
    np.testing.assert_array_equal(sim._get_structure_factor_squared_unshifted(), np.array([]))


def test_clock_step_uses_random_kernel(monkeypatch) -> None:
    """Clock random mode should dispatch to the random Metropolis kernel."""
    sim = ClockSimulation(size=4, temp=1.0, update='random')
    calls: list[str] = []

    def _fake_random(**kwargs):
        calls.append('random')
        return kwargs['spins']

    monkeypatch.setattr(clock_model, 'clock_step_random_numba', _fake_random)

    sim.step()

    assert calls == ['random']
    assert sim.steps == 1


def test_clock_step_uses_wolff_kernel(monkeypatch) -> None:
    """Clock wolff mode should dispatch to the wolff cluster kernel."""
    sim = ClockSimulation(size=4, temp=1.0, A=0.0, update='wolff')
    calls: list[str] = []

    def _fake_wolff(**kwargs):
        calls.append('wolff')
        return kwargs['spins'], 0

    monkeypatch.setattr(clock_model, 'o2_wolff_step_numba', _fake_wolff)

    sim.step()

    assert calls == ['wolff']
    assert sim.steps == 1


def test_clock_step_uses_parallel_checkerboard_kernel(monkeypatch) -> None:
    """Clock checkerboard mode with parallel=True should dispatch to parallel kernel."""
    sim = ClockSimulation(size=4, temp=1.0, update='checkerboard', parallel=True)
    calls: list[str] = []

    def _fake_parallel(**kwargs):
        calls.append('parallel')
        return kwargs['spins']

    monkeypatch.setattr(clock_model, 'clock_step_parallel_numba', _fake_parallel)

    sim.step()

    assert calls == ['parallel']
    assert sim.steps == 1


def test_clock_step_uses_serial_checkerboard_kernel(monkeypatch) -> None:
    """Clock checkerboard mode with parallel=False should dispatch to serial kernel."""
    sim = ClockSimulation(size=4, temp=1.0, update='checkerboard', parallel=False)
    calls: list[str] = []

    def _fake_serial(**kwargs):
        calls.append('serial')
        return kwargs['spins']

    monkeypatch.setattr(clock_model, 'clock_step_numba', _fake_serial)

    sim.step()

    assert calls == ['serial']
    assert sim.steps == 1


def test_clock_seeded_step_seeds_numba_rng(monkeypatch) -> None:
    """Clock with seed should reseed Numba RNG with seed + step."""
    sim = ClockSimulation(size=4, temp=1.0, seed=30)
    seeded: list[int] = []

    def _fake_seed_numba(*, seed: int) -> None:
        seeded.append(seed)

    def _fake_serial(**kwargs):
        return kwargs['spins']

    monkeypatch.setattr('models.simulation_base._seed_numba', _fake_seed_numba)
    monkeypatch.setattr(clock_model, 'clock_step_numba', _fake_serial)

    sim.step()
    sim.step()

    assert seeded == [
        _derive_step_seed(seed=30, step=0),
        _derive_step_seed(seed=30, step=1),
    ]


# ---------------------------------------------------------------------------
# DiscreteClockSimulation
# ---------------------------------------------------------------------------


def test_discrete_clock_ordered_init_state_is_zero() -> None:
    """Ordered discrete-clock initialization should set all sites to state 0."""
    sim = DiscreteClockSimulation(size=6, temp=1.2, init_state='ordered', q=6)
    assert sim.spins is not None
    np.testing.assert_array_equal(sim.spins, 0)


def test_discrete_clock_step_increments_even_if_spins_missing() -> None:
    """Defensive branch: DiscreteClock step counter should still advance when spins is None."""
    sim = DiscreteClockSimulation(size=4, temp=1.0, q=6)
    sim.spins = None
    start = sim.steps

    sim.step()

    assert sim.steps == start + 1


def test_discrete_clock_getters_fallback_when_spins_missing() -> None:
    """DiscreteClock getter fallbacks should return safe defaults without spin storage."""
    sim = DiscreteClockSimulation(size=4, temp=1.0, q=6)
    sim.spins = None

    assert sim._get_magnetization() == 0.0
    assert sim._get_energy() == 0.0
    np.testing.assert_array_equal(sim._calculate_vorticity(), np.array([]))
    assert sim._get_vortex_density() == 0.0
    assert sim._get_helicity_data() == (0.0, 0.0)
    np.testing.assert_array_equal(sim._get_structure_factor_squared_unshifted(), np.array([]))


def test_discrete_clock_step_uses_random_kernel(monkeypatch) -> None:
    """Discrete clock random mode should dispatch to random kernel."""
    sim = DiscreteClockSimulation(size=4, temp=1.0, q=6, update='random')
    calls: list[str] = []

    def _fake_random(**kwargs):
        calls.append('random')
        return kwargs['spins']

    monkeypatch.setattr(clock_model, 'discrete_clock_step_random_numba', _fake_random)

    sim.step()

    assert calls == ['random']
    assert sim.steps == 1


def test_discrete_clock_step_uses_parallel_checkerboard_kernel(monkeypatch) -> None:
    """Discrete checkerboard mode with parallel=True should dispatch to parallel kernel."""
    sim = DiscreteClockSimulation(size=4, temp=1.0, q=6, update='checkerboard', parallel=True)
    calls: list[str] = []

    def _fake_parallel(**kwargs):
        calls.append('parallel')
        return kwargs['spins']

    monkeypatch.setattr(clock_model, 'discrete_clock_step_parallel_numba', _fake_parallel)

    sim.step()

    assert calls == ['parallel']
    assert sim.steps == 1


def test_discrete_clock_step_uses_serial_checkerboard_kernel(monkeypatch) -> None:
    """Discrete checkerboard mode with parallel=False should dispatch to serial kernel."""
    sim = DiscreteClockSimulation(size=4, temp=1.0, q=6, update='checkerboard', parallel=False)
    calls: list[str] = []

    def _fake_serial(**kwargs):
        calls.append('serial')
        return kwargs['spins']

    monkeypatch.setattr(clock_model, 'discrete_clock_step_numba', _fake_serial)

    sim.step()

    assert calls == ['serial']
    assert sim.steps == 1


def test_discrete_clock_seeded_step_seeds_numba_rng(monkeypatch) -> None:
    """Discrete clock with seed should reseed Numba RNG with seed + step."""
    sim = DiscreteClockSimulation(size=4, temp=1.0, q=6, seed=40)
    seeded: list[int] = []

    def _fake_seed_numba(*, seed: int) -> None:
        seeded.append(seed)

    def _fake_serial(**kwargs):
        return kwargs['spins']

    monkeypatch.setattr('models.simulation_base._seed_numba', _fake_seed_numba)
    monkeypatch.setattr(clock_model, 'discrete_clock_step_numba', _fake_serial)

    sim.step()
    sim.step()

    assert seeded == [
        _derive_step_seed(seed=40, step=0),
        _derive_step_seed(seed=40, step=1),
    ]
