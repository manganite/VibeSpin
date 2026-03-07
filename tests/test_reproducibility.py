"""
Tests for scientific reproducibility across models.
Verifies that identical seeds produce identical simulation trajectories.
"""

import numpy as np

from models.clock_model import ClockSimulation
from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation


def test_ising_reproducibility():
    """Two Ising simulations with the same seed must be identical."""
    L, T, seed = 32, 2.0, 42
    sim1 = IsingSimulation(size=L, temp=T, seed=seed)
    sim2 = IsingSimulation(size=L, temp=T, seed=seed)

    # Check initial configurations
    assert np.array_equal(sim1.spins, sim2.spins)

    # Run multiple steps
    for _ in range(5):
        sim1.step()
        sim2.step()

    assert np.array_equal(sim1.spins, sim2.spins)
    assert sim1._get_energy() == sim2._get_energy()
    assert sim1._get_magnetization() == sim2._get_magnetization()


def test_xy_reproducibility():
    """Two XY simulations with the same seed must be identical."""
    L, T, seed = 32, 0.5, 123
    sim1 = XYSimulation(size=L, temp=T, seed=seed)
    sim2 = XYSimulation(size=L, temp=T, seed=seed)

    assert np.array_equal(sim1.spins, sim2.spins)

    for _ in range(5):
        sim1.step()
        sim2.step()

    assert np.array_equal(sim1.spins, sim2.spins)
    assert sim1._get_energy() == sim2._get_energy()


def test_clock_reproducibility():
    """Two Clock simulations with the same seed must be identical."""
    L, T, Q, seed = 32, 0.2, 6, 999
    sim1 = ClockSimulation(size=L, temp=T, q=Q, seed=seed)
    sim2 = ClockSimulation(size=L, temp=T, q=Q, seed=seed)

    assert np.array_equal(sim1.spins, sim2.spins)

    for _ in range(5):
        sim1.step()
        sim2.step()

    assert np.array_equal(sim1.spins, sim2.spins)


def test_different_seeds_produce_different_results():
    """Simulations with different seeds should (almost certainly) diverge."""
    L, T = 32, 2.0
    sim1 = IsingSimulation(size=L, temp=T, seed=1)
    sim2 = IsingSimulation(size=L, temp=T, seed=2)

    # Initial state might be different
    # But after one step they should definitely diverge
    sim1.step()
    sim2.step()

    assert not np.array_equal(sim1.spins, sim2.spins)
