"""
Enhanced coverage tests for VibeSpin simulation models.
Targets under-tested areas: random updates, parallelization, and specialized observables.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from models.ising_model import IsingSimulation
from models.simulation_base import _seed_numba
from models.xy_model import XYSimulation


def test_numba_seed():
    """Verify that _seed_numba runs without error."""
    _seed_numba(seed=42)

def test_ising_random_update():
    """Verify Ising random update scheme."""
    size = 10
    sim = IsingSimulation(size=size, temp=2.0, update='random')
    sim.step()
    assert sim.steps == 1
    assert sim.update == 'random'
    # Check that spins remain valid
    assert np.all(np.logical_or(sim.spins == 1, sim.spins == -1))

def test_ising_checkerboard_update():
    """Verify Ising standard checkerboard update."""
    size = 10
    sim = IsingSimulation(size=size, temp=2.0, update='checkerboard')
    sim.step()
    assert sim.steps == 1
    assert sim.update == 'checkerboard'

def test_ising_parallel_update():
    """Verify Ising parallel checkerboard update."""
    size = 16
    sim = IsingSimulation(size=size, temp=2.0, parallel=True)
    sim.step()
    assert sim.steps == 1
    assert sim.parallel is True

def test_xy_random_update():
    """Verify XY random update scheme."""
    size = 10
    sim = XYSimulation(size=size, temp=1.0, update='random')
    sim.step()
    assert sim.steps == 1
    assert sim.update == 'random'
    norms = np.linalg.norm(sim.spins, axis=-1)
    np.testing.assert_allclose(norms, 1.0)

def test_xy_checkerboard_update():
    """Verify XY standard checkerboard update."""
    size = 10
    sim = XYSimulation(size=size, temp=1.0, update='checkerboard')
    sim.step()
    assert sim.steps == 1

def test_xy_parallel_update():
    """Verify XY parallel checkerboard update."""
    size = 16
    sim = XYSimulation(size=size, temp=1.0, parallel=True)
    sim.step()
    assert sim.steps == 1
    assert sim.parallel is True

def test_xy_helicity_data():
    """Verify helicity data calculation for XY model."""
    size = 10
    sim = XYSimulation(size=size, temp=1.0)
    # Uniform state
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0

    cos_sum, sin_sum = sim._get_helicity_data()
    # In uniform state, cos(theta_i - theta_j) = 1, sin(...) = 0
    # There are N*N bonds in x-direction
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)

def test_clock_random_update():
    """Verify Clock random update scheme."""
    size = 10
    sim = ClockSimulation(size=size, temp=0.5, q=6, update='random')
    sim.step()
    assert sim.steps == 1
    norms = np.linalg.norm(sim.spins, axis=-1)
    np.testing.assert_allclose(norms, 1.0)

def test_clock_checkerboard_update():
    """Verify Clock standard checkerboard update."""
    size = 10
    sim = ClockSimulation(size=size, temp=0.5, q=6, update='checkerboard')
    sim.step()
    assert sim.steps == 1

def test_clock_parallel_update():
    """Verify Clock parallel checkerboard update."""
    size = 16
    sim = ClockSimulation(size=size, temp=0.5, q=6, parallel=True)
    sim.step()
    assert sim.steps == 1

def test_clock_helicity_data():
    """Verify helicity data calculation for Clock model."""
    size = 10
    sim = ClockSimulation(size=size, temp=0.5, q=6)
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0
    cos_sum, sin_sum = sim._get_helicity_data()
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)

def test_discrete_clock_checkerboard_update():
    """Verify DiscreteClock standard checkerboard update."""
    size = 10
    sim = DiscreteClockSimulation(size=size, temp=0.5, q=6, update='checkerboard')
    sim.step()
    assert sim.steps == 1

def test_discrete_clock_parallel_update():
    """Verify DiscreteClock parallel checkerboard update."""
    size = 16
    sim = DiscreteClockSimulation(size=size, temp=0.5, q=6, parallel=True)
    sim.step()
    assert sim.steps == 1

def test_discrete_clock_helicity_data():
    """Verify helicity data calculation for DiscreteClock model."""
    size = 10
    sim = DiscreteClockSimulation(size=size, temp=0.5, q=6)
    sim.spins = np.zeros((size, size), dtype=np.int32)
    cos_sum, sin_sum = sim._get_helicity_data()
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)

def test_discrete_clock_vorticity_detection():
    """Verify vorticity detection in DiscreteClock (via conversion to vectors)."""
    size = 4
    sim = DiscreteClockSimulation(size=size, temp=0.5, q=4)
    # Manual vortex: 0, 1, 2, 3 around a plaquette
    # s(0,0)=0, s(0,1)=1, s(1,1)=2, s(1,0)=3
    sim.spins = np.zeros((size, size), dtype=np.int32)
    sim.spins[0, 0] = 0
    sim.spins[0, 1] = 1
    sim.spins[1, 1] = 2
    sim.spins[1, 0] = 3

    vorticity = sim._calculate_vorticity()
    assert vorticity[0, 0] == 1.0

def test_base_structure_factor():
    """Verify base class structure factor calculation."""
    size = 8
    sim = IsingSimulation(size=size, temp=2.0)
    sim.spins = np.ones((size, size))
    sf = sim._calculate_structure_factor()
    assert sf.shape == (size, size)
    # For uniform state, S(0) = N^2, shifted center should be large
    assert sf[size//2, size//2] == pytest.approx(float(size * size))

def test_clock_energy_calc():
    """Trigger energy calculation for continuous clock model."""
    size = 4
    sim = ClockSimulation(size=size, temp=0.5, q=6)
    e = sim._get_energy()
    assert isinstance(e, float)

def test_xy_structure_factor_squared():
    """Verify unshifted squared SF for vector spins."""
    size = 4
    sim = XYSimulation(size=size, temp=1.0)
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0
    sk_sq = sim._get_structure_factor_squared_unshifted()
    assert sk_sq.shape == (size, size)
    assert sk_sq[0, 0] == pytest.approx(float((size*size)**2))

def test_clock_structure_factor_squared():
    """Verify unshifted squared SF for continuous clock model."""
    size = 4
    sim = ClockSimulation(size=size, temp=0.5, q=6)
    sk_sq = sim._get_structure_factor_squared_unshifted()
    assert sk_sq.shape == (size, size)

def test_discrete_clock_structure_factor_squared():
    """Verify unshifted squared SF for discrete clock model."""
    size = 4
    sim = DiscreteClockSimulation(size=size, temp=0.5, q=6)
    sk_sq = sim._get_structure_factor_squared_unshifted()
    assert sk_sq.shape == (size, size)
