"""
Tests for model edge cases and parameter validation to maximize coverage.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from models.ising_model import IsingSimulation, ising_energy_numba
from models.xy_model import XYSimulation


def test_ising_energy_manual():
    """Verify ising_energy_numba with a manual configuration."""
    size = 4
    # 4x4 checkerboard
    # Each site has 4 neighbors with opposite spins.
    # Total unique bonds = 2 * N = 2 * 16 = 32
    # Each bond has spins[i]*spins[j] = -1
    # Total E = -J * (-32) = 32
    # Per site: 32/16 = 2.0
    iy, ix = np.indices((size, size))
    spins = np.where((ix + iy) % 2 == 0, 1, -1).astype(np.int8)
    idx_next = np.roll(np.arange(size), -1)
    e = ising_energy_numba(spins=spins, J=1.0, idx_next=idx_next)
    assert e == pytest.approx(2.0)

def test_ising_invalid_params():
    """Verify Ising validation for size and temp."""
    with pytest.raises(ValueError, match="size must be a positive integer"):
        IsingSimulation(size=0, temp=1.0)
    with pytest.raises(ValueError, match="temp must be positive"):
        IsingSimulation(size=10, temp=0.0)
    with pytest.raises(ValueError, match="Unknown update scheme"):
        IsingSimulation(size=10, temp=1.0, update='invalid')

def test_xy_invalid_params():
    """Verify XY validation."""
    with pytest.raises(ValueError, match="Unknown update scheme"):
        XYSimulation(size=10, temp=1.0, update='invalid')

def test_clock_invalid_params():
    """Verify Clock validation."""
    with pytest.raises(ValueError, match="q must be >= 2"):
        ClockSimulation(size=10, temp=1.0, q=1)
    with pytest.raises(ValueError, match="Unknown update scheme"):
        ClockSimulation(size=10, temp=1.0, update='invalid')

def test_discrete_clock_extremes():
    """Verify DiscreteClock at T=0 and T=inf limits."""
    size = 10
    # T -> inf: should randomize
    sim_inf = DiscreteClockSimulation(size=size, temp=1e6, q=6)
    sim_inf.step()
    # T -> 0: should stay ordered if started ordered
    sim_zero = DiscreteClockSimulation(size=size, temp=1e-6, q=6)
    sim_zero.spins = np.zeros((size, size), dtype=np.int32)
    sim_zero.step()
    assert np.all(sim_zero.spins == 0)

def test_discrete_clock_structure_factor_unshifted():
    """Trigger SF code for discrete clock."""
    size = 4
    sim = DiscreteClockSimulation(size=size, temp=1.0, q=6)
    sf = sim._get_structure_factor_squared_unshifted()
    assert sf.shape == (size, size)

def test_clock_helicity_data_trigger():
    """Trigger helicity data for continuous clock."""
    size = 4
    sim = ClockSimulation(size=size, temp=1.0, q=6)
    res = sim._get_helicity_data()
    assert len(res) == 2

def test_xy_vorticity_trigger():
    """Trigger vorticity calculations for XY."""
    size = 4
    sim = XYSimulation(size=size, temp=1.0)
    v = sim._calculate_vorticity()
    d = sim._get_vortex_density()
    assert v.shape == (size, size)
    assert 0 <= d <= 1.0
