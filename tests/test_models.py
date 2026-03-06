"""
Unit tests for the Monte Carlo simulation models (Ising, XY, Clock).
"""

import numpy as np
import pytest

from models.clock_model import ClockSimulation
from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation


@pytest.fixture
def standard_params():
    """Fixture providing standard test parameters."""
    return {"size": 10, "temp": 2.0}

def test_ising_initialization(standard_params):
    """Verify correct initialization of the Ising model."""
    size, temp = standard_params["size"], standard_params["temp"]
    sim = IsingSimulation(size, temp)
    assert sim.size == size
    assert sim.temp == temp
    assert sim.spins.shape == (size, size)
    assert np.all(np.logical_or(sim.spins == 1, sim.spins == -1))

def test_ising_step(standard_params):
    """Verify that a single MC step in the Ising model executes without error."""
    sim = IsingSimulation(**standard_params)
    sim.step()
    assert sim.steps == 1

def test_ising_run(standard_params):
    """Verify that a short simulation run returns the expected number of measurements."""
    sim = IsingSimulation(**standard_params)
    n_steps = 5
    mags, engs = sim.run(n_steps)
    assert len(mags) == n_steps
    assert len(engs) == n_steps
    for m in mags:
        assert 0 <= m <= 1.0

def test_ising_low_temp_magnetization():
    """Verify that Ising model maintains high magnetization at very low temperature."""
    # Start from an ordered state (ground state)
    size = 20
    sim = IsingSimulation(size=size, temp=0.1)
    sim.spins = np.ones((size, size), dtype=np.int8)

    sim.equilibrate(100)
    mags, _ = sim.run(100)
    # At T=0.1, it should stay very close to M=1
    assert np.mean(mags) > 0.99

def test_ising_high_temp_magnetization():
    """Verify that Ising model has low magnetization at very high temperature."""
    # T = 100 is well above Tc ≈ 2.269
    size = 20
    sim = IsingSimulation(size=size, temp=100.0)
    sim.equilibrate(500)
    mags, _ = sim.run(100)
    # For L=20, M ~ 1/sqrt(N) = 1/20 = 0.05. 0.2 is a safe upper bound.
    assert np.mean(mags) < 0.2

def test_xy_initialization(standard_params):
    """Verify correct initialization and spin normalization of the XY model."""
    size, temp = standard_params["size"], standard_params["temp"]
    sim = XYSimulation(size, temp)
    assert sim.size == size
    assert sim.spins.shape == (size, size, 2)
    # Check normalization
    norms = np.linalg.norm(sim.spins, axis=-1)
    np.testing.assert_allclose(norms, 1.0)

def test_xy_step(standard_params):
    """Verify that a single MC step in the XY model maintains spin normalization."""
    sim = XYSimulation(**standard_params)
    sim.step()
    assert sim.steps == 1
    norms = np.linalg.norm(sim.spins, axis=-1)
    np.testing.assert_allclose(norms, 1.0)

def test_xy_vorticity_detection():
    """Verify that _calculate_vorticity correctly detects a manually placed vortex."""
    # Create a simple vortex at the center of a 4x4 lattice
    size = 4
    sim = XYSimulation(size, temp=1.0)

    # Angles arranged in a loop around the first plaquette (0,0)
    # s(0,0)=0, s(0,1)=pi/2, s(1,1)=pi, s(1,0)=3pi/2
    angles = np.zeros((size, size))
    angles[0, 0] = 0
    angles[0, 1] = np.pi/2
    angles[1, 1] = np.pi
    angles[1, 0] = 1.5 * np.pi

    sim.spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    vorticity = sim._calculate_vorticity()

    # The (0,0) plaquette should have winding number 1
    assert vorticity[0, 0] == 1.0

def test_clock_initialization(standard_params):
    """Verify correct initialization of the q-state clock model."""
    size, temp = standard_params["size"], standard_params["temp"]
    q = 6
    sim = ClockSimulation(size, temp, q=q)
    assert sim.size == size
    assert sim.q == q
    assert sim.spins.shape == (size, size, 2)
    norms = np.linalg.norm(sim.spins, axis=-1)
    np.testing.assert_allclose(norms, 1.0)

def test_clock_step(standard_params):
    """Verify that a single MC step in the clock model executes correctly."""
    sim = ClockSimulation(**standard_params)
    sim.step()
    assert sim.steps == 1
    norms = np.linalg.norm(sim.spins, axis=-1)
    np.testing.assert_allclose(norms, 1.0)

def test_correlation_function(standard_params):
    """Verify the calculation of the radially averaged correlation function G(r)."""
    # Test base class method via Ising
    size = standard_params["size"]
    sim = IsingSimulation(**standard_params)
    r, g_r = sim._calculate_correlation_function()
    assert len(r) == size // 2
    assert len(g_r) == size // 2
    assert pytest.approx(g_r[0]) == 1.0

def test_invalid_initialization():
    """Verify that models raise ValueError for invalid parameters."""
    # Test size validation
    with pytest.raises(ValueError):
        IsingSimulation(size=0, temp=1.0)
    with pytest.raises(ValueError):
        IsingSimulation(size=-5, temp=1.0)

    # Test temperature validation
    with pytest.raises(ValueError):
        IsingSimulation(size=10, temp=0.0)
    with pytest.raises(ValueError):
        IsingSimulation(size=10, temp=-1.0)

    # Test Ising update scheme validation
    with pytest.raises(ValueError):
        IsingSimulation(size=10, temp=1.0, update='invalid_scheme')

    # Test Clock model q-state validation
    with pytest.raises(ValueError):
        ClockSimulation(size=10, temp=1.0, q=1)
