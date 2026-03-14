"""
Unit tests for the Monte Carlo simulation models (Ising, XY, Clock).
"""
from __future__ import annotations

import numpy as np
import pytest

from models.clock_model import ClockSimulation
from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation


@pytest.fixture
def standard_params():
    """Fixture providing standard test parameters."""
    return {'size': 10, 'temp': 2.0}


def test_ising_initialization(standard_params):
    """Verify correct initialization of the Ising model."""
    size, temp = standard_params['size'], standard_params['temp']
    sim = IsingSimulation(size=size, temp=temp)
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
    mags, engs = sim.run(n_steps=n_steps)
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

    sim.equilibrate(n_steps=100)
    mags, _ = sim.run(n_steps=100)
    # At T=0.1, it should stay very close to M=1
    assert np.mean(mags) > 0.99


def test_ising_high_temp_magnetization():
    """Verify that Ising model has low magnetization at very high temperature."""
    # T = 100 is well above Tc ≈ 2.269
    size = 20
    sim = IsingSimulation(size=size, temp=100.0)
    sim.equilibrate(n_steps=500)
    mags, _ = sim.run(n_steps=100)
    # For L=20, M ~ 1/sqrt(N) = 1/20 = 0.05. 0.2 is a safe upper bound.
    assert np.mean(mags) < 0.2


def test_xy_initialization(standard_params):
    """Verify correct initialization and spin normalization of the XY model."""
    size, temp = standard_params['size'], standard_params['temp']
    sim = XYSimulation(size=size, temp=temp)
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
    sim = XYSimulation(size=size, temp=1.0)

    # Angles arranged in a loop around the first plaquette (0,0)
    # s(0,0)=0, s(0,1)=pi/2, s(1,1)=pi, s(1,0)=3pi/2
    angles = np.zeros((size, size))
    angles[0, 0] = 0
    angles[0, 1] = np.pi / 2
    angles[1, 1] = np.pi
    angles[1, 0] = 1.5 * np.pi

    sim.spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    vorticity = sim._calculate_vorticity()

    # The (0,0) plaquette should have winding number 1
    assert vorticity[0, 0] == 1.0


def test_xy_vortex_density_single_vortex():
    """Verify XY vortex density matches the non-zero winding fraction from the vorticity map."""
    size = 4
    sim = XYSimulation(size=size, temp=1.0)

    angles = np.zeros((size, size))
    angles[0, 0] = 0.0
    angles[0, 1] = np.pi / 2
    angles[1, 1] = np.pi
    angles[1, 0] = 1.5 * np.pi

    sim.spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    density = sim._get_vortex_density()
    vorticity = sim._calculate_vorticity()
    expected_density = np.count_nonzero(np.abs(vorticity) > 0.0) / (size * size)
    assert density == pytest.approx(expected_density)


def test_xy_vortex_density_uniform_state_is_zero():
    """Verify XY vortex density is zero for a uniform spin field."""
    size = 6
    sim = XYSimulation(size=size, temp=1.0)
    sim.spins = np.zeros((size, size, 2), dtype=np.float64)
    sim.spins[..., 0] = 1.0

    density = sim._get_vortex_density()
    assert density == pytest.approx(0.0)


def test_clock_initialization(standard_params):
    """Verify correct initialization of the q-state clock model."""
    size, temp = standard_params['size'], standard_params['temp']
    q = 6
    sim = ClockSimulation(size=size, temp=temp, q=q)
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


def test_clock_vortex_density_single_vortex():
    """Verify Clock vortex density matches the non-zero winding fraction from the vorticity map."""
    size = 4
    sim = ClockSimulation(size=size, temp=1.0, q=6)

    angles = np.zeros((size, size))
    angles[0, 0] = 0.0
    angles[0, 1] = np.pi / 2
    angles[1, 1] = np.pi
    angles[1, 0] = 1.5 * np.pi

    sim.spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    density = sim._get_vortex_density()
    vorticity = sim._calculate_vorticity()
    expected_density = np.count_nonzero(np.abs(vorticity) > 0.0) / (size * size)
    assert density == pytest.approx(expected_density)


def test_clock_vortex_density_uniform_state_is_zero():
    """Verify Clock vortex density is zero for a uniform spin field."""
    size = 6
    sim = ClockSimulation(size=size, temp=1.0, q=6)
    sim.spins = np.zeros((size, size, 2), dtype=np.float64)
    sim.spins[..., 0] = 1.0

    density = sim._get_vortex_density()
    assert density == pytest.approx(0.0)


def test_correlation_function(standard_params):
    """Verify the calculation of the radially averaged correlation function G(r)."""
    # Test base class method via Ising
    size = standard_params['size']
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


def test_reproducibility():
    """Verify that simulations with the same seed produce identical results."""
    seed = 42
    size = 10
    temp = 2.0
    n_steps = 10

    # Test Ising
    sim1 = IsingSimulation(size=size, temp=temp, seed=seed)
    sim2 = IsingSimulation(size=size, temp=temp, seed=seed)
    m1, e1 = sim1.run(n_steps=n_steps)
    m2, e2 = sim2.run(n_steps=n_steps)
    np.testing.assert_array_equal(sim1.spins, sim2.spins)
    np.testing.assert_array_almost_equal(m1, m2)
    np.testing.assert_array_almost_equal(e1, e2)

    # Test XY
    sim1_xy = XYSimulation(size=size, temp=temp, seed=seed)
    sim2_xy = XYSimulation(size=size, temp=temp, seed=seed)
    m1_xy, e1_xy = sim1_xy.run(n_steps=n_steps)
    m2_xy, e2_xy = sim2_xy.run(n_steps=n_steps)
    np.testing.assert_array_almost_equal(sim1_xy.spins, sim2_xy.spins)
    np.testing.assert_array_almost_equal(m1_xy, m2_xy)
    np.testing.assert_array_almost_equal(e1_xy, e2_xy)

    # Test Clock
    sim1_c = ClockSimulation(size=size, temp=temp, seed=seed)
    sim2_c = ClockSimulation(size=size, temp=temp, seed=seed)
    m1_c, e1_c = sim1_c.run(n_steps=n_steps)
    m2_c, e2_c = sim2_c.run(n_steps=n_steps)
    np.testing.assert_array_almost_equal(sim1_c.spins, sim2_c.spins)
    np.testing.assert_array_almost_equal(m1_c, m2_c)
    np.testing.assert_array_almost_equal(e1_c, e2_c)
