"""
Unit tests for physics-related utility functions in utils/physics_helpers.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.ising_model import IsingSimulation
from utils.physics_helpers import (
    calculate_thermodynamics,
    compute_kinetics_metrics,
    get_averaged_correlation,
    pair_correlation_x,
    power_fit,
    radial_average_sk,
)


def test_calculate_thermodynamics_basic():
    """Verify thermodynamic observable calculations with known values."""
    mags = np.array([0.5, 0.7, 0.6])
    engs = np.array([-1.2, -1.0, -1.1])
    T = 2.0
    L = 10

    avg_mag, avg_eng, chi, cv = calculate_thermodynamics(mags=mags, engs=engs, T=T, L=L)

    assert avg_mag == pytest.approx(0.6)
    assert avg_eng == pytest.approx(-1.1)

    # chi = N * Var(M) / T = 100 * Var([0.5, 0.7, 0.6]) / 2.0
    # Var = ((0.5-0.6)^2 + (0.7-0.6)^2 + (0.6-0.6)^2) / 3 = (0.01 + 0.01 + 0) / 3 = 0.02 / 3
    # chi = 100 * (0.02 / 3) / 2.0 = 50 * 0.02 / 3 = 1.0 / 3 = 0.333...
    assert chi == pytest.approx(100 * np.var(mags) / T)

    # Cv = N * Var(E) / T^2 = 100 * Var([-1.2, -1.0, -1.1]) / 4.0
    # Var = (( -0.1)^2 + (0.1)^2 + 0) / 3 = 0.02 / 3
    # Cv = 100 * (0.02 / 3) / 4.0 = 25 * 0.02 / 3 = 0.5 / 3 = 0.1666...
    assert cv == pytest.approx(100 * np.var(engs) / (T**2))

def test_calculate_thermodynamics_invalid_params():
    """Verify that calculate_thermodynamics raises ValueError for invalid inputs."""
    mags = np.array([0.5])
    engs = np.array([-1.0])

    with pytest.raises(ValueError, match="T must be positive"):
        calculate_thermodynamics(mags=mags, engs=engs, T=0.0, L=10)

    with pytest.raises(ValueError, match="L must be a positive integer"):
        calculate_thermodynamics(mags=mags, engs=engs, T=1.0, L=0)

def test_get_averaged_correlation():
    """Verify correlation averaging over multiple steps."""
    size = 10
    sim = IsingSimulation(size=size, temp=1.0, seed=42)
    total_steps = 10
    sample_interval = 5

    r, g_avg = get_averaged_correlation(
        sim=sim, total_steps=total_steps, sample_interval=sample_interval
    )

    assert len(r) == size // 2
    assert len(g_avg) == size // 2
    assert g_avg[0] == pytest.approx(1.0)
    assert sim.steps == total_steps

def test_get_averaged_correlation_invalid_params():
    """Verify validation for get_averaged_correlation parameters."""
    sim = IsingSimulation(size=10, temp=1.0)
    with pytest.raises(ValueError, match="sample_interval must be >= 1"):
        get_averaged_correlation(sim=sim, total_steps=10, sample_interval=0)
    with pytest.raises(ValueError, match="total_steps must be non-negative"):
        get_averaged_correlation(sim=sim, total_steps=-1, sample_interval=1)

def test_radial_average_sk_ising():
    """Verify radial structure factor for an Ising lattice."""
    size = 16
    spins = np.ones((size, size)) # Uniform state
    k, sk = radial_average_sk(spins=spins)

    assert len(k) == size // 2 + 1
    # For uniform spins, S(0) should be N^2. 16*16 = 256.
    assert sk[0] == pytest.approx(float(size * size))
    assert np.all(sk[1:] < 1e-10)

def test_radial_average_sk_vector():
    """Verify radial structure factor for vector spins (XY/Clock)."""
    size = 16
    # Uniform vector spins pointing along x
    spins = np.zeros((size, size, 2))
    spins[..., 0] = 1.0
    k, sk = radial_average_sk(spins=spins)

    assert sk[0] == pytest.approx(float(size * size))
    assert np.all(sk[1:] < 1e-10)

def test_pair_correlation_x():
    """Verify pair correlation calculation along x-axis."""
    size = 16
    # Case 1: All spins same
    spins = np.ones((size, size))
    r, g = pair_correlation_x(spins=spins)
    assert np.allclose(g, 1.0)

    # Case 2: Checkerboard along x
    # s[i, j] = (-1)^j
    spins = np.indices((size, size))[1] % 2
    spins = 2 * spins - 1 # transform to -1, 1
    r, g = pair_correlation_x(spins=spins)
    # G(r) should be (-1)^r
    expected = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(r))])
    assert np.allclose(g, expected)

def test_compute_kinetics_metrics():
    """Verify R_sk and xi metrics for a known state."""
    size = 20
    sim = IsingSimulation(size=size, temp=1.0)
    # Uniform state
    sim.spins = np.ones((size, size))
    metrics = compute_kinetics_metrics(sim=sim)

    assert 'R_sk' in metrics
    assert 'xi' in metrics
    # For uniform state, xi should be large (half-box size)
    assert metrics['xi'] == pytest.approx(size // 2)

def test_power_fit():
    """Verify log-log power law fitting."""
    t = np.array([1.0, 10.0, 100.0])
    # y = 2 * t^0.5
    y = 2.0 * (t**0.5)
    mask = np.ones_like(t, dtype=bool)

    exponent, prefactor = power_fit(t_arr=t, y_arr=y, mask=mask)
    assert exponent == pytest.approx(0.5)
    assert prefactor == pytest.approx(2.0)

    # Test insufficient data
    exp2, pref2 = power_fit(t_arr=t, y_arr=y, mask=np.array([True, True, False]))
    assert exp2 is None
    assert pref2 is None
