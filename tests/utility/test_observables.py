"""
Unit tests for physics-related utility functions in utils/observables.py.
Covers thermodynamic averages, entropy, autocorrelation, spatial diagnostics,
kinetics metrics, and power-law fitting.
"""
from __future__ import annotations

# mypy: disable-error-code=no-untyped-def
import numpy as np
import pytest

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation
from utils.observables import (
    compute_kinetics_metrics,
    get_averaged_correlation,
    pair_correlation_x,
    radial_average_sk,
)
from utils.statistics import (
    _as_1d_float_array,
    _validate_confidence,
)


@pytest.fixture
def ising_sim():
    """Fixture for a small Ising simulation."""
    return IsingSimulation(size=10, temp=2.0)


# ---- calculate_thermodynamics ----




















# ---- calculate_entropy ----






























# ---- calculate_autocorr ----














# ---- uncertainty helpers ----


















# ---- get_averaged_correlation ----


def test_get_averaged_correlation():
    """Verify correlation averaging over multiple steps."""
    size = 10
    sim = IsingSimulation(size=size, temp=1.0, seed=42)
    r, g_avg = get_averaged_correlation(
        sim=sim, total_steps=10, sample_interval=5
    )
    assert len(r) == size // 2
    assert len(g_avg) == size // 2
    assert g_avg[0] == pytest.approx(1.0)
    assert sim.steps == 10


def test_get_averaged_correlation_returns_two_arrays(ising_sim):
    """Should return a tuple of two numpy arrays."""
    r, G_r = get_averaged_correlation(sim=ising_sim, total_steps=20, sample_interval=5)
    assert isinstance(r, np.ndarray)
    assert isinstance(G_r, np.ndarray)


def test_get_averaged_correlation_output_lengths_match(ising_sim):
    """r and G_r must have the same length."""
    r, G_r = get_averaged_correlation(sim=ising_sim, total_steps=20, sample_interval=5)
    assert len(r) == len(G_r)


def test_normalization_at_zero(ising_sim):
    """G(0) should be 1 (normalized by definition)."""
    r, G_r = get_averaged_correlation(sim=ising_sim, total_steps=20, sample_interval=5)
    assert pytest.approx(G_r[0], abs=1e-5) == 1.0


def test_output_length_is_half_lattice(ising_sim):
    """Length of r should be size // 2 (radial profile up to half the box)."""
    r, G_r = get_averaged_correlation(sim=ising_sim, total_steps=10, sample_interval=5)
    assert len(r) == ising_sim.size // 2


def test_get_averaged_correlation_invalid_params():
    """Verify validation for get_averaged_correlation parameters."""
    sim = IsingSimulation(size=10, temp=1.0)
    with pytest.raises(ValueError, match="sample_interval must be >= 1"):
        get_averaged_correlation(sim=sim, total_steps=10, sample_interval=0)
    with pytest.raises(ValueError, match="total_steps must be non-negative"):
        get_averaged_correlation(sim=sim, total_steps=-1, sample_interval=1)


def test_get_averaged_correlation_invalid_inputs(ising_sim):
    """get_averaged_correlation should raise ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        get_averaged_correlation(sim=ising_sim, total_steps=-1, sample_interval=1)
    with pytest.raises(ValueError):
        get_averaged_correlation(sim=ising_sim, total_steps=10, sample_interval=0)


# ---- radial_average_sk ----


def test_radial_average_sk_ising():
    """Verify radial structure factor for an Ising lattice."""
    size = 16
    spins = np.ones((size, size))
    k, sk = radial_average_sk(spins=spins)
    assert len(k) == size // 2 + 1
    assert sk[0] == pytest.approx(float(size * size))
    assert np.all(sk[1:] < 1e-10)


def test_radial_average_sk_vector():
    """Verify radial structure factor for vector spins (XY/Clock)."""
    size = 16
    spins = np.zeros((size, size, 2))
    spins[..., 0] = 1.0
    k, sk = radial_average_sk(spins=spins)
    assert sk[0] == pytest.approx(float(size * size))
    assert np.all(sk[1:] < 1e-10)


def test_radial_average_sk_returns_arrays():
    """radial_average_sk should return matching-length arrays."""
    spins = np.ones((32, 32))
    k, sk = radial_average_sk(spins=spins)
    assert isinstance(k, np.ndarray)
    assert isinstance(sk, np.ndarray)
    assert len(k) == len(sk)
    assert len(k) > 0


# ---- pair_correlation_x ----


def test_pair_correlation_x():
    """Uniform -> G=1, checkerboard -> G=(-1)^r."""
    size = 16
    spins_uniform = np.ones((size, size))
    r, g = pair_correlation_x(spins=spins_uniform)
    np.testing.assert_allclose(g, 1.0)

    iy, ix = np.indices((size, size))
    spins_checker = np.where((ix + iy) % 2 == 0, 1, -1).astype(np.float64)
    r_c, g_c = pair_correlation_x(spins=spins_checker)
    expected = np.array([(-1) ** rr for rr in range(len(g_c))], dtype=np.float64)
    np.testing.assert_allclose(g_c, expected, atol=1e-10)


def test_pair_correlation_x_ising():
    """pair_correlation_x should work for scalar spins."""
    spins = np.ones((32, 32))
    r, g = pair_correlation_x(spins=spins)
    assert len(r) == len(g)
    assert g[0] == pytest.approx(1.0)


def test_pair_correlation_x_xy():
    """pair_correlation_x should work for vector spins."""
    spins = np.ones((32, 32, 2))
    r, g = pair_correlation_x(spins=spins)
    assert len(r) == len(g)
    assert g[0] == pytest.approx(1.0)


# ---- compute_kinetics_metrics ----


def test_compute_kinetics_metrics():
    """Returns R_sk and xi; xi = L/2 for uniform state (maximum)."""
    size = 16
    sim = IsingSimulation(size=size, temp=1.0)
    sim.spins = np.ones((size, size), dtype=np.int8)
    metrics = compute_kinetics_metrics(sim=sim)
    assert 'R_sk' in metrics
    assert 'xi' in metrics
    assert isinstance(metrics['R_sk'], float)
    assert isinstance(metrics['xi'], float)


def test_compute_kinetics_metrics_xy():
    """compute_kinetics_metrics should return R_sk and xi for XY."""
    sim = XYSimulation(size=16, temp=1.0)
    metrics = compute_kinetics_metrics(sim=sim)
    assert 'R_sk' in metrics
    assert 'xi' in metrics


# ---- Observables: helicity, structure factor, energy ----


def test_xy_helicity_data():
    """Verify helicity data calculation for XY model."""
    size = 10
    sim = XYSimulation(size=size, temp=1.0)
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0
    cos_sum, sin_sum = sim._get_helicity_data()
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)


def test_clock_helicity_data():
    """Verify helicity data calculation for Clock model."""
    size = 10
    sim = ClockSimulation(size=size, temp=0.5, q=6)
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0
    cos_sum, sin_sum = sim._get_helicity_data()
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)


def test_discrete_clock_helicity_data():
    """Verify helicity data calculation for DiscreteClock model."""
    size = 10
    sim = DiscreteClockSimulation(size=size, temp=0.5, q=6)
    sim.spins = np.zeros((size, size), dtype=np.int32)
    cos_sum, sin_sum = sim._get_helicity_data()
    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)


def test_discrete_clock_vorticity_detection():
    """Verify vorticity detection via vector conversion for DiscreteClock."""
    size = 4
    q = 4
    sim = DiscreteClockSimulation(size=size, temp=1.0, q=q)
    sim.spins = np.zeros((size, size), dtype=np.int32)
    sim.spins[0, 0] = 0
    sim.spins[0, 1] = 1
    sim.spins[1, 1] = 2
    sim.spins[1, 0] = 3
    vort = sim._calculate_vorticity()
    assert vort[0, 0] == 1.0


def test_base_structure_factor():
    """Verify structure factor via base class method."""
    size = 8
    sim = IsingSimulation(size=size, temp=1.0)
    sim.spins = np.ones((size, size), dtype=np.int8)
    sf = sim._calculate_structure_factor()
    assert sf.shape == (size, size)
    center = size // 2
    assert sf[center, center] == pytest.approx(float(size * size))


def test_xy_structure_factor_squared():
    """Verify XY structure factor squared (unshifted)."""
    size = 8
    sim = XYSimulation(size=size, temp=1.0)
    sf = sim._get_structure_factor_squared_unshifted()
    assert sf.shape == (size, size)


def test_clock_structure_factor_squared():
    """Verify Clock structure factor squared (unshifted)."""
    size = 8
    sim = ClockSimulation(size=size, temp=1.0, q=6)
    sf = sim._get_structure_factor_squared_unshifted()
    assert sf.shape == (size, size)


def test_discrete_clock_structure_factor_squared():
    """Verify DiscreteClock structure factor squared (unshifted)."""
    size = 8
    sim = DiscreteClockSimulation(size=size, temp=1.0, q=6)
    sf = sim._get_structure_factor_squared_unshifted()
    assert sf.shape == (size, size)


def test_clock_energy_calc():
    """Verify Clock energy returns a float."""
    size = 8
    sim = ClockSimulation(size=size, temp=1.0, q=6)
    sim.spins = np.zeros((size, size, 2))
    sim.spins[..., 0] = 1.0
    e = sim._get_energy()
    assert isinstance(e, float)


# ---- power_fit ----


























class TestPhysicsHelpersValidation:
    """Verify error handling for invalid inputs in observables.py."""

    def test_validate_confidence_errors(self) -> None:
        """Should raise ValueError for confidence outside (0, 1)."""

        with pytest.raises(ValueError, match='confidence must satisfy'):
            _validate_confidence(confidence=0.0)
        with pytest.raises(ValueError, match='confidence must satisfy'):
            _validate_confidence(confidence=1.0)
        with pytest.raises(ValueError, match='confidence must satisfy'):
            _validate_confidence(confidence=1.5)

    def test_as_1d_float_array_errors(self) -> None:
        """Should raise ValueError for non-1D or too small arrays."""

        # 2D array
        with pytest.raises(ValueError, match='must be 1-D'):
            _as_1d_float_array(time_series=np.zeros((2, 2)), name='test')
        # Too small
        with pytest.raises(ValueError, match='at least 2 elements'):
            _as_1d_float_array(time_series=np.array([1.0]), name='test')


