"""Unit tests for shared base simulation utilities and kernels."""
from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from models.simulation_base import (
    MonteCarloSimulation,
    _seed_numba,
    calculate_vortex_density_numba,
    calculate_vorticity_numba,
    get_helicity_data_numba,
)


class _DummySimulation(MonteCarloSimulation):
    """Minimal concrete simulation to exercise base-class behavior."""

    def __init__(self, *, size: int, temp: float, init_state: str = 'random') -> None:
        super().__init__(size=size, temp=temp, init_state=init_state)
        self.spins = np.ones((size, size), dtype=np.int8)
        self._sk = np.ones((size, size), dtype=float)

    def step(self) -> None:
        self.steps += 1

    def _get_magnetization(self) -> float:
        return float(self.steps)

    def _get_energy(self) -> float:
        return -float(self.steps)

    def _get_structure_factor_squared_unshifted(self) -> np.ndarray:
        return self._sk


@njit(cache=True)
def _numba_random() -> float:
    return np.random.random()


def test_base_init_rejects_invalid_init_state() -> None:
    """Base class should reject unknown initialization labels."""
    with pytest.raises(ValueError, match="init_state must be 'random' or 'ordered'"):
        _DummySimulation(size=8, temp=1.0, init_state='checkerboard')


def test_base_accepts_numpy_integer_size() -> None:
    """NumPy integer lattice sizes should be accepted by base-class validation."""
    sim = _DummySimulation(size=np.int64(6), temp=1.5)
    assert sim.size == 6


def test_calculate_structure_factor_applies_shift_and_normalization() -> None:
    """Structure factor should be fft-shifted and normalized by lattice area."""
    sim = _DummySimulation(size=4, temp=1.0)
    sim._sk = np.arange(16, dtype=float).reshape(4, 4)

    out = sim._calculate_structure_factor()
    expected = np.fft.fftshift(sim._sk) / (sim.size**2)

    np.testing.assert_allclose(out, expected)


def test_calculate_correlation_function_zero_profile_is_stable() -> None:
    """Zero structure factor should produce finite all-zero radial correlations."""
    sim = _DummySimulation(size=8, temp=1.0)
    sim._sk = np.zeros((8, 8), dtype=float)

    r, g_r = sim._calculate_correlation_function()

    assert len(r) == sim.size // 2
    assert len(g_r) == sim.size // 2
    np.testing.assert_allclose(g_r, 0.0)


def test_equilibrate_advances_steps() -> None:
    """equilibrate() should call step() exactly n_steps times."""
    sim = _DummySimulation(size=4, temp=1.0)
    sim.equilibrate(n_steps=7)
    assert sim.steps == 7


def test_run_records_magnetization_and_energy() -> None:
    """run() should return stepwise magnetization/energy arrays of requested length."""
    sim = _DummySimulation(size=4, temp=1.0)

    mags, engs = sim.run(n_steps=5)

    np.testing.assert_allclose(mags, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(engs, -np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_vorticity_and_density_helpers_consistent() -> None:
    """Vortex density helper should match non-zero winding fraction from vorticity map."""
    size = 4
    idx_next = np.roll(np.arange(size), -1)
    angles = np.zeros((size, size), dtype=float)
    angles[0, 0] = 0.0
    angles[0, 1] = np.pi / 2
    angles[1, 1] = np.pi
    angles[1, 0] = 1.5 * np.pi
    spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    vorticity = calculate_vorticity_numba(spins=spins, idx_next=idx_next)
    density = calculate_vortex_density_numba(spins=spins, idx_next=idx_next)
    expected_density = np.count_nonzero(np.abs(vorticity) > 0.0) / (size * size)

    assert density == pytest.approx(expected_density)


def test_helicity_data_uniform_state() -> None:
    """Uniform x-aligned spins should have maximal cosine sum and zero sine sum."""
    size = 5
    idx_next = np.roll(np.arange(size), -1)
    spins = np.zeros((size, size, 2), dtype=float)
    spins[..., 0] = 1.0

    cos_sum, sin_sum = get_helicity_data_numba(spins=spins, idx_next=idx_next)

    assert cos_sum == pytest.approx(float(size * size))
    assert sin_sum == pytest.approx(0.0)


def test_seed_numba_makes_numba_random_reproducible() -> None:
    """Seeding Numba RNG should make subsequent JIT random draws reproducible."""
    _seed_numba(seed=12345)
    a = _numba_random()
    _seed_numba(seed=12345)
    b = _numba_random()

    assert a == pytest.approx(b)
