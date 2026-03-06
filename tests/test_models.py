"""
Unit tests for the Monte Carlo simulation models (Ising, XY, Clock).
"""

import unittest
import numpy as np
from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation
from models.clock_model import ClockSimulation

class TestModels(unittest.TestCase):
    """
    Test suite for verifying model initialization, basic execution, and physical observables.
    """
    def setUp(self) -> None:
        """Set up standard parameters for small-scale testing."""
        self.size = 10
        self.temp = 2.0

    def test_ising_initialization(self) -> None:
        """Verify correct initialization of the Ising model."""
        sim = IsingSimulation(self.size, self.temp)
        self.assertEqual(sim.size, self.size)
        self.assertEqual(sim.temp, self.temp)
        self.assertEqual(sim.spins.shape, (self.size, self.size))
        self.assertTrue(np.all(np.logical_or(sim.spins == 1, sim.spins == -1)))

    def test_ising_step(self) -> None:
        """Verify that a single MC step in the Ising model executes without error."""
        sim = IsingSimulation(self.size, self.temp)
        sim.step()
        self.assertEqual(sim.steps, 1)

    def test_ising_run(self) -> None:
        """Verify that a short simulation run returns the expected number of measurements."""
        sim = IsingSimulation(self.size, self.temp)
        n_steps = 5
        mags, engs = sim.run(n_steps)
        self.assertEqual(len(mags), n_steps)
        self.assertEqual(len(engs), n_steps)
        for m in mags:
            self.assertGreaterEqual(m, 0)
            self.assertLessEqual(m, 1.0)

    def test_ising_low_temp_magnetization(self) -> None:
        """Verify that Ising model maintains high magnetization at very low temperature."""
        # Start from an ordered state (ground state)
        sim = IsingSimulation(size=20, temp=0.1)
        sim.spins = np.ones((20, 20), dtype=np.int8)
        
        sim.equilibrate(100)
        mags, _ = sim.run(100)
        # At T=0.1, it should stay very close to M=1
        self.assertGreater(np.mean(mags), 0.99)

    def test_ising_high_temp_magnetization(self) -> None:
        """Verify that Ising model has low magnetization at very high temperature."""
        # T = 100 is well above Tc ≈ 2.269
        sim = IsingSimulation(size=20, temp=100.0)
        sim.equilibrate(500)
        mags, _ = sim.run(100)
        # For L=20, M ~ 1/sqrt(N) = 1/20 = 0.05. 0.2 is a safe upper bound.
        self.assertLess(np.mean(mags), 0.2)

    def test_xy_initialization(self) -> None:
        """Verify correct initialization and spin normalization of the XY model."""
        sim = XYSimulation(self.size, self.temp)
        self.assertEqual(sim.size, self.size)
        self.assertEqual(sim.spins.shape, (self.size, self.size, 2))
        # Check normalization
        norms = np.linalg.norm(sim.spins, axis=-1)
        np.testing.assert_allclose(norms, 1.0)

    def test_xy_step(self) -> None:
        """Verify that a single MC step in the XY model maintains spin normalization."""
        sim = XYSimulation(self.size, self.temp)
        sim.step()
        self.assertEqual(sim.steps, 1)
        norms = np.linalg.norm(sim.spins, axis=-1)
        np.testing.assert_allclose(norms, 1.0)

    def test_xy_vorticity_detection(self) -> None:
        """Verify that _calculate_vorticity correctly detects a manually placed vortex."""
        # Create a simple vortex at the center of a 4x4 lattice
        # Using a small lattice where indices are easier to control
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
        self.assertEqual(vorticity[0, 0], 1.0)

    def test_clock_initialization(self) -> None:
        """Verify correct initialization of the q-state clock model."""
        q = 6
        sim = ClockSimulation(self.size, self.temp, q=q)
        self.assertEqual(sim.size, self.size)
        self.assertEqual(sim.q, q)
        self.assertEqual(sim.spins.shape, (self.size, self.size, 2))
        norms = np.linalg.norm(sim.spins, axis=-1)
        np.testing.assert_allclose(norms, 1.0)

    def test_clock_step(self) -> None:
        """Verify that a single MC step in the clock model executes correctly."""
        sim = ClockSimulation(self.size, self.temp)
        sim.step()
        self.assertEqual(sim.steps, 1)
        norms = np.linalg.norm(sim.spins, axis=-1)
        np.testing.assert_allclose(norms, 1.0)

    def test_correlation_function(self) -> None:
        """Verify the calculation of the radially averaged correlation function G(r)."""
        # Test base class method via Ising
        sim = IsingSimulation(self.size, self.temp)
        r, g_r = sim._calculate_correlation_function()
        self.assertEqual(len(r), self.size // 2)
        self.assertEqual(len(g_r), self.size // 2)
        self.assertAlmostEqual(g_r[0], 1.0)

    def test_invalid_initialization(self) -> None:
        """Verify that models raise ValueError for invalid parameters."""
        # Test size validation
        with self.assertRaises(ValueError):
            IsingSimulation(size=0, temp=1.0)
        with self.assertRaises(ValueError):
            IsingSimulation(size=-5, temp=1.0)

        # Test temperature validation
        with self.assertRaises(ValueError):
            IsingSimulation(size=10, temp=0.0)
        with self.assertRaises(ValueError):
            IsingSimulation(size=10, temp=-1.0)

        # Test Ising update scheme validation
        with self.assertRaises(ValueError):
            IsingSimulation(size=10, temp=1.0, update='invalid_scheme')

        # Test Clock model q-state validation
        with self.assertRaises(ValueError):
            ClockSimulation(size=10, temp=1.0, q=1)

if __name__ == '__main__':
    unittest.main()
