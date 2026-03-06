"""
Unit tests for utility functions in utils/physics_helpers.py and utils/system_helpers.py.
"""

import unittest
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no display required
import matplotlib.pyplot as plt

from utils.physics_helpers import calculate_thermodynamics, get_averaged_correlation
from utils.system_helpers import ensure_results_dir, plot_temperature_sweep, parallel_sweep, save_plot
from models.ising_model import IsingSimulation


def _square_worker(x: int) -> int:
    """Module-level worker function for parallel_sweep test."""
    return x * x


class TestCalculateThermodynamics(unittest.TestCase):
    """Tests for calculate_thermodynamics."""

    def test_returns_four_floats(self) -> None:
        """Should return exactly four float values."""
        mags = np.array([0.8, 0.9, 0.85, 0.82])
        engs = np.array([-1.5, -1.6, -1.55, -1.52])
        result = calculate_thermodynamics(mags, engs, T=2.0, L=10)
        self.assertEqual(len(result), 4)
        for val in result:
            self.assertIsInstance(val, float)

    def test_average_magnetization(self) -> None:
        """avg_mag should be the mean of the input magnetization array."""
        mags = np.array([0.4, 0.6, 0.8, 1.0])
        engs = np.array([-1.0, -1.0, -1.0, -1.0])
        avg_mag, _, _, _ = calculate_thermodynamics(mags, engs, T=2.0, L=5)
        self.assertAlmostEqual(avg_mag, float(np.mean(mags)))

    def test_average_energy(self) -> None:
        """avg_eng should be the mean of the input energy array."""
        mags = np.array([0.5, 0.5])
        engs = np.array([-2.0, -4.0])
        _, avg_eng, _, _ = calculate_thermodynamics(mags, engs, T=1.0, L=4)
        self.assertAlmostEqual(avg_eng, -3.0)

    def test_susceptibility_zero_variance(self) -> None:
        """Susceptibility should be zero for constant magnetization (no fluctuations)."""
        mags = np.ones(20) * 0.9
        engs = np.ones(20) * -1.5
        _, _, susc, _ = calculate_thermodynamics(mags, engs, T=1.0, L=10)
        self.assertAlmostEqual(susc, 0.0)

    def test_specific_heat_zero_variance(self) -> None:
        """Specific heat should be zero for constant energy (no fluctuations)."""
        mags = np.ones(20) * 0.5
        engs = np.ones(20) * -1.0
        _, _, _, spec_h = calculate_thermodynamics(mags, engs, T=1.0, L=10)
        self.assertAlmostEqual(spec_h, 0.0)

    def test_susceptibility_scales_with_n(self) -> None:
        """Susceptibility (chi = N * Var(M) / T) should scale with lattice size N = L^2."""
        mags = np.array([0.0, 1.0])   # variance = 0.25
        engs = np.array([-1.0, -1.0])
        T, L = 1.0, 10
        _, _, susc, _ = calculate_thermodynamics(mags, engs, T=T, L=L)
        expected = (L**2) * np.var(mags) / T
        self.assertAlmostEqual(susc, expected)

    def test_invalid_inputs(self) -> None:
        """calculate_thermodynamics should raise ValueError for invalid T or L."""
        mags = np.ones(10)
        engs = np.ones(10)
        # Test temperature validation
        with self.assertRaises(ValueError):
            calculate_thermodynamics(mags, engs, T=0, L=10)
        with self.assertRaises(ValueError):
            calculate_thermodynamics(mags, engs, T=-1, L=10)
        # Test lattice size validation
        with self.assertRaises(ValueError):
            calculate_thermodynamics(mags, engs, T=1, L=0)
        with self.assertRaises(ValueError):
            calculate_thermodynamics(mags, engs, T=1, L=-5)


class TestGetAveragedCorrelation(unittest.TestCase):
    """Tests for get_averaged_correlation."""

    def setUp(self) -> None:
        """Small Ising simulation for correlation tests."""
        self.sim = IsingSimulation(size=10, temp=2.0)

    def test_returns_two_arrays(self) -> None:
        """Should return a tuple of two numpy arrays."""
        r, G_r = get_averaged_correlation(self.sim, total_steps=20, sample_interval=5)
        self.assertIsInstance(r, np.ndarray)
        self.assertIsInstance(G_r, np.ndarray)

    def test_output_lengths_match(self) -> None:
        """r and G_r must have the same length."""
        r, G_r = get_averaged_correlation(self.sim, total_steps=20, sample_interval=5)
        self.assertEqual(len(r), len(G_r))

    def test_normalization_at_zero(self) -> None:
        """G(0) should be 1 (normalized by definition)."""
        r, G_r = get_averaged_correlation(self.sim, total_steps=20, sample_interval=5)
        self.assertAlmostEqual(G_r[0], 1.0, places=5)

    def test_output_length_is_half_lattice(self) -> None:
        """Length of r should be size // 2 (radial profile up to half the box)."""
        r, G_r = get_averaged_correlation(self.sim, total_steps=10, sample_interval=5)
        self.assertEqual(len(r), self.sim.size // 2)

    def test_invalid_inputs(self) -> None:
        """get_averaged_correlation should raise ValueError for invalid total_steps or sample_interval."""
        with self.assertRaises(ValueError):
            get_averaged_correlation(self.sim, total_steps=-1, sample_interval=1)
        with self.assertRaises(ValueError):
            get_averaged_correlation(self.sim, total_steps=10, sample_interval=0)


class TestSystemHelpers(unittest.TestCase):
    """Tests for system utility functions."""

    def setUp(self) -> None:
        """Ensure a clean test results directory."""
        self.test_dir = 'test_results'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def tearDown(self) -> None:
        """Cleanup test results directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_ensure_results_dir(self) -> None:
        """ensure_results_dir should create a directory if it does not exist."""
        path = ensure_results_dir(self.test_dir)
        self.assertEqual(path, self.test_dir)
        self.assertTrue(os.path.isdir(self.test_dir))

    def test_save_plot(self) -> None:
        """save_plot should create a PNG file in the specified directory."""
        plt.figure()
        plt.plot([1, 2], [1, 2])
        filename = 'test_plot.png'
        save_plot(filename, directory=self.test_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, filename)))
        plt.close()

    def test_parallel_sweep(self) -> None:
        """parallel_sweep should correctly execute a worker function across parameters."""
        params = [1, 2, 3, 4, 5]
        results = parallel_sweep(_square_worker, params, num_processes=2)
        self.assertEqual(results, [1, 4, 9, 16, 25])


class TestPlotTemperatureSweep(unittest.TestCase):
    """Tests for plot_temperature_sweep."""

    def tearDown(self) -> None:
        """Close all figures after each test to avoid resource warnings."""
        plt.close('all')

    def test_creates_figure_with_four_axes(self) -> None:
        """Should produce a figure containing exactly 4 axes."""
        temps = np.linspace(0.5, 3.0, 10)
        dummy = np.ones(10)
        plot_temperature_sweep(
            temps, dummy, dummy, dummy, dummy,
            title='Test', filename='_test.png', directory='results',
        )
        fig = plt.gcf()
        self.assertEqual(len(fig.axes), 4)

    def test_runs_without_error(self) -> None:
        """plot_temperature_sweep should not raise for well-formed inputs."""
        temps = np.array([1.0, 2.0, 3.0])
        data = np.array([0.5, 0.3, 0.1])
        try:
            plot_temperature_sweep(
                temps, data, data, data, data,
                title='Smoke test', filename='_smoke.png', directory='results',
            )
        except Exception as exc:
            self.fail(f"plot_temperature_sweep raised unexpectedly: {exc}")


if __name__ == '__main__':
    unittest.main()
