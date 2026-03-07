# Multiferroic Simulation Project

This project is a high-performance Python framework for simulating and analyzing phase transitions in 2D lattice models, including **Ising**, **XY**, and **q-state Clock** models. It is designed to study complex physical phenomena like the BKT (Berezinskii-Kosterlitz-Thouless) transition, topological defects (vortices), and phase ordering kinetics.

## Features

- **High Performance**: Uses [Numba](https://numba.pydata.org/) JIT compilation with `fastmath=True` and disk caching for C-like simulation speeds.
- **Scientific Reproducibility**: Deterministic seed management synchronizing both NumPy and Numba RNG states.
- **Ordering Kinetics**: Specialized tools for verifying growth laws (e.g., $L(t) \sim t^{1/2}$) and defect decay (e.g., $n_v \sim t^{-1}$) across all models.
- **Ordering Evolution**: High-resolution multi-row visualizations showing the spatial development of order, vorticity maps, and correlation functions over time.
- **Scaling Benchmarks**: Comprehensive performance profiling suite measuring simulation throughput and analysis overhead across varying lattice sizes.
- **Professional Tooling**: Comprehensive CLI support, structured logging, and unified plotting interfaces.
- **Physical Observables**: Calculates magnetization, energy, specific heat, magnetic susceptibility, helicity modulus, vorticity, and correlation functions.

## Core Technologies

- **Python 3.x** (≥ 3.9)
- **NumPy**: Efficient array operations and FFTs.
- **Numba**: Just-In-Time compilation for core simulation kernels.
- **Matplotlib**: Visualization of results and ordering processes.
- **pytest & pytest-cov**: Testing framework with code coverage analysis.
- **Ruff**: Fast Python linter and code formatter.
- **Mypy**: Static type checker.

## Project Structure

- `models/`: Core simulation logic and model implementations (Ising, XY, Clock).
- `utils/`: Unified physics analysis (`physics_helpers.py`) and technical utilities (`system_helpers.py`).
- `scripts/`: Model-specific analysis tools (Sweeps, Kinetics, Evolution).
- `results/`: Directory where simulation plots and data are saved.
- `tests/`: Comprehensive unit tests for models and utilities.
- `benchmark.py`: Scaling analysis and performance visualization tool.

## Getting Started

### Installation

1. Clone the repository and navigate to the project directory.
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

### Running Simulations

All major scripts support a Command Line Interface (CLI).

**Example: Ordering Kinetics**
```bash
# Study domain coarsening in the Ising model
python scripts/ising/ordering_kinetics.py --size 256 --max-steps 1000 --verbose
```

**Example: Phase Ordering Evolution**
```bash
# Generate 4-column visual evolution for the XY model
python scripts/xy/ordering_evolution.py --size 256 --targets 1 10 100 1000
```

**Example: Scaling Benchmark**
```bash
# Profile all models across multiple lattice sizes
python benchmark.py --sizes 32 64 128 256 --sweeps 200
```

Results and visualizations will be saved to model-specific folders in the `results/` directory.

### Code Quality

Check linting and type safety:
```bash
ruff check .
mypy --explicit-package-bases models/ utils/ scripts/
```

## Development Conventions

- **Performance**: Always use `@njit(cache=True, fastmath=True)` for loops. Avoid modulo operators in PBCs; use pre-calculated indices.
- **Reproducibility**: Use the `seed` parameter when initializing models.
- **Unified Logic**: Prefer shared utilities in `utils/` for kinetics and evolution analysis to ensure project-wide consistency.
- **Logging**: Use the project-wide logger (`setup_logging`) instead of `print()`.
