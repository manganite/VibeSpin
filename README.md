# Multiferroic Simulation Project

This project is a high-performance Python framework for simulating and analyzing phase transitions in 2D lattice models, including **Ising**, **XY**, and **q-state Clock** models. It is designed to study complex physical phenomena like the BKT (Berezinskii-Kosterlitz-Thouless) transition, topological defects (vortices), and domain coarsening dynamics.

## Features

- **High Performance**: Uses [Numba](https://numba.pydata.org/) JIT compilation with `fastmath=True` and disk caching.
- **Scientific Reproducibility**: Deterministic seed management for both NumPy and Numba.
- **Professional Tooling**: CLI support for all scripts and structured logging.
- **Physical Observables**: Calculates magnetization, energy, specific heat, magnetic susceptibility, helicity modulus, and vorticity.
- **Statistical Analysis**: Fast spin-spin correlation functions G(r) and structure factors S(k).
- **Domain Coarsening**: Specialized tools for verifying growth laws (e.g., $R(t) \sim t^{1/2}$).

## Core Technologies

- **Python 3.x** (≥ 3.9)
- **NumPy**: Efficient array operations and FFTs.
- **Numba**: Just-In-Time compilation for core kernels.
- **Matplotlib**: Visualization of results and spin configurations.
- **pytest & pytest-cov**: Testing framework with code coverage analysis.
- **Ruff**: Fast Python linter and code formatter.
- **Mypy**: Static type checker.

## Project Structure

- `models/`: Core simulation logic and model implementations.
- `utils/`: Shared helper functions for system operations and physics calculations.
- `scripts/`: Model-specific analysis and simulation scripts.
- `results/`: Directory where simulation plots and data are saved.
- `tests/`: Unit tests for models and utilities.

## Getting Started

### Installation

1. Clone the repository and navigate to the project directory.
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

### Running Simulations

All major scripts support a Command Line Interface (CLI).

**Example: Run Ising temperature sweep**
```bash
python scripts/ising/temperature_sweep.py --size 64 --t-points 40 --verbose
```

**Example: Run XY helicity modulus analysis**
```bash
python scripts/xy/helicity_modulus.py --size 64 --meas-steps 50000
```

Results and visualizations will be saved to the `results/` directory.

### Running Tests

To run the full test suite with coverage:
```bash
pytest --cov
```

### Code Quality

Check linting and type safety:
```bash
ruff check .
mypy --explicit-package-bases models/ utils/
```

## Development Conventions

- **Performance**: Always use `@njit(cache=True, fastmath=True)` for nested loops.
- **Reproducibility**: Use the `seed` parameter when initializing models for deterministic results.
- **Logging**: Use the project-wide logger (`setup_logging`).
- **Type Safety**: Use modern Python type hints project-wide.
