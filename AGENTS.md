# Multiferroic Simulation Project

This project is a high-performance Python framework for simulating and analyzing phase transitions in 2D lattice models, including **Ising**, **XY**, and **q-state Clock** models. It is designed to study complex physical phenomena like the BKT (Berezinskii-Kosterlitz-Thouless) transition, topological defects (vortices), and domain coarsening dynamics.

## Project Overview

The codebase provides a modular architecture for Monte Carlo simulations:
- **High Performance**: Uses [Numba](https://numba.pydata.org/) JIT compilation with `fastmath=True` and `cache=True` to achieve C-like speeds. Startup time is minimized via disk caching.
- **Optimized Kernels**: Implements pre-calculated neighbor indices for Periodic Boundary Conditions (PBC) and vectorized angle extraction for vorticity analysis.
- **Scientific Reproducibility**: Implements deterministic seed management across both NumPy and Numba random number generators to ensure simulation results are perfectly replicable.
- **Professional Tooling**: Comprehensive Command Line Interface (CLI) via `argparse` and structured logging for all simulation scripts.
- **Physical Observables**: Calculates magnetization, energy, specific heat, magnetic susceptibility, helicity modulus, and vorticity.
- **Statistical Analysis**: Implements radially averaged correlation functions G(r) using pre-calculated masks for efficiency.

## Core Technologies

- **Python 3.x** (≥ 3.9)
- **NumPy**: Efficient array operations and FFTs.
- **Numba**: Just-In-Time compilation for core kernels.
- **Matplotlib**: Visualization of results and spin configurations.
- **pytest & pytest-cov**: Testing framework with code coverage analysis.
- **Ruff**: Fast Python linter and formatter.
- **Mypy**: Static type checker for Python.
- **setuptools**: Project is installable as an editable package via `pyproject.toml`.

## Project Structure

- `models/`: Core simulation logic and model implementations.
  - `simulation_base.py`: Defines the `MonteCarloSimulation` abstract base class. Provides shared Numba-accelerated helpers and pre-calculates PBC indices and radial masks.
  - `ising_model.py`: 2D Ising model with Metropolis-Hastings updates. Supports checkerboard and random sequential updates.
  - `xy_model.py`: 2D XY model with continuous spin rotations, vorticity calculation, and helicity modulus measurement.
  - `clock_model.py`: q-state clock model with configurable anisotropy strength `A` and number of clock states `q`.
- `utils/`: Shared helper functions.
  - `system_helpers.py`: Technical utilities — `setup_logging`, `ensure_results_dir`, `save_plot`, `parallel_sweep`, and `plot_temperature_sweep`.
  - `physics_helpers.py`: Functions for calculating physical quantities (`calculate_thermodynamics`, `get_averaged_correlation`).
- `scripts/`: Model-specific analysis and simulation scripts (all support CLI).
- `results/`: Directory where simulation plots (PNG) and data are saved.
- `tests/`: Unit tests for models and utilities.
- `pyproject.toml`: Package metadata and tool configurations (pytest, coverage, ruff, mypy).

## Getting Started

### Installation

Install the package in editable mode:
```bash
pip install -e .
```

### Running Simulations

All major scripts support a CLI for easy parameter configuration.

**Example: Ising temperature sweep**
```bash
python scripts/ising/temperature_sweep.py --size 64 --t-points 50 --verbose
```

**Example: Domain growth with logging to file**
```bash
python scripts/ising/domain_growth.py --size 512 --log-file growth.log
```

## Development Conventions

- **Performance**: Always use `@njit(cache=True, fastmath=True)` for nested loops. Use pre-calculated `idx_next` and `idx_prev` for PBCs to avoid modulo overhead.
- **Reproducibility**: Models must accept an optional `seed` parameter. Use `self.rng` (a `np.random.Generator`) and ensure `_seed_numba()` is called within `step()` to synchronize Numba's internal state.
- **Logging**: Use the project-wide logger via `setup_logging` from `utils.system_helpers`. Avoid bare `print()` statements in production code.
- **CLI**: Use `argparse` for all new simulation or analysis scripts to allow flexible configuration.
- **Type Safety**: Use modern Python type hints project-wide.
- **Validation & Quality**:
  - Run the full test suite using `pytest --cov`.
  - Ensure code quality by running `ruff check .` and `ruff format .`.
  - Verify type safety by running `mypy --explicit-package-bases models/ utils/`.

## Git & Workflow

- **Commit Style**: Use [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `perf:`).
- **Ignored Files**: Never commit `__pycache__`, `results/`, `.vscode/`, or `.egg-info/`.
- **Environment**: Always keep `.devcontainer/` tracked for reproducible environments.
