# Multiferroic Simulation Project

This project is a high-performance Python framework for simulating and analyzing phase transitions in 2D lattice models, including **Ising**, **XY**, and **q-state Clock** models. It is designed to study complex physical phenomena like the BKT (Berezinskii-Kosterlitz-Thouless) transition, topological defects (vortices), and domain coarsening dynamics.

## Project Overview

The codebase provides a modular architecture for Monte Carlo simulations:
- **High Performance**: Uses [Numba](https://numba.pydata.org/) JIT compilation to achieve C-like speeds for lattice updates. All measurement loops use pre-allocated NumPy arrays rather than Python lists.
- **Physical Observables**: Calculates magnetization, energy, specific heat, magnetic susceptibility, helicity modulus, and vorticity.
- **Statistical Analysis**: Implements real-space spin-spin correlation functions G(r) and structure factors S(k) via Fast Fourier Transforms (FFT).
- **Domain Coarsening**: Tools for analysing domain growth after a thermal quench, including structure-factor, mean-intercept-length, and correlation-length estimators with power-law fitting.
- **Parallelization**: Supports multi-core temperature sweeps and independent simulation runs using `parallel_sweep` from `utils.system_helpers`.

## Core Technologies

- **Python 3.x** (≥ 3.9)
- **NumPy**: Efficient array operations and FFTs.
- **Numba**: Just-In-Time compilation for the Monte Carlo kernels.
- **Matplotlib**: Visualization of results and spin configurations.
- **tqdm**: Progress tracking for long-running simulations.
- **setuptools**: Project is installable as an editable package via `pyproject.toml`.

## Project Structure

- `models/`: Core simulation logic and model implementations.
  - `simulation_base.py`: Defines the `MonteCarloSimulation` abstract base class (inherits `abc.ABC`; `step()`, `_get_magnetization()`, `_get_energy()`, `_get_structure_factor_squared_unshifted()` are `@abstractmethod`). Also provides shared Numba-accelerated helpers `calculate_vorticity_numba` (winding number calculation for XY/Clock models) and `get_helicity_data_numba` (helicity modulus data).
  - `ising_model.py`: 2D Ising model with Metropolis-Hastings updates. Supports two update schemes via the `update` parameter: `'checkerboard'` (default, faster parallel sweep) and `'random'` (random sequential Metropolis, more physical dynamics for coarsening studies).
  - `xy_model.py`: 2D XY model with continuous spin rotations, vorticity calculation, and helicity modulus measurement.
  - `clock_model.py`: q-state clock model with configurable anisotropy strength `A` and number of clock states `q`, vorticity calculation, and helicity modulus measurement.
- `utils/`: Shared helper functions.
  - `system_helpers.py`: Technical utilities — `ensure_results_dir`, `save_plot`, `parallel_sweep` (multiprocessing with tqdm progress tracking), and `plot_temperature_sweep` (standardised 4-panel thermodynamic plot).
  - `physics_helpers.py`: Functions for calculating physical quantities (`calculate_thermodynamics`, `get_averaged_correlation`).
- `scripts/`: Model-specific analysis and simulation scripts.
  - `ising/`: Temperature sweep, correlation comparison, correlation divergence (critical exponent ν extraction), domain growth analysis (R(t) ~ t^(1/2) Allen-Cahn law verification), and domain snapshot visualisation (spin configurations, S(|k|), and G(r) at multiple time steps).
  - `xy/`: Temperature sweep, helicity modulus (superfluid stiffness / BKT universal jump), BKT transition (vortex proliferation counting), and correlation comparison (power-law vs exponential decay).
  - `clock/`: Temperature sweep and phase transitions.
- `results/`: Directory where simulation plots (PNG) and data are saved, organized by model subfolders.
- `tests/`: Unit tests for models (`test_models.py`) and utilities (`test_utils.py`).
- `pyproject.toml`: Package metadata; install with `pip install -e .` for dependency-free imports.
- `requirements.txt`: Pinned dependency list (`numpy`, `matplotlib`, `tqdm`, `numba`).

## Getting Started

### Installation

Install the package in editable mode (this removes the need for any `sys.path` manipulation):
```bash
pip install -e .
```

### Running Simulations

Standardized temperature sweeps are available for all models. Each script generates a 4-panel plot of thermodynamic observables.

**Example: Run Ising temperature sweep**
```bash
python scripts/ising/temperature_sweep.py
```

**Example: Run XY temperature sweep**
```bash
python scripts/xy/temperature_sweep.py
```

**Example: Run Clock temperature sweep**
```bash
python scripts/clock/temperature_sweep.py
```

**Example: Ising domain growth analysis**
```bash
python scripts/ising/domain_growth.py
```

**Example: Ising domain snapshot visualisation**
```bash
python scripts/ising/domain_snapshots.py
```

Results and visualizations will be saved to the `results/<model>/` directory.

### Running Tests

To run the full test suite, use `pytest` from the project root:
```bash
pytest tests/
```

## Development Conventions

- **Performance**: Always use `@njit` (Numba) for nested loops involving lattice updates or energy calculations. Keep these functions outside of classes to ensure Numba can optimize them effectively. Use pre-allocated NumPy arrays (not Python lists) for any measurement loop.
- **Type Safety**: Use Python type hints project-wide. Prefer modern built-in generic syntax (`list[x]`, `tuple[x, y]`) and union syntax (`X | None`) over the legacy `typing` module equivalents (`List`, `Tuple`, `Optional`). For abstract types such as `Callable`, `Iterable`, `Sequence`, and `Sized`, import from `collections.abc` rather than `typing`.
- **Documentation**: Provide comprehensive docstrings for all modules, classes, and functions. Follow a consistent style (e.g., Google or NumPy style) to describe arguments, return values, and physical formulas where applicable.
- **Shared Utilities**: Place model-independent logic in the `utils/` package. Separate technical system operations (`system_helpers.py`) from physical calculation logic (`physics_helpers.py`). Use `plot_temperature_sweep` from `system_helpers` for the standard 4-panel sweep plot.
- **Imports**: The project is installed as a package (`pip install -e .`). Do not use `sys.path` hacks in scripts — import `models` and `utils` directly.
- **Modularity**: When adding a new model, inherit from `MonteCarloSimulation` in `models/simulation_base.py` and implement all `@abstractmethod` methods (`step`, `_get_magnetization`, `_get_energy`, `_get_structure_factor_squared_unshifted`).
- **Parallelism**: Use `parallel_sweep` from `utils.system_helpers` for sweeps over independent parameters (temperatures, correlation runs, etc.). Worker functions must be defined at module level for multiprocessing pickling.
- **Error handling**: Use specific exception types in `except` clauses — never use bare `except:`.
