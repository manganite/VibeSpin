# Multiferroic Simulation Project

This project is a high-performance Python framework for simulating and analyzing phase transitions in 2D lattice models, including **Ising**, **XY**, and **q-state Clock** models. It is designed to study complex physical phenomena like the BKT (Berezinskii-Kosterlitz-Thouless) transition, topological defects (vortices), and domain coarsening dynamics.

## Features

- **High Performance**: Uses [Numba](https://numba.pydata.org/) JIT compilation to achieve C-like speeds for lattice updates.
- **Physical Observables**: Calculates magnetization, energy, specific heat, magnetic susceptibility, helicity modulus, and vorticity.
- **Statistical Analysis**: Implements real-space spin-spin correlation functions G(r) and structure factors S(k) via Fast Fourier Transforms (FFT).
- **Domain Coarsening**: Tools for analysing domain growth after a thermal quench.
- **Parallelization**: Supports multi-core temperature sweeps and independent simulation runs.

## Core Technologies

- **Python 3.x** (≥ 3.9)
- **NumPy**: Efficient array operations and FFTs.
- **Numba**: Just-In-Time compilation for the Monte Carlo kernels.
- **Matplotlib**: Visualization of results and spin configurations.
- **tqdm**: Progress tracking for long-running simulations.
- **pytest**: Modern testing framework for unit and integration tests.
- **Ruff**: Extremely fast Python linter and code formatter.

## Project Structure

- `models/`: Core simulation logic and model implementations.
- `utils/`: Shared helper functions for system operations and physics calculations.
- `scripts/`: Model-specific analysis and simulation scripts.
- `results/`: Directory where simulation plots and data are saved.
- `tests/`: Unit tests for models and utilities.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd multiferroic
   ```
3. Install the package in editable mode with development dependencies:
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

Results and visualizations will be saved to the `results/<model>/` directory.

### Running Tests

To run the full test suite, use `pytest` from the project root:
```bash
pytest tests/
```

### Code Quality

To maintain high code quality, this project uses **Ruff** for linting and formatting. You can run these tools from the project root:

**Check for linting issues:**
```bash
ruff check .
```

**Automatically fix issues and format code:**
```bash
ruff check . --fix
ruff format .
```

## Development Conventions

- **Performance**: Use `@njit` (Numba) for nested loops involving lattice updates or energy calculations.
- **Type Safety**: Use Python type hints project-wide.
- **Documentation**: Provide comprehensive docstrings for all modules, classes, and functions.
- **Modularity**: When adding a new model, inherit from `MonteCarloSimulation` in `models/simulation_base.py`.
- **Parallelism**: Use `parallel_sweep` from `utils.system_helpers` for sweeps over independent parameters.
