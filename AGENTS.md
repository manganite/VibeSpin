# Multiferroic Simulation Project

This project is a high-performance Python framework for simulating and analyzing phase transitions in 2D lattice models, including **Ising**, **XY**, and **q-state Clock** models. It is designed to study complex physical phenomena like the BKT (Berezinskii-Kosterlitz-Thouless) transition, topological defects (vortices), and phase ordering kinetics.

## Project Overview

The codebase provides a modular architecture for Monte Carlo simulations:
- **High Performance**: Uses [Numba](https://numba.pydata.org/) JIT compilation with `fastmath=True` and `cache=True` to achieve C-like speeds. Startup time is minimized via disk caching.
- **Optimized Kernels**: Implements pre-calculated neighbor indices for Periodic Boundary Conditions (PBC) and vectorized angle extraction for vorticity analysis.
- **Scientific Reproducibility**: Implements deterministic seed management across both NumPy and Numba random number generators to ensure simulation results are perfectly replicable.
- **Professional Tooling**: Comprehensive Command Line Interface (CLI) via `argparse` and structured logging for all simulation scripts.
- **Unified Analysis**: Shared utilities for power-law fitting, length-scale extraction ($R_{S(k)}$, $\xi$), and complex multi-row plotting.
- **Performance Profiling**: Dedicated scaling benchmark suite measuring simulation throughput and analysis costs across multiple lattice sizes.

## Project Structure

- `models/`: Core simulation logic and model implementations.
  - `simulation_base.py`: Defines the `MonteCarloSimulation` abstract base class. Provides shared Numba-accelerated helpers and pre-calculates PBC indices and radial masks.
  - `ising_model.py`: 2D Ising model with Metropolis-Hastings updates.
  - `xy_model.py`: 2D XY model with continuous spin rotations, vorticity calculation, and helicity modulus measurement.
  - `clock_model.py`: q-state clock model with configurable anisotropy strength `A` and states `q`.
- `utils/`: Shared helper functions.
  - `system_helpers.py`: Technical utilities — `setup_logging`, `parallel_sweep`, `plot_ordering_kinetics`, and `plot_ordering_evolution`.
  - `physics_helpers.py`: Functions for physical metrics (`compute_kinetics_metrics`, `radial_average_sk`, `power_fit`).
- `scripts/`: Model-specific analysis and simulation scripts.
  - `ordering_kinetics.py`: Quantitative growth laws and defect decay analysis.
  - `ordering_evolution.py`: Visual snapshots of spatial order development.
  - `temperature_sweep.py`: Standard thermodynamic sweeps (M, E, chi, Cv).
- `results/`: Directory where simulation plots (PNG) and data are saved.
- `tests/`: Comprehensive unit tests for models, utilities, and reproducibility.
- `benchmark.py`: Scaling analysis and performance visualization tool.

## Getting Started

### Installation

Install the package in editable mode:
```bash
pip install -e .
```

### Running Simulations

All major scripts support a CLI for easy parameter configuration.

**Example: Ordering Kinetics**
```bash
python scripts/xy/ordering_kinetics.py --size 256 --max-steps 1000 --verbose
```

**Example: Order Evolution Visuals**
```bash
python scripts/clock/ordering_evolution.py --size 256 --targets 1 10 100 1000
```

## Development Conventions

- **Performance**: Always use `@njit(cache=True, fastmath=True)` for nested loops. Use pre-calculated indices for PBCs to avoid modulo overhead.
- **Reproducibility**: Models must accept an optional `seed` parameter. Ensure `_seed_numba()` is called to synchronize Numba's internal RNG state.
- **Consistency**: Use unified utilities in `utils/` for kinetics and plotting. Distinguish between **Domain Coarsening** (Ising) and **Phase Ordering Dynamics** (XY/Clock).
- **Logging**: Use the project-wide logger (`setup_logging`) and provide progress tracking via `tqdm`.
- **Type Safety**: Enforced via Mypy; modern type hints (e.g., `X | None`, `list[int]`) are preferred.
