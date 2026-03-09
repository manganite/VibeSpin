# Multiferroic Simulation Project

A high-performance Python framework for simulating and analyzing phase transitions in 2D lattice models, including **Ising**, **XY**, and **q-state Clock** models. Designed for studying complex physical phenomena like the BKT transition and phase ordering kinetics.

## Features

- **High Performance**: Numba JIT-compiled kernels achieving C-like simulation speeds.
- **Scientific Rigor**: Deterministic seed management for perfect reproducibility.
- **Deep Analysis**: Specialized tools for verifying growth laws ($L(t) \sim t^{1/2}$) and defect decay.
- **Visualization**: High-resolution snapshots of spatial order, vorticity maps, and correlations.
- **Extensive Metrics**: Calculates magnetization, energy, susceptibility, helicity modulus, and more.

## Core Technologies

- **Python 3.x** (≥ 3.9)
- **NumPy & Numba**: Numerical computation and JIT acceleration.
- **Matplotlib**: Data visualization and physical snapshots.
- **pytest**: Comprehensive testing suite.
- **Ruff & Mypy**: Code quality and type safety.

## Getting Started

### Installation

```bash
pip install -e .
```

### Quick Start Examples

**Study Phase Ordering (Ising Model)**
```bash
python scripts/ising/ordering_kinetics.py --size 256 --max-steps 1000
```

**Visualize Vortex Dynamics (XY Model)**
```bash
python scripts/xy/ordering_evolution.py --size 256 --targets 1 10 100 1000
```

**Run Scaling Benchmarks**
```bash
python benchmark.py --sizes 64 128 256
```

## Development Conventions

- **Numba**: Always use `@njit(cache=True, fastmath=True)` for simulation kernels.
- **Reproducibility**: Use the `seed` parameter when initializing models.
- **Logging**: Use `utils/system_helpers.py:setup_logging` instead of standard `print()`.
- **Quality**: Run `ruff check .` and `mypy` before submitting changes.

For detailed developer and AI agent instructions, please refer to [AGENTS.md](AGENTS.md).
