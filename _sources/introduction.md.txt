# VibeSpin

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://manganite.github.io/VibeSpin/)
[![Tests](https://github.com/manganite/VibeSpin/actions/workflows/tests.yml/badge.svg)](https://github.com/manganite/VibeSpin/actions/workflows/tests.yml)

VibeSpin is a Python framework for high-performance simulation and analysis of two-dimensional lattice spin models. The codebase focuses on three foundational systems: the **Ising model**, the **XY model**, and the **q-state Clock model** (in both continuous and discrete representations). It combines Numba-accelerated Monte Carlo dynamics with a robust analysis suite for equilibrium observables, coarsening kinetics, and topological defect tracking.

The implementation is optimized for speed, scalability, and physical repeatability. Core kernels utilize **Numba JIT compilation** with optional **multi-core parallelization**, periodic boundaries are handled via precomputed index arrays, and all stochastic trajectories are fully deterministic when seeded.

## Scope and methods

VibeSpin supports both **Checkerboard Updates** (optimized for equilibrium throughput and SIMD vectorization) and **Random Site Selection** (mandatory for non-equilibrium kinetics and aging studies). 

### Physical Analysis
- **Thermodynamics**: Magnetization magnitude $|M|$, total energy $E$, susceptibility $\chi$, and specific heat $C_v$.
- **Spatial Diagnostics**: Radially averaged spin-spin correlation functions $G(r)$ and 2D structure factor $S(k)$ mapping.
- **Topological Analysis**: Directed phase-wrapping for vorticity maps, vortex density tracking, and helicity modulus calculations.
- **Kinetics**: Integrated autocorrelation time $\tau_{\text{int}}$ and phase-ordering growth law extraction.

## Installation

For standard simulation use:

```bash
pip install -e .
```

For full development capabilities (benchmarking, tests, and documentation):

```bash
pip install -e ".[dev,notebook,docs]"
pre-commit install
```

## Benchmarking & Performance

VibeSpin includes a comprehensive performance analysis suite. The benchmark tool measures throughput (sweeps/s), identifies hardware-bound scaling regimes (ns/site), and quantifies the overhead of thermodynamic vs. topological measurements.

```bash
# Run a scaling benchmark across multiple lattice sizes
python benchmark.py --sizes 128 256 512 1024 --sweeps 100
```

Key performance features include:
- **Parallel Numba**: Checkerboard updates can be distributed across multiple CPU cores using `parallel=True`.
- **Discrete Representation**: The discrete Clock model replaces trigonometric evaluations with integer lookups, providing up to a ~2.5x speedup over continuous variants.
- **Pure Metrics**: Benchmarking isolates **Pure Simulation Time** from measurement overhead, allowing for deep algorithmic profiling.

## Typical usage

Launch an equilibrium temperature sweep for the XY model:

```bash
python scripts/xy/temperature_sweep.py --size 64 --t-min 0.2 --t-max 1.5 --t-points 20
```

Investigate phase-ordering kinetics in the Ising model using random-site updates:

```bash
python scripts/ising/ordering_kinetics.py --size 512 --max-steps 5000
```

Generate a visual ordering evolution for the XY model:

```bash
python scripts/xy/ordering_evolution.py --size 256 --targets 1 10 100 1000 5000
```

## Development guidance

VibeSpin maintains rigorous engineering and physical standards. All simulation code must strictly adhere to **Metropolis-Hastings prerequisites**, including detailed balance and ergodicity.

### Kernel Constraints
- Simulation kernels must use `@njit(cache=True, fastmath=True)` and minimize memory allocation.
- Periodic boundaries must use `idx_next` and `idx_prev` arrays (no `%` or `np.mod` in inner loops).
- Models must sync Numba's internal RNG using `_seed_numba(seed)`.

### Verification Suite
Before proposing changes, ensure all verification checks pass:

```bash
pytest
ruff check .
mypy --explicit-package-bases models/ utils/ scripts/
```

## Documentation

Full documentation is available at **[https://manganite.github.io/VibeSpin/](https://manganite.github.io/VibeSpin/)**.

To build the documentation locally:

```bash
cd docs
make html
```

For deeper insights, refer to the source guides:
- {doc}`Physics and Algorithm Guide <physics>`: Detailed explanation of physical models, observables, and algorithm prerequisites.
- {doc}`Scripts Catalog <scripts>`: Comprehensive catalog of entry-point scripts.
- {doc}`Agent Instruction Guide <agents>`: Mandatory technical constraints for AI Agents.
- {doc}`Performance Benchmarks <benchmarks>`: Detailed scaling analysis.

## Project context

VibeSpin was developed using AI-assisted scientific coding workflows. The framework demonstrates how high-level physical design and validation can be accelerated through iterative modeling, benchmarking, and automated testing.

For detailed procedural instructions, see {doc}`Agent Instruction Guide <agents>`.
