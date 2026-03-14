# VibeSpin

VibeSpin is a Python framework for simulation and analysis of two-dimensional lattice spin models. The codebase focuses on three systems, the Ising model, the XY model, and the q-state Clock model. It combines Monte Carlo dynamics with analysis tools for equilibrium observables, coarsening kinetics, topological defects, and finite-size trends.

The implementation is built for speed and repeatability. Core kernels run through Numba JIT compilation, periodic boundaries are handled with precomputed index arrays, and seeded runs are deterministic across repeated executions. This makes side-by-side comparisons across update rules and model families straightforward.

## Scope and methods

The project supports checkerboard and random-site updates, with a clear distinction between equilibrium sweeps and non-equilibrium kinetics. Thermodynamic measurements include magnetization, energy, susceptibility, and heat capacity. Spatial diagnostics include correlation functions, structure factor analysis, vorticity maps, and helicity-related quantities for vector-spin models.

## Installation

```bash
pip install -e .
```

## Typical usage

A temperature sweep for XY equilibrium analysis can be launched with the command below.

```bash
python scripts/xy/temperature_sweep.py --L 32 --T-min 0.2 --T-max 1.5 --steps 10
```

BKT-focused analysis can be started with the following script.

```bash
python scripts/xy/bkt_transition.py --size 64 --temp 0.89
```

Ising phase-ordering kinetics can be explored with:

```bash
python scripts/ising/ordering_kinetics.py --size 256 --max-steps 1000
```

XY ordering evolution with snapshot targets can be run with:

```bash
python scripts/xy/ordering_evolution.py --size 256 --targets 1 10 100 1000
```

## Development guidance

Simulation kernels should remain in `@njit(cache=True, fastmath=True)` functions, and reproducibility-sensitive runs should specify `seed` values. Logging should go through `utils/system_helpers.py:setup_logging` rather than direct print calls. Before proposing a commit, run `pytest`, `ruff check .`, and `mypy --explicit-package-bases models/ utils/ scripts/`.

When a Numba typing failure appears, include the traceback and the kernel source in the debugging prompt. That usually reveals unsupported object usage quickly.

## Project context

This repository was developed with AI-assisted coding workflows in VS Code using Copilot Chat and Gemini CLI during iterative modeling, testing, and documentation work. The goal is to show a practical workflow in which a researcher drives the physical design and validation while AI tools accelerate implementation.

For detailed developer and agent instructions, see [AGENTS.md](AGENTS.md).
