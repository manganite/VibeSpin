# VibeSpin

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://manganite.github.io/VibeSpin/)
[![Tests](https://github.com/manganite/VibeSpin/actions/workflows/tests.yml/badge.svg)](https://github.com/manganite/VibeSpin/actions/workflows/tests.yml)

VibeSpin is a Python framework for high-performance simulation and analysis of two-dimensional lattice spin models. The codebase focuses on three foundational systems: the **Ising model**, the **XY model**, and the **q-state Clock model** — provided as `ClockSimulation` (continuous XY-plus-anisotropy form) and `DiscreteClockSimulation` (integer state indices with cosine lookup tables). It combines Numba-accelerated Monte Carlo dynamics with a robust analysis suite for equilibrium observables, coarsening kinetics, and topological defect tracking.

The implementation is optimized for speed, scalability, and physical repeatability. Core kernels utilize **Numba JIT compilation** with optional **multi-core parallelization**, periodic boundaries are handled via precomputed index arrays, and all stochastic trajectories are fully deterministic when seeded (serial kernels; the optional parallel checkerboard kernels are not seed-reproducible because Numba seeds only the calling thread's RNG).

## Project Policies & Scope Discipline

VibeSpin enforces strict development policies for physical correctness, code quality, reproducibility, and documentation. All edits must follow the **Scope Discipline**: never change, rewrite, or delete code/text unrelated to the current task. See [AGENTS.md](./AGENTS.md) for the full list of mandatory development policies.

## Reference Policy
All references used in VibeSpin are listed in [BIBLIOGRAPHY.md](./BIBLIOGRAPHY.md). Reference policy and link validation are enforced as described in AGENTS.md.

### Background and References
The Ising model [[1]](#bibliography), XY model [[2]](#bibliography), and q-state Clock model [[3]](#bibliography) are canonical systems in statistical physics for studying phase transitions, critical phenomena, and topological defects. Monte Carlo methods, especially the Metropolis-Hastings algorithm [[4]](#bibliography), are standard for simulating these models. The Wolff cluster algorithm [[5]](#bibliography) is highly effective near criticality, reducing autocorrelation times by exploiting collective spin updates. For a comprehensive introduction to these models and algorithms, see the references below.

## Scope and Methods

VibeSpin supports three update schemes tailored to specific physical regimes. **Checkerboard Updates** maximize equilibrium throughput via SIMD vectorization and multi-core execution. **Random Site Selection** is mandatory for non-equilibrium kinetics and aging studies, where preserving the stochastic trajectory is essential for physical validity. The **Wolff Cluster Algorithm** provides high efficiency near critical temperatures by mitigating critical slowing down through collective spin updates.

The framework provides a comprehensive suite of diagnostics for physical analysis. Thermodynamic measurements include magnetization magnitude, total energy, susceptibility, and specific heat. Spatial correlations are analyzed through radially averaged spin-spin correlation functions and 2D structure factor mapping. For topological systems, the engine supports directed phase-wrapping for vorticity maps, vortex density tracking, and helicity modulus calculations. Kinetics studies utilize integrated autocorrelation times and phase-ordering growth law extraction to quantify the temporal evolution of the system.

## Installation

The recommended method for managing this project is using [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

To install dependencies and set up the local virtual environment in editable mode:

```bash
uv sync --all-extras
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

Alternatively, you can install the package using standard pip:

```bash
pip install -e .
```

Or for full development capabilities with standard pip:

```bash
pip install -e ".[dev,notebook,docs]"
pre-commit install --hook-type pre-commit --hook-type pre-push
```

The repository enforces documentation consistency at both commit and push time. The pre-push hook runs link validation, API docs sync checks, and a Sphinx HTML build with warnings treated as errors.

### Devcontainer Jupyter Auto-Start

In this repository's devcontainer, JupyterLab starts automatically and listens on port 8888. The startup path is defined in the repository's [devcontainer.json on GitHub](https://github.com/manganite/VibeSpin/blob/master/.devcontainer/devcontainer.json) and launches the companion [start-jupyter.sh helper on GitHub](https://github.com/manganite/VibeSpin/blob/master/.devcontainer/scripts/start-jupyter.sh), which waits for the project virtual environment to be ready and then starts JupyterLab in detached mode.

After rebuilding the container, verify the server with the venv binary directly:

```bash
/workspaces/vibespin/.venv/bin/jupyter notebook list
```

The startup log is written to `/tmp/jupyter.log`, and lifecycle-hook output is written to `/tmp/devcontainer-poststart.log`.

If no server is listed, inspect both logs first:

```bash
tail -n 200 /tmp/devcontainer-poststart.log
tail -n 200 /tmp/jupyter.log
```

## Benchmarking & Performance

VibeSpin includes a comprehensive performance analysis suite that measures throughput and identifies hardware-bound scaling regimes. The benchmark tool quantifies simulation efficiency in terms of sweeps per second and nanoseconds per site, while also isolating the overhead of thermodynamic and topological measurements from the pure simulation time. This granularity allows for deep algorithmic profiling across different lattice sizes and update schemes.

```bash
# Run a scaling benchmark across multiple lattice sizes
python scripts/benchmarks/throughput.py --sizes 128 256 512 1024 --sweeps 100
```

Key engineering features ensure high performance across all models. Checkerboard updates are distributed across multiple CPU cores using Numba parallelization. For q-state clock models, the discrete representation replaces trigonometric evaluations with integer lookups, providing up to a ~2.5x speedup over continuous variants. Benchmarking results consistently isolate the simulation kernel performance, enabling a clear understanding of numerical cost in various regimes.

## Typical Usage

Launch an equilibrium temperature sweep for the XY model:

```bash
python scripts/xy/temperature_sweep.py --size 64 --t-min 0.2 --t-max 1.5 --t-points 20
```

Temperature sweep scripts use **Two-Start Convergence equilibration**: for every temperature/seed point, two independent simulations (one starting from a random disordered state and one from a fully ordered state) are evolved in parallel. The burn-in continues in chunks (defined by `--eq-probe-steps`) until both smoothed magnetization traces pass a mutual cross-band test: the random-start trace must lie within a band of $\pm k$ standard deviations of the ordered-start tail mean, and vice versa. A sigma floor prevents band collapse when either trace is nearly variance-free.

This protocol includes **Quasi-Steady Stuck Detection** to handle metastable domain-wall trapping in the deep ordered phase. If the random-start trace becomes stranded in a plateau while the ordered-start trace remains stable and cleanly ordered, the system identifies the stuck state and declares convergence early, automatically falling back to the stable ordered-start simulation for measurements. This non-equilibrium diagnostic guarantees that measurements are free from initialization bias. To maximize throughput, individual points are parallelized across the worker pool, ensuring full CPU utilization even for single-seed sweeps. The `--eq-max-steps` flag provides a hard cap on the total equilibration time.

Investigate phase-ordering kinetics in the Ising model using random-site updates:

```bash
python scripts/ising/ordering_kinetics.py --size 512 --max-steps 5000
```

Generate a visual ordering evolution for the XY model:

```bash
python scripts/xy/ordering_evolution.py --size 256 --targets 1 10 100 1000 5000
```

Compare Wolff cluster and Metropolis efficiency near the Ising critical point:

```bash
python scripts/ising/wolff_efficiency.py --size 64 --t-min 1.8 --t-max 3.2 --t-points 20
```

Measure the dynamical critical exponent $z$ at $T_c$:

```bash
python scripts/ising/measure_z.py --sizes 16 32 48 64 96 128 --n-seeds 10
```

## Development Guidance

VibeSpin maintains rigorous engineering and physical standards. All update algorithms must strictly satisfy **detailed balance** and **ergodicity** — whether via the Metropolis-Hastings acceptance rule (for single-spin updates) or the Fortuin-Kasteleyn bond construction (for Wolff cluster updates).

Performance-critical kernels are implemented with Numba JIT compilation to minimize execution time and memory allocation. These kernels utilize `@njit(cache=True, fastmath=True)` and avoid expensive modulo operations by using precomputed neighbor index arrays. To maintain reproducibility, models synchronize Numba's internal random number generator with the project seed.

### Verification Suite
Before proposing changes, ensure all verification checks pass using `uv run` (or standard environment commands):

```bash
uv run pytest
uv run ruff check .
uv run mypy --explicit-package-bases models/ utils/ scripts/
uv run pre-commit run --all-files --hook-stage pre-push
```

## Documentation

Full documentation is available at **[https://manganite.github.io/VibeSpin/](https://manganite.github.io/VibeSpin/)**.

To build the documentation locally:

```bash
cd docs
make html
```

### Sphinx Setup Notes

The documentation stack uses Sphinx with MyST Markdown and nbsphinx for notebook pages. A standard development install for documentation is:

```bash
pip install -e ".[docs]"
```

In this repository's devcontainer, documentation dependencies are installed automatically during container creation via the `.[dev,notebook,docs]` extras group.

Notebook rendering requires Pandoc. The docs configuration includes a fallback that uses the bundled binary from `pypandoc-binary` when a system `pandoc` executable is not available, so normal builds should work without manual Pandoc installation.

### Docs Troubleshooting

If `make html` fails after opening or rebuilding the container, verify the project environment was created successfully and reinstall extras:

```bash
pip install -e ".[dev,notebook,docs]"
cd docs
make html
```

If this still fails, check that the selected interpreter is the project virtual environment at `.venv/bin/python`.

For deeper insights, refer to the source guides:
- {doc}`Physics and Algorithm Guide <physics>`: Detailed explanation of physical models, observables, and algorithm prerequisites.
- {doc}`Architecture and Developer Guide <code>`: Technical blueprint, engineering rationale, and testing strategy.
- {doc}`Scripts Catalog <scripts>`: Comprehensive catalog of entry-point scripts.
- {doc}`Agent Instruction Guide <AGENTS>`: Mandatory technical constraints for AI Agents.
- {doc}`Performance Benchmarks <benchmarks>`: Detailed scaling analysis.
- {doc}`Ising Temperature Sweep <ising_temperature_sweep>`: Equilibrium thermodynamics of the 2D Ising model across the Onsager transition.
- {doc}`XY Temperature Sweep <xy_temperature_sweep>`: Thermodynamic sweep across the BKT crossover for the XY model.
- {doc}`Clock Temperature Sweep <clock_temperature_sweep>`: Temperature sweep for the q-state clock model showing the two-crossover regime structure.
- {doc}`BKT Transition <bkt_transition>`: Vortex density, helicity modulus, and correlation functions across the BKT transition.
- {doc}`Correlation and Coarsening <correlation_and_coarsening>`: Equilibrium correlation functions and non-equilibrium coarsening dynamics across all three models.
- {doc}`Ising Relaxation and Autocorrelation Analysis <ising_relaxation_and_autocorrelation_analysis>`: Comparative analysis of equilibration and autocorrelation across update schemes.
- {doc}`Dynamic Critical Exponents <dynamic_critical_exponents>`: Scaling analysis of autocorrelation times at criticality.
- {doc}`Wolff Efficiency <wolff_efficiency>`: Quantitative comparison of Metropolis and Wolff cluster algorithms near criticality.

## Project Context

VibeSpin was developed using AI-assisted scientific coding workflows. The framework demonstrates how high-level physical design and validation can be accelerated through iterative modeling, benchmarking, and automated testing.

For detailed procedural instructions, see {doc}`Agent Instruction Guide <AGENTS>`.

## Bibliography

[[1]](#bibliography) L. Onsager, "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition," *Physical Review*, vol. 65, no. 3-4, pp. 117–149, 1944. [APS Open Access](https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117)

[[2]](#bibliography) J. M. Kosterlitz and D. J. Thouless, "Ordering, metastability and phase transitions in two-dimensional systems," *Journal of Physics C: Solid State Physics*, vol. 6, no. 7, pp. 1181–1203, 1973. [IOP Open Access](https://iopscience.iop.org/article/10.1088/0022-3719/6/7/010)

[[3]](#bibliography) J. Villain, "Theory of one- and two-dimensional magnets with an easy magnetization plane. II. The planar, classical, two-dimensional magnet," *J. Phys. France* 36, 581-590 (1975). [Open Access](https://doi.org/10.1051/jphys:01975003606058100)

[[4]](#bibliography) W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications," *Biometrika*, vol. 57, no. 1, pp. 97–109, 1970. [Oxford Academic Open Access](https://academic.oup.com/biomet/article/57/1/97/252073)

[[5]](#bibliography) U. Wolff, "Collective Monte Carlo Updating for Spin Systems," *Physical Review Letters*, vol. 62, no. 4, pp. 361–364, 1989. [APS Open Access](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.62.361)
