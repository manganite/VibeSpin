# Changelog

All notable changes to VibeSpin are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version types align with [Conventional Commits](https://www.conventionalcommits.org/): `feat` → Added, `fix` → Fixed, `perf` → Performance, `refactor` → Changed, `docs` → Documentation, `phys` → Physics, `test` → Tests, `chore` → Chores.

**Policy:** Update the `[Unreleased]` section for every commit that changes user-facing behavior (physics, CLI, NPZ schema, public API, or documentation). When cutting a release, move `[Unreleased]` entries to a dated version block.

---

## [Unreleased]

### Performance
- Fixed excessive memory allocation in `wolff_step_numba` across Ising, XY, and Clock models by pre-allocating the cluster mask and stack buffers once per simulation run, significantly lowering garbage collection overhead during large sweeps.

### Changed
- Removed deprecated `system_helpers.py` re-exports (`ensure_results_dir`, `adaptive_equilibrate`, etc.). All tests and Jupyter notebooks have been updated to explicitly import from `utils.equilibration` and `utils.plotting` directly.

### Chores
- Raise `requires-python` floor from `>=3.9` to `>=3.12` to match Ruff, mypy, and runtime targets.
- Add `scripts/` to pytest coverage tracking (`--cov=scripts`) and `[tool.coverage.run]` source list.
- Replace hard-coded `test_results/` fixture path in `tests/utility/test_system_helpers.py` with `tmp_path` / `tempfile.mkdtemp()` to eliminate leaked test artifacts.
- Remove stale `notebooks/results/` directory (leftover from an older notebook version; canonical path is `../results/`).
- Remove orphaned development result directories (`results/ising_n5/`, `results/ising_replacement_demo/`, `results/ising_replacement_stress/`) and unreferenced test images (`results/_smoke.png`, `results/_test.png`, `results/ising_simulation.png`).
- Remove `test_results/` from `.gitignore` (no longer written by any test).

---

## [0.1.0] — 2026-03-25

This is the baseline release capturing the full initial development history. The entries below summarize the major milestones grouped by category.

### Added
- **Models:** `IsingSimulation`, `XYSimulation`, `ClockSimulation`, and `DiscreteClockSimulation` with Numba-accelerated checkerboard, random-site, and Wolff cluster update schemes. Each model exposes a `main()` CLI entry point.
- **Equilibration:** Two-Start Convergence protocol with mutual cross-band criterion and Quasi-Steady Stuck Detection for low-temperature metastable trapping. Adaptive equilibration fallback for lightweight use cases.
- **Parallelization:** Granular point-level parallelization — every `(T, seed)` pair is an independent worker task, maximizing CPU utilization even for single-seed sweeps.
- **Uncertainty schema:** Standardized NPZ contract for all temperature-sweep outputs: `<obs>_value`, `<obs>_err`, `<obs>_ci_low`, `<obs>_ci_high`, `<obs>_tau_int`, `<obs>_n_eff`, `<obs>_samples`, plus entropy and `tau_int` asymmetric interval extras. Legacy keys preserved for backward compatibility.
- **Scripts — Ising:** `temperature_sweep.py`, `ordering_kinetics.py`, `ordering_evolution.py`, `correlation_divergence.py`, `correlation_comparison.py`, `wolff_efficiency.py`, `measure_z.py`, `diag_eq_traces.py`.
- **Scripts — XY:** `temperature_sweep.py`, `ordering_kinetics.py`, `ordering_evolution.py`, `bkt_transition.py`, `helicity_modulus.py`, `compare_correlations.py`.
- **Scripts — Clock:** `temperature_sweep.py`, `ordering_kinetics.py`, `ordering_evolution.py`, `compare_discrete_vs_continuous.py`.
- **Scripts — Benchmarks:** `throughput.py` for cross-model scaling and per-call analysis cost profiling.
- **Notebooks:** `Ising_Temperature_Sweep.ipynb`, `Ising_Relaxation_and_Autocorrelation_Analysis.ipynb`, `Dynamic_Critical_Exponents.ipynb`, `Wolff_Efficiency.ipynb`, `Performance_Benchmarks.ipynb`.
- **Documentation:** `README.md`, `PHYSICS.md`, `CODE.md`, `SCRIPTS.md`, `BIBLIOGRAPHY.md`, `AGENTS.md`, Sphinx HTML docs with nbsphinx notebook rendering and API autodoc.
- **CI:** GitHub Actions workflows for tests (`tests.yml`) and documentation (`docs.yml`). Pre-commit hooks for ruff, mypy, markdown link validation, API docs sync, Sphinx build, and docstring style.
- **Seed management:** Deterministic seeding for full reproducibility of all stochastic trajectories.
- **Transition presets, diagnostics overlays, and quality summary panels** for temperature-sweep plots.
- **Entropy uncertainty** via replicate aggregation, block-bootstrap, and single-seed propagation from specific-heat errors.
- **Hierarchical multi-seed uncertainty** combining within-seed blocking error with between-seed spread.

### Performance
- Numba JIT (`@njit(cache=True, fastmath=True)`) on all simulation kernels; `parallel=True` + `prange` on checkerboard kernels.
- Pre-calculated neighbor index arrays (`idx_next`, `idx_prev`) replace modulo for periodic boundary conditions.
- Discrete state representation for Clock models replaces per-site trigonometric evaluations with lookup tables (~2.5x speedup).
- Vectorized angle extraction for vorticity calculation; radial mask pre-computation for correlation functions.

### Changed
- `ordering_kinetics` scripts unified and refactored; renamed from `domain_growth`.
- `ordering_evolution` scripts unified from earlier `domain_snapshots`.
- Test suite reorganized from a flat layout into five layer directories: `algorithm/`, `model/`, `utility/`, `style/`, `integration/`.
- Equilibration and plotting logic extracted from `utils/system_helpers.py` into dedicated `utils/equilibration.py` and `utils/plotting.py` modules; re-exports maintained in `system_helpers.py` for backward compatibility.

### Fixed
- Diagnostics header/title overlap in temperature-sweep diagnostic plots.
- Restored notebook JSON after corruption; ensured stuck-detection prose and figures are consistent.
- Low-temperature sweep relaxation logic now correctly accepts stable ordered starts as converged.
- Default `--eq-max-steps` lowered from 200,000 to 20,000 to prevent excessively long equilibration runs.

[Unreleased]: https://github.com/manganite/VibeSpin/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/manganite/VibeSpin/releases/tag/v0.1.0
