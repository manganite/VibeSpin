# Changelog

All notable changes to VibeSpin are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version types align with [Conventional Commits](https://www.conventionalcommits.org/): `feat` → Added, `fix` → Fixed, `perf` → Performance, `refactor` → Changed, `docs` → Documentation, `phys` → Physics, `test` → Tests, `chore` → Chores.

**Policy:** Update the `[Unreleased]` section for every commit that changes user-facing behavior (physics, CLI, NPZ schema, public API, or documentation). When cutting a release, move `[Unreleased]` entries to a dated version block.

---

## [Unreleased]

### Added
- New script `scripts/clock/correlation_comparison.py`: computes equilibrium $G(r)$ for the $q = 6$ clock model in ordered, quasi-ordered, and disordered phases using convergence equilibration. Saves `results/clock/correlation_comparison.npz`.
- New script `scripts/ising/coarsening_analysis.py`: precomputes quench-depth sensitivity, equilibrium-to-coarsening crossover ($\xi_{\mathrm{eq}}$ measurement), and stochastic ensemble data for the Ising model. Saves `results/ising/coarsening_analysis.npz`.
- Multi-seed ensemble support in `utils/kinetics_helpers.run_ordering_kinetics`: new `n_seeds` and `base_seed` parameters produce median trajectories with IQR error bars. Per-seed arrays stored as additive NPZ keys (`R_sk_seeds`, `R_xi_seeds`, `third_metric_seeds`, `R_sk_err`, `R_xi_err`, `third_metric_err`).
- All three `ordering_kinetics.py` scripts (Ising, XY, Clock) now accept `--seeds` and `--base-seed` CLI arguments for multi-seed runs.
- Generated `results/xy/correlation_comparison.npz` ($L = 128$) from existing `scripts/xy/compare_correlations.py`.

### Changed
- `notebooks/Correlation_and_Coarsening.ipynb`: quench-depth (cell 31), crossover (cell 34), and stochastic ensemble (cell 39) cells now load precomputed data from `results/ising/coarsening_analysis.npz` with inline computation as fallback only. Clock correlation cell (cell 13) and XY correlation cell (cell 8) similarly load from cached NPZ files. Methodology section updated to describe multi-seed ensemble averaging.

### Added
- Part II of `notebooks/Correlation_and_Coarsening.ipynb` now renders the ordering-evolution snapshot figure inline via a live simulation cell (L=128, T=0.1), showing spin configurations, structure factor, and correlation length $\xi$ at $t = 1, 10, 100, 1000$ Monte Carlo sweeps. The figure is no longer loaded from a pre-generated PNG; it is generated reproducibly within the notebook kernel on each run.
- Part II of `notebooks/Correlation_and_Coarsening.ipynb` now includes a new "Vortex Spacing vs Structure Factor and Correlation Length" subsection for the XY and Clock models, showing $R_v = n_v^{-1/2}$ (mean vortex spacing) alongside $R_{S(k)}$ and $R_\xi$ on log-log axes with Allen-Cahn reference, mirroring the Ising MIL comparison. All three estimators show exponents $n \approx 0.4$–$0.5$.
- Part II of `notebooks/Correlation_and_Coarsening.ipynb` now includes a "Domain Coarsening vs Boundary-Wall Decay" subsection with a unified log-log comparison plot overlaying all three coarsening-length estimators ($R_{S(k)}$, $R_\xi$, $R_{\mathrm{MIL}}$) against the Allen-Cahn $t^{1/2}$ reference line.
- Kinetics loading cell now exposes `pre_third` (the MIL power-law prefactor) so the boundary-wall proxy fit line can be drawn in the comparison plot.

### Changed
- Improved Ising ordering kinetics generation defaults in `scripts/ising/ordering_kinetics.py`: `max_steps` reduced from 1000 to 400, `samples` increased from 10 to 25, `fit_min` reduced from 20 to 5. The regenerated `results/ising/ordering_kinetics.npz` yields fitted exponents $n \approx 0.490$ ($R_{S(k)}$), $0.487$ ($R_\xi$), $0.517$ ($R_{\mathrm{MIL}}$), close to the Allen-Cahn prediction $n = 1/2$.
- Improved XY ordering kinetics generation defaults in `scripts/xy/ordering_kinetics.py`: `temp` changed from 0.1 to 0.5 (well below $T_{\mathrm{BKT}} \approx 0.893$, better vortex mobility), `max_steps` increased to 1500, `samples` increased to 30, `fit_min` reduced to 5. Fitted exponents: $n \approx 0.448$ ($R_{S(k)}$), $0.477$ ($R_\xi$).
- Improved Clock ordering kinetics generation defaults in `scripts/clock/ordering_kinetics.py`: same parameter changes as XY. Fitted exponents: $n \approx 0.413$ ($R_{S(k)}$), $0.449$ ($R_\xi$).
- Regenerated `results/xy/ordering_kinetics.npz` and `results/clock/ordering_kinetics.npz` with new parameters (L=256, T=0.50). The exponent comparison table in `notebooks/Correlation_and_Coarsening.ipynb` now shows all three models with finite, physically meaningful values instead of NaN.
- Regenerated `results/ising/correlation_comparison.npz` with $L=128$ (64 radial points, $r \in [0, 63]$) and 10 000 measurement steps. The previous production cache had been overwritten by the test suite with $L=16$ and only 5 steps. The Part I Ising correlation plot now shows clear separation between the three temperature regimes over more than a decade in $r$.
- Fallback `_compute_ising_correlations` function in `notebooks/Correlation_and_Coarsening.ipynb` updated to use $L=128$ and 2 000 measurement steps (was $L=40$, 1 500 steps).
- $r^{-1/4}$ guide in the Ising $G(r)$ plot is now anchored dynamically to the critical data at $r = 2$ and spans the full data range; previously it was hardcoded to $[1, 15]$ regardless of $L$.
- Domain Growth Curves figure: y-axis relabelled from "Domain size $R(t)$" to "Characteristic length $R(t)$ (lattice units)" and suptitle changed from "Coarsening: domain growth after infinite-temperature quench" to "Coarsening kinetics after infinite-temperature quench". For the XY model, $R(t)$ tracks the vortex-antivortex spacing, not a domain size.
- Part II prose updated throughout to replace "domain size" and "domain-size proxies" with physically accurate language ("coarsening length scale", "coarsening-length estimators") that applies to both the Ising (domain-wall) and XY (vortex annihilation) regimes.

### Fixed
- `scripts/xy/compare_correlations.py`: replaced naive fixed-step `sim.equilibrate(n_steps=2000)` with `convergence_equilibrate_with_status` and ordered-start fallback. At $T = 0.4$ (deep BKT phase) the random-start simulation was stuck in a metastable state; the script now detects this and falls back to the ordered start. Regenerated `results/xy/correlation_comparison.npz`: $G(50) \approx 0.67$ (was 0.017), fitted $\eta \approx 0.07$ (was 0.57, theory: 0.064).
- XY and Clock correlation plots in `notebooks/Correlation_and_Coarsening.ipynb` now apply an $r \leq L/4$ cutoff to exclude periodic-boundary wrap-around artefacts, matching the established practice in the Clock plot.
- Part I notebook prose rewritten for all three models: XY and Clock sections now state the spin-wave formula $\eta(T) = T/(2\pi J)$ with explicit numerical predictions at the plotted temperatures ($\eta \approx 0.064$ at $T = 0.4$ for XY, $\eta \approx 0.13$ at $T = 0.8$ for Clock $q = 6$). Clock section specifies the exponent bounds $4/q^2 = 1/9$ at $T_1$ and $1/4$ at $T_2$. Removed stale references to the old two-panel XY layout and incorrect claim of $\eta \approx 0.65$ from broken data.

### Added
- New notebook `notebooks/XY_Temperature_Sweep.ipynb`: full thermodynamic sweep across the BKT crossover for the XY model, with data loading/fallback, convergence diagnostics, and autocorrelation analysis.
- New notebook `notebooks/Clock_Temperature_Sweep.ipynb`: temperature sweep for the q-state clock model showing the two-crossover regime structure, with entropy reference at ln(q).
- New notebook `notebooks/BKT_Transition.ipynb`: dedicated notebook examining vortex density, helicity modulus, and correlation function across the BKT transition.
- New notebook `notebooks/Correlation_and_Coarsening.ipynb`: combined notebook covering equilibrium correlation functions (Ising, XY) and non-equilibrium coarsening dynamics across all three models.
- NPZ output for `scripts/xy/bkt_transition.py`, `scripts/xy/helicity_modulus.py`, `scripts/xy/compare_correlations.py`, `scripts/ising/correlation_comparison.py`, `scripts/ising/correlation_divergence.py`.
- Optional `npz_path` parameter in `utils/kinetics_helpers.run_ordering_kinetics` for persisting kinetics data. All three ordering_kinetics scripts now save NPZ output.
- Data Pipeline mapping table in `SCRIPTS.md` documenting the script-to-NPZ-to-notebook data flow.

### Changed
- `utils/observables.py`: replaced upward dependency on `models.simulation_base.MonteCarloSimulation` with a structural `Protocol` class `_Sim`, eliminating the utils-to-models import cycle.

### Added
- `utils/kinetics_helpers.py`: new shared module providing the ordering-kinetics simulation loop for the Ising, XY, and Clock scripts. Exports `compute_mean_intercept_length` (stereological domain-size estimator for scalar-spin lattices) and `run_ordering_kinetics` (loop over logarithmically spaced step targets, power-law fitting, and figure generation).
- `utils/evolution_helpers.py`: new shared module providing the ordering-evolution snapshot loop for the Ising, XY, and Clock scripts. Exports `run_ordering_evolution`, which captures spin configurations, correlation functions, and (for vector-spin models) vorticity maps at each target step, then writes the multi-panel figure.
- `utils/sweep_helpers.py`: new shared module providing the two-layer temperature-sweep worker infrastructure used by all three model sweep scripts. Exports `ThermoPoint` (typed NamedTuple worker payload), `RawThermoData` (raw measurement arrays), `simulate_at_temperature` (Layer 1: physics and equilibration), `compute_thermo_observables` (Layer 2: statistical summarization), `simulate_thermo_point` (convenience wrapper), `build_uncertainty_bundle`, and `build_quality_flags`.

### Changed
- Refactored `utils/analysis.py` into `utils/statistics.py` (domain-agnostic statistical estimators and schemas) and `utils/observables.py` (physics observables and spatial tools).
- Refactored `scripts/ising/ordering_kinetics.py`, `scripts/xy/ordering_kinetics.py`, and `scripts/clock/ordering_kinetics.py` to delegate the full simulation loop, power-law fitting, and figure generation to `utils/kinetics_helpers.run_ordering_kinetics`. Each script is now a thin argument-parsing wrapper.
- Refactored `scripts/ising/ordering_evolution.py`, `scripts/xy/ordering_evolution.py`, and `scripts/clock/ordering_evolution.py` to delegate the snapshot loop and figure generation to `utils/evolution_helpers.run_ordering_evolution`. Vorticity capture is controlled by the `capture_vorticity` flag (True for XY and Clock, False for Ising).
- Refactored `scripts/ising/temperature_sweep.py`, `scripts/xy/temperature_sweep.py`, and `scripts/clock/temperature_sweep.py` to delegate all worker, uncertainty-bundle, and quality-flag logic to `utils/sweep_helpers`.
- Updated integration tests in `tests/integration/test_script_infrastructure.py` to import from `utils.sweep_helpers` rather than private script symbols. Added `TestSweepHelpersContract` class covering picklability, required return keys, and bundle schema shapes.

### Performance
- Fixed excessive memory allocation in `wolff_step_numba` across Ising, XY, and Clock models by pre-allocating the cluster mask and stack buffers once per simulation run, significantly lowering garbage collection overhead during large sweeps.

### Changed
- Refactored `utils/analysis.py` into `utils/statistics.py` (domain-agnostic statistical estimators and schemas) and `utils/observables.py` (physics observables and spatial tools).
- Removed deprecated `system_helpers.py` re-exports (`ensure_results_dir`, `adaptive_equilibrate`, etc.). All tests and Jupyter notebooks have been updated to explicitly import from `utils.equilibration` and `utils.plotting` directly.
- Renamed `run_scaling_benchmark()` to `main()` in `scripts/benchmarks/throughput.py` for consistency with other analysis scripts.

### Fixed
- Improved robustness of `blocking_error` and `summarize_primary_observable` in `utils/physics_helpers.py` when handling all-NaN time series.
- Fixed indentation in `tests/style/test_docstring_style.py` that caused test collection failures.

### Test
- Increased project test coverage from 63% to 75% by adding integration tests for the `main()` entry points of numerous analysis scripts (ordering kinetics, evolution, throughput, BKT transition, and helicity modulus).
- Expanded coverage of `utils/plotting.py` by exercising optional diagnostics and error-marking paths in `plot_temperature_sweep`.
- Strengthened `utils/physics_helpers.py` validation by adding tests for invalid parameter ranges and edge-case physical conditions.
- Enforced NumPy-style docstring sections and docstring presence for all public core modules in `tests/style/test_docstring_style.py`.

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
- Refactored `utils/analysis.py` into `utils/statistics.py` (domain-agnostic statistical estimators and schemas) and `utils/observables.py` (physics observables and spatial tools).
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

## [Unreleased]
### Refactored
- `utils/analysis.py` has been split into `utils/statistics.py` (domain-agnostic statistical estimators and NPZ schemas) and `utils/observables.py` (physics observables and spatial analysis) to separate concerns and improve maintainability without introducing subfolders.
