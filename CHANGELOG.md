# Changelog

All notable changes to VibeSpin are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version types align with [Conventional Commits](https://www.conventionalcommits.org/): `feat` → Added, `fix` → Fixed, `perf` → Performance, `refactor` → Changed, `docs` → Documentation, `phys` → Physics, `test` → Tests, `chore` → Chores.

**Policy:** Update the `[Unreleased]` section for every commit that changes user-facing behavior (physics, CLI, NPZ schema, public API, or documentation). When cutting a release, move `[Unreleased]` entries to a dated version block.

---

## [Unreleased]

### Added
- All three temperature sweeps now write the `n_seeds` and `bootstrap_resamples` NPZ metadata keys required by the uncertainty schema contract; the Clock sweep additionally writes `entropy_uncertainty_method` (previously Ising/XY only). Existing keys are preserved (additive change).
- New shared `validate_sweep_uncertainty_args` helper in `utils/sweep_helpers.py`: the XY and Clock sweeps now reject invalid `--confidence-level`, `--n-seeds`, strict-mode thresholds, and bootstrap configurations up front, exactly like the Ising sweep (previously they failed deep inside worker statistics or silently produced empty sweeps).
- `summarize_replicate_samples` in `utils/statistics.py` now returns an `err` field (half the percentile band width) so replicate summaries conform to the standard uncertainty schema. This fixes a `KeyError` crash in `scripts/ising/measure_z.py` that occurred at NPZ save time, after the full sweep had completed.
- New `UNCERTAINTY_METHOD_REPLICATE` constant (`'replicate_percentile'`). `scripts/ising/measure_z.py` now labels its NPZ `uncertainty_method` metadata truthfully instead of claiming `'blocking'`.
- `ClockSimulation` emits a `UserWarning` when `update='wolff'` is combined with a non-zero anisotropy `A`: the Wolff-Evertz reflection ignores the anisotropy term, so detailed balance is exact only for `A=0`.
- New script `scripts/clock/correlation_comparison.py`: computes equilibrium $G(r)$ for the $q = 6$ clock model in ordered, quasi-ordered, and disordered phases using convergence equilibration. Saves `results/clock/correlation_comparison.npz`.
- New script `scripts/ising/coarsening_analysis.py`: precomputes quench-depth sensitivity, equilibrium-to-coarsening crossover ($\xi_{\mathrm{eq}}$ measurement), and stochastic ensemble data for the Ising model. Saves `results/ising/coarsening_analysis.npz`.
- Multi-seed ensemble support in `utils/kinetics_helpers.run_ordering_kinetics`: new `n_seeds` and `base_seed` parameters produce median trajectories with IQR error bars. Per-seed arrays stored as additive NPZ keys (`R_sk_seeds`, `R_xi_seeds`, `third_metric_seeds`, `R_sk_err`, `R_xi_err`, `third_metric_err`).
- All three `ordering_kinetics.py` scripts (Ising, XY, Clock) now accept `--seeds` and `--base-seed` CLI arguments for multi-seed runs.
- Generated `results/xy/correlation_comparison.npz` ($L = 128$) from existing `scripts/xy/compare_correlations.py`.
- Part II of `notebooks/Correlation_and_Coarsening.ipynb` now renders the ordering-evolution snapshot figure inline via a live simulation cell (L=128, T=0.1), showing spin configurations, structure factor, and correlation length $\xi$ at $t = 1, 10, 100, 1000$ Monte Carlo sweeps. The figure is no longer loaded from a pre-generated PNG; it is generated reproducibly within the notebook kernel on each run.
- Part II of `notebooks/Correlation_and_Coarsening.ipynb` now includes a new "Vortex Spacing vs Structure Factor and Correlation Length" subsection for the XY and Clock models, showing $R_v = n_v^{-1/2}$ (mean vortex spacing) alongside $R_{S(k)}$ and $R_\xi$ on log-log axes with Allen-Cahn reference, mirroring the Ising MIL comparison. All three estimators show exponents $n \approx 0.4$–$0.5$.
- Part II of `notebooks/Correlation_and_Coarsening.ipynb` now includes a "Domain Coarsening vs Boundary-Wall Decay" subsection with a unified log-log comparison plot overlaying all three coarsening-length estimators ($R_{S(k)}$, $R_\xi$, $R_{\mathrm{MIL}}$) against the Allen-Cahn $t^{1/2}$ reference line.
- Kinetics loading cell now exposes `pre_third` (the MIL power-law prefactor) so the boundary-wall proxy fit line can be drawn in the comparison plot.
- New notebook `notebooks/XY_Temperature_Sweep.ipynb`: full thermodynamic sweep across the BKT crossover for the XY model, with data loading/fallback, convergence diagnostics, and autocorrelation analysis.
- New notebook `notebooks/Clock_Temperature_Sweep.ipynb`: temperature sweep for the q-state clock model showing the two-crossover regime structure, with entropy reference at ln(q).
- New notebook `notebooks/BKT_Transition.ipynb`: dedicated notebook examining vortex density, helicity modulus, and correlation function across the BKT transition.
- New notebook `notebooks/Correlation_and_Coarsening.ipynb`: combined notebook covering equilibrium correlation functions (Ising, XY) and non-equilibrium coarsening dynamics across all three models.
- NPZ output for `scripts/xy/bkt_transition.py`, `scripts/xy/helicity_modulus.py`, `scripts/xy/compare_correlations.py`, `scripts/ising/correlation_comparison.py`, `scripts/ising/correlation_divergence.py`.
- Optional `npz_path` parameter in `utils/kinetics_helpers.run_ordering_kinetics` for persisting kinetics data. All three ordering_kinetics scripts now save NPZ output.
- Data Pipeline mapping table in `SCRIPTS.md` documenting the script-to-NPZ-to-notebook data flow.
- `utils/kinetics_helpers.py`: new shared module providing the ordering-kinetics simulation loop for the Ising, XY, and Clock scripts. Exports `compute_mean_intercept_length` (stereological domain-size estimator for scalar-spin lattices) and `run_ordering_kinetics` (loop over logarithmically spaced step targets, power-law fitting, and figure generation).
- `utils/evolution_helpers.py`: new shared module providing the ordering-evolution snapshot loop for the Ising, XY, and Clock scripts. Exports `run_ordering_evolution`, which captures spin configurations, correlation functions, and (for vector-spin models) vorticity maps at each target step, then writes the multi-panel figure.
- `utils/sweep_helpers.py`: new shared module providing the two-layer temperature-sweep worker infrastructure used by all three model sweep scripts. Exports `ThermoPoint` (typed NamedTuple worker payload), `RawThermoData` (raw measurement arrays), `simulate_at_temperature` (Layer 1: physics and equilibration), `compute_thermo_observables` (Layer 2: statistical summarization), `simulate_thermo_point` (convenience wrapper), `build_uncertainty_bundle`, and `build_quality_flags`.

### Changed
- Sweep workers compute blocking analysis and autocorrelation once per time series and share the results between the primary and derived summarizers (previously two blocking scans and two FFTs per series per point). `summarize_primary_observable` and `summarize_derived_observable` accept optional precomputed `blocking`/`tau_int` inputs; new `estimate_tau_int_or_nan` helper. `summarize_derived_observable` also gains an `rng_seed` parameter for its block-bootstrap (default 0 preserves the previous stream).
- `convergence_equilibrate_with_status` spaces its convergence checks geometrically (25% growth) once the accumulated trace outgrows a chunk, bounding total analysis cost at O(max_steps) instead of O(max_steps^2/chunk). Convergence occurring between two checks is caught at the next check, so measured equilibration step counts can be slightly larger than before, never smaller.
- `save_plot` accepts an explicit `fig` and a `close` flag; the temperature-sweep, kinetics, and evolution plot helpers now pass their figure explicitly and close it after saving, so long sweeps no longer accumulate open figures in pyplot's global registry.
- Public model observable API: `get_magnetization`, `get_energy`, and `calculate_correlation_function` on `MonteCarloSimulation`, plus `calculate_vorticity`, `get_vortex_density`, and `get_helicity_data` via a new `VectorSpinObservablesMixin` on the XY and Clock models. Utility protocols, helper modules, scripts, and notebooks now use these public methods instead of the private underscore methods.
- Keyword-only arguments on public analysis entry points: `estimate_relaxation_time_two_start`, `convergence_equilibrate`, `convergence_equilibrate_with_status`, `parse_args_compat`, and `compute_mean_intercept_length`; all call sites (scripts, tests, notebooks) updated.
- `blocking_error` now treats any NaN-containing series like the all-NaN case and returns the graceful NaN result dict (previously a partially NaN series silently produced a meaningless plateau selection).
- `run_ordering_evolution` raises `RuntimeError` instead of silently skipping a snapshot when the lattice is uninitialized; `run_ordering_kinetics` validates `max_steps`, `samples`, `fit_min`, and `n_seeds` before computing step targets.
- New shared `correlation_length_1e` helper in `utils/observables.py` replaces three divergent inline copies of the 1/e-crossing interpolation (kinetics metrics and two plot helpers); removed the dead, uncalled `plot_ordering_evolution_snapshots` from `utils/plotting.py`.
- `docs/check_markdown_links.py` now validates intra-document anchors (`#fragment`) against heading slugs in both same-file and cross-file links, and gains an opt-in `--external` mode that probes external URLs over the network. The Sphinx configuration no longer suppresses `myst.xref_missing`, so dead internal cross-references now fail the `-W` docs build.
- `docs/generate_api_docs.py` includes `scripts.benchmarks` in the generated API reference (the throughput benchmark was previously missing from the docs site).
- **Breaking (seeded trajectories):** per-sweep Numba reseeding now mixes `(seed, step)` through a SplitMix64 finalizer (`models.simulation_base._derive_step_seed`) instead of using `seed + step`. Previously, seed `s` at sweep `t + 1` consumed exactly the same random stream as seed `s + 1` at sweep `t`, so replicas seeded with consecutive integers (the `base_seed + k` convention) were not statistically independent. Seeded trajectories differ from those produced by earlier versions; unseeded behavior is unchanged.
- Clock continuous Wolff kernel (`clock_wolff_step_numba`) now uses the pre-allocated cluster buffers and reflects only tracked cluster members instead of allocating fresh arrays per call and scanning the full lattice. It returns the cluster size, and `last_cluster_size` is now initialized on all models in `MonteCarloSimulation` (previously Ising-only).
- `build_uncertainty_bundle` single-seed path now applies the Gaussian z-multiplier for the requested confidence level to `ci_low`/`ci_high` (previously fixed at plus/minus one standard error regardless of `confidence`).
- `README.md` documents that the optional parallel checkerboard kernels are not seed-reproducible (Numba seeds only the calling thread's RNG); model docstrings state the same caveat on the `parallel` parameter.
- Updated pre-commit hook versions to match the locked project environment: `ruff-pre-commit` v0.15.0 → v0.15.13 and `mirrors-mypy` v1.18.2 → v2.1.0. The mypy hook's `additional_dependencies` are now pinned to the versions from `requirements.txt`, fixing spurious `prange` iteration errors caused by version drift between the isolated hook environment and the project venv.
- Transitioned project package and virtual environment management to use `uv`.
- Configured `.devcontainer/devcontainer.json` postCreateCommand to globally install `uv` and use `uv sync --all-extras` for deterministic environment provisioning.
- Pinned and locked all direct and transitive dependencies into `uv.lock`.
- Regenerated standard `requirements.txt` using `uv pip compile` to ensure backward compatibility.
- Updated `README.md` to recommend `uv sync` and `uv run` commands for local installation and test suites.
- `notebooks/Correlation_and_Coarsening.ipynb`: quench-depth (cell 31), crossover (cell 34), and stochastic ensemble (cell 39) cells now load precomputed data from `results/ising/coarsening_analysis.npz` with inline computation as fallback only. Clock correlation cell (cell 13) and XY correlation cell (cell 8) similarly load from cached NPZ files. Methodology section updated to describe multi-seed ensemble averaging.
- Improved Ising ordering kinetics generation defaults in `scripts/ising/ordering_kinetics.py`: `max_steps` reduced from 1000 to 400, `samples` increased from 10 to 25, `fit_min` reduced from 20 to 5. The regenerated `results/ising/ordering_kinetics.npz` yields fitted exponents $n \approx 0.490$ ($R_{S(k)}$), $0.487$ ($R_\xi$), $0.517$ ($R_{\mathrm{MIL}}$), close to the Allen-Cahn prediction $n = 1/2$.
- Improved XY ordering kinetics generation defaults in `scripts/xy/ordering_kinetics.py`: `temp` changed from 0.1 to 0.5 (well below $T_{\mathrm{BKT}} \approx 0.893$, better vortex mobility), `max_steps` increased to 1500, `samples` increased to 30, `fit_min` reduced to 5. Fitted exponents: $n \approx 0.448$ ($R_{S(k)}$), $0.477$ ($R_\xi$).
- Improved Clock ordering kinetics generation defaults in `scripts/clock/ordering_kinetics.py`: same parameter changes as XY. Fitted exponents: $n \approx 0.413$ ($R_{S(k)}$), $0.449$ ($R_\xi$).
- Regenerated `results/xy/ordering_kinetics.npz` and `results/clock/ordering_kinetics.npz` with new parameters (L=256, T=0.50). The exponent comparison table in `notebooks/Correlation_and_Coarsening.ipynb` now shows all three models with finite, physically meaningful values instead of NaN.
- Regenerated `results/ising/correlation_comparison.npz` with $L=128$ (64 radial points, $r \in [0, 63]$) and 10 000 measurement steps. The previous production cache had been overwritten by the test suite with $L=16$ and only 5 steps. The Part I Ising correlation plot now shows clear separation between the three temperature regimes over more than a decade in $r$.
- Fallback `_compute_ising_correlations` function in `notebooks/Correlation_and_Coarsening.ipynb` updated to use $L=128$ and 2 000 measurement steps (was $L=40$, 1 500 steps).
- $r^{-1/4}$ guide in the Ising $G(r)$ plot is now anchored dynamically to the critical data at $r = 2$ and spans the full data range; previously it was hardcoded to $[1, 15]$ regardless of $L$.
- Domain Growth Curves figure: y-axis relabelled from "Domain size $R(t)$" to "Characteristic length $R(t)$ (lattice units)" and suptitle changed from "Coarsening: domain growth after infinite-temperature quench" to "Coarsening kinetics after infinite-temperature quench". For the XY model, $R(t)$ tracks the vortex-antivortex spacing, not a domain size.
- Part II prose updated throughout to replace "domain size" and "domain-size proxies" with physically accurate language ("coarsening length scale", "coarsening-length estimators") that applies to both the Ising (domain-wall) and XY (vortex annihilation) regimes.
- `utils/observables.py`: replaced upward dependency on `models.simulation_base.MonteCarloSimulation` with a structural `Protocol` class `_Sim`, eliminating the utils-to-models import cycle.
- Refactored `scripts/ising/ordering_kinetics.py`, `scripts/xy/ordering_kinetics.py`, and `scripts/clock/ordering_kinetics.py` to delegate the full simulation loop, power-law fitting, and figure generation to `utils/kinetics_helpers.run_ordering_kinetics`. Each script is now a thin argument-parsing wrapper.
- Refactored `scripts/ising/ordering_evolution.py`, `scripts/xy/ordering_evolution.py`, and `scripts/clock/ordering_evolution.py` to delegate the snapshot loop and figure generation to `utils/evolution_helpers.run_ordering_evolution`. Vorticity capture is controlled by the `capture_vorticity` flag (True for XY and Clock, False for Ising).
- Refactored `scripts/ising/temperature_sweep.py`, `scripts/xy/temperature_sweep.py`, and `scripts/clock/temperature_sweep.py` to delegate all worker, uncertainty-bundle, and quality-flag logic to `utils/sweep_helpers`.
- Updated integration tests in `tests/integration/test_script_infrastructure.py` to import from `utils.sweep_helpers` rather than private script symbols. Added `TestSweepHelpersContract` class covering picklability, required return keys, and bundle schema shapes.
- Removed deprecated `system_helpers.py` re-exports (`ensure_results_dir`, `adaptive_equilibrate`, etc.). All tests and Jupyter notebooks have been updated to explicitly import from `utils.equilibration` and `utils.plotting` directly.
- Renamed `run_scaling_benchmark()` to `main()` in `scripts/benchmarks/throughput.py` for consistency with other analysis scripts.

### Fixed
- `BIBLIOGRAPHY.md` now includes the Berezinskii (1972) and Nelson & Kosterlitz (1977) references cited in `notebooks/BKT_Transition.ipynb`, per the Bibliography Inclusion policy.
- `run_ordering_kinetics` no longer converts a legitimate fitted exponent or prefactor of exactly 0.0 into NaN in its NPZ output (falsy-zero `or` bug); a zero exponent is also logged again. `power_fit` now excludes non-positive `t` values before the log-log fit instead of silently corrupting it.
- Documentation integrity: removed the verbatim duplicated 'Sweep Worker Infrastructure' section in `CODE.md`; consolidated the fragmented `[Unreleased]` changelog section (duplicate category headings, repeated entries, stray second `[Unreleased]` block); renamed the reference-policy stub heading in `README.md` and lowercased citation anchors so `[[N]]` links resolve to the bibliography on GitHub and in Sphinx; corrected stale module references to the removed `utils/physics_helpers.py` and `utils/system_helpers.py` in `PHYSICS.md`, `models/clock_model.py`, `utils/plotting.py`, and test docstrings; updated the utility-test-layer listing in `AGENTS.md`; removed `SCRIPTS.md` claims about nonexistent `--update` CLI flags, unimplemented clock entropy normalization, and unwritten `tau_int_ci_*` NPZ keys.
- `scripts/xy/compare_correlations.py`: replaced naive fixed-step `sim.equilibrate(n_steps=2000)` with `convergence_equilibrate_with_status` and ordered-start fallback. At $T = 0.4$ (deep BKT phase) the random-start simulation was stuck in a metastable state; the script now detects this and falls back to the ordered start. Regenerated `results/xy/correlation_comparison.npz`: $G(50) \approx 0.67$ (was 0.017), fitted $\eta \approx 0.07$ (was 0.57, theory: 0.064).
- XY and Clock correlation plots in `notebooks/Correlation_and_Coarsening.ipynb` now apply an $r \leq L/4$ cutoff to exclude periodic-boundary wrap-around artefacts, matching the established practice in the Clock plot.
- Part I notebook prose rewritten for all three models: XY and Clock sections now state the spin-wave formula $\eta(T) = T/(2\pi J)$ with explicit numerical predictions at the plotted temperatures ($\eta \approx 0.064$ at $T = 0.4$ for XY, $\eta \approx 0.13$ at $T = 0.8$ for Clock $q = 6$). Clock section specifies the exponent bounds $4/q^2 = 1/9$ at $T_1$ and $1/4$ at $T_2$. Removed stale references to the old two-panel XY layout and incorrect claim of $\eta \approx 0.65$ from broken data.
- Improved robustness of `blocking_error` and `summarize_primary_observable` in `utils/statistics.py` when handling all-NaN time series.
- Fixed indentation in `tests/style/test_docstring_style.py` that caused test collection failures.

### Performance
- Fixed excessive memory allocation in `wolff_step_numba` across Ising, XY, and Clock models by pre-allocating the cluster mask and stack buffers once per simulation run, significantly lowering garbage collection overhead during large sweeps.

### Test
- Replaced two placeholder algorithm-integrity tests with empirical kernel tests: `test_ising_detailed_balance` now verifies the random-site Metropolis kernel against the exactly enumerated Boltzmann distribution of a 2x2 lattice (previously it compared the Metropolis formula against itself without calling the kernel), and `test_xy_proposal_symmetry` now measures the actual proposal distribution at J=0 (previously an empty `pass` placeholder).
- Increased project test coverage from 63% to 75% by adding integration tests for the `main()` entry points of numerous analysis scripts (ordering kinetics, evolution, throughput, BKT transition, and helicity modulus).
- Expanded coverage of `utils/plotting.py` by exercising optional diagnostics and error-marking paths in `plot_temperature_sweep`.
- Strengthened `utils/statistics.py` validation by adding tests for invalid parameter ranges and edge-case physical conditions.
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
