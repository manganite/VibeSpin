# Developer and Architecture Guide

This guide describes the software architecture, design patterns, and engineering principles of the VibeSpin framework. It serves as a technical blueprint for developers and researchers extending the simulation engine or implementing new analysis workflows.

## Software Architecture

The VibeSpin engine follows a strict separation between high-level state management and performance-critical numerical kernels. The system architecture centers on a three-tier hierarchy that organizes code by its proximity to the simulation loop. At the foundation, Numba-accelerated kernels handle the microscopic spin updates and observable calculations. These kernels are stateless functions designed for maximum JIT compatibility. Above the kernels, specialized simulation classes manage the lattice state, handle parameter validation, and provide a stable API for experimentation. The top layer consists of analysis scripts and Jupyter notebooks that drive the simulations and aggregate results for physical interpretation.

The class hierarchy begins with the `MonteCarloSimulation` abstract base class located in `models/simulation_base.py`. This base class defines the interface for all models and implements shared infrastructure for periodic boundaries, radial distance binning, and Fourier-based spatial analysis. Specialized implementations like `IsingSimulation`, `XYSimulation`, and `ClockSimulation` inherit from this base and provide model-specific logic. Each simulation class encapsulates its own lattice data and configuration while delegating the inner loops to JIT-compiled functions. This design ensures that the high-level API remains expressive and easy to test without sacrificing the execution speed required for large-scale Monte Carlo sampling.

## Performance Engineering with Numba

High simulation throughput is achieved through the aggressive use of Numba JIT compilation. Every performance-critical loop in VibeSpin is decorated with `@njit(cache=True, fastmath=True)` to ensure that Python overhead is eliminated during the simulation. These kernels operate directly on NumPy arrays and avoid high-level Python abstractions that would trigger object-mode fallback. To maximize hardware utilization, checkerboard kernels utilize `parallel=True` and `prange` to distribute sublattice updates across multiple CPU cores. This parallelism is restricted to equilibrium regimes where the independent update of non-neighboring sites is physically valid and numerically efficient.

Numerical efficiency in the inner loops also relies on the elimination of expensive operations like the modulo operator for periodic boundary conditions. The simulation engine instead uses pre-calculated index arrays named `idx_next` and `idx_prev` to resolve neighbor positions. This approach replaces arithmetic divisions with fast memory lookups, which significantly reduces the computational cost per spin flip. When a simulation is initialized with a random seed, the engine synchronizes the Python random state with Numba's internal generator using the `_seed_numba` helper. This guarantee of determinism is essential for reproducing trajectories and debugging rare stochastic failures in complex energy landscapes.

## Parallelization and Equilibration Strategy

VibeSpin employs a granular, point-level parallelization strategy to maximize hardware utilization across all sweep configurations. Unlike legacy approaches that parallelize over full temperature sweeps (one process per seed replica), the current engine treats every $(T, \text{seed})$ pair as an independent task. This ensures that even single-seed sweeps utilize all available CPU cores by distributing individual temperature points across the worker pool. For multi-seed sweeps, this granular distribution provides superior load balancing, as slower points (e.g., those near a critical transition) are processed by workers as they become available, preventing faster processes from idling while waiting for a single slow replica to complete.

Equilibration is managed by the **Two-Start Convergence** protocol, which concurrently evolves a random-start and an ordered-start simulation for every point. The system identifies convergence when both smoothed magnetization traces satisfy a mutual cross-band criterion over a sustained dwell window. To handle the physical reality of metastable domain-wall trapping at low temperatures, the engine implements **Quasi-Steady Stuck Detection**. If the random-start trace becomes stranded in a plateau while the ordered-start trace remains stable and cleanly ordered, the system declares convergence and falls back to the ordered-start simulation for measurements. This prevents unnecessary MC steps and early-exit failures in regimes where initialization bias is physically expected but the target equilibrium state is known.

## Error Handling and Type Safety

The framework maintains a rigorous approach to type safety and error reporting to support scientific reliability. Every source file employs `from __future__ import annotations` and uses comprehensive type hints to enable static analysis with Mypy. Public methods in the simulation classes use keyword-only arguments to prevent parameter ambiguity, which is particularly useful when configuring models with many physical constants. This strict interface design reduces the likelihood of silent failures or misconfiguration during automated parameter sweeps.

Errors are managed through a three-tier exception hierarchy defined in `utils/exceptions.py`. General configuration issues or invalid user inputs trigger standard `ValueError` exceptions. Physical or mathematical failures during data processing, such as an undefined autocorrelation time due to zero variance, raise specialized `NumericalAnalysisError` subclasses. Internal logic violations or impossible states result in a `RuntimeError`. This granular approach to error handling allows analysis scripts to catch and handle known failure modes gracefully while ensuring that fundamental bugs are immediately visible to the developer.

## Testing Strategy and Quality Gates

The VibeSpin verification suite is organized into five conceptual layers to ensure both code quality and physical correctness. Microscopic integrity tests in the algorithm layer validate the fundamental properties of the Monte Carlo kernels, including detailed balance and ergodicity [[1]](#Bibliography). The model layer focuses on API contracts, edge cases at extreme temperatures, and CLI behavior. Utility tests cover physics observables and system helpers. A dedicated style layer enforces documentation standards and docstring compliance. Finally, the integration layer uses reusable infrastructure patterns to verify script behavior, output schemas, and deterministic reproducibility.

Development workflows are protected by multi-stage quality gates implemented via pre-commit and pre-push hooks. These hooks run the full suite of static analysis tools, including Ruff for linting and Mypy for type checking. Documentation consistency is also enforced at push time: the system validates markdown links, ensures that API documentation is synchronized with the source, and builds the Sphinx HTML output with all warnings treated as errors. This rigorous pipeline ensures that every contribution maintains the high engineering standards required for reproducible scientific computing.

## Uncertainty Data Contract and Pipeline

Analysis scripts and notebooks in VibeSpin share a standardized uncertainty schema for all serialized observables. This contract is defined in `utils/statistics.py` and enforced by the integration test layer; any script that writes an NPZ file should conform to it.

### Schema Fields

For each observable `<obs>` saved by a temperature-sweep script, the NPZ file contains the following keys. The suffix `<obs>` takes values such as `avg_m`, `avg_e`, `susc`, and `spec_h`.

| Key | Shape | Meaning |
|-----|-------|---------|
| `<obs>_value` | `(T,)` | Point estimate (mean across seeds, or single-seed value) |
| `<obs>_err` | `(T,)` | Total standard error (single-seed blocking error for `n_seeds=1`; hierarchical within+between-seed error for `n_seeds>1`) |
| `<obs>_ci_low` | `(T,)` | Lower confidence-interval bound |
| `<obs>_ci_high` | `(T,)` | Upper confidence-interval bound |
| `<obs>_tau_int` | `(T,)` | Integrated autocorrelation time per temperature point |
| `<obs>_n_eff` | `(T,)` | Effective sample size; `NaN` where `tau_int` is undefined |
| `<obs>_samples` | `(T, S)` | Raw per-seed samples (S = 1 for single-seed runs) |

Global metadata keys are written once per file: `uncertainty_method` (string, currently `'blocking'`), `confidence_level` (float, default `0.68`), `n_seeds` (int), `bootstrap_resamples` (int, 0 if unused), and `nan_or_undefined_count` (float, count of temperature points with undefined `tau_int`).

Temperature sweeps also serialize additive uncertainty fields for derived extras. Entropy now includes `entropy_value`, `entropy_err`, `entropy_ci_low`, `entropy_ci_high`, and `entropy_samples`, with method metadata in `entropy_uncertainty_method`. For single-seed sweeps, entropy uncertainty falls back to propagation from the specific-heat uncertainty instead of remaining undefined. Critical-slowing diagnostics include `tau_int_err`, `tau_int_ci_low`, and `tau_int_ci_high`, where interval bounds are percentile-based and can be asymmetric. Per-temperature quality diagnostics are written as `undefined_autocorr_flag`, `low_effective_sample_flag`, and `tau_interval_unstable_flag`.

Legacy keys (`avg_m`, `avg_e`, `susc`, `spec_h`, `entropy`, `tau_int`, `temperatures`) are preserved alongside the new keys for backward compatibility. Consumers should prefer the standardized keys where available.

### Utility API

The canonical implementation lives in `utils/statistics.py`. The key public functions are as follows. `estimate_effective_sample_size` computes $N_\mathrm{eff} = N / (2\tau_{\mathrm{int}})$ from a time series, optionally accepting a pre-computed `tau_int` to avoid redundant autocorrelation calculation. `blocking_error` applies the plateau-selection blocking method and returns the plateau standard error alongside the estimated `tau_int` and `n_eff`. `summarize_primary_observable` wraps blocking into a single dict conforming to the schema fields above, handling the zero-variance edge case by storing `NaN` for the undefined fields. `summarize_derived_observable` extends this to nonlinear quadratic estimators like susceptibility and specific heat, supporting both the blocking propagation (default) and an optional block-bootstrap. `summarize_seed_ensemble` combines per-seed point estimates and within-seed errors into a hierarchical total uncertainty for multi-seed sweeps. `summarize_entropy_observable` propagates specific-heat uncertainty to entropy via replicate aggregation, bootstrap over replicate curves, or single-seed propagation from specific-heat errors. `summarize_asymmetric_replicate_uncertainty` provides percentile-based asymmetric intervals for skewed replicate distributions such as $\tau_{\mathrm{int}}$.

### Design Rationale

The schema is written additively: new standardized keys are always appended rather than replacing the legacy layout. This allows existing notebooks and downstream loaders to continue reading the pre-schema keys while new code targets the standardized ones. All constants (`DEFAULT_CONFIDENCE_LEVEL`, `UNCERTAINTY_METHOD_BLOCKING`, `UNCERTAINTY_METHOD_BOOTSTRAP`, `UNCERTAINTY_FIELDS`, `UNCERTAINTY_METADATA_FIELDS`) are defined at module level in `utils/statistics.py` and imported by all scripts, ensuring a single authoritative source for the contract.

### Sweep Worker Infrastructure

The three temperature-sweep scripts (Ising, XY, Clock) share a two-layer worker architecture defined in `utils/sweep_helpers.py`. This module eliminates the ~300 lines of previously duplicated per-model worker code and centralizes the equilibration, measurement, and statistical summarization logic.

The input to the worker is a `ThermoPoint` named tuple. It carries all configuration needed for one `(temperature, seed)` point: lattice size, measurement steps, equilibration parameters, model dispatch information (`model_cls` and `model_kwargs`), and statistical config (`confidence`, `derived_method`, `bootstrap_resamples`). Embedding statistical config as fields rather than module-level globals makes the payload fully self-contained and safe for multiprocessing dispatch, since workers receive all their configuration through the pickled payload rather than relying on shared global state.

The worker logic is split into two explicit layers. Layer 1, `simulate_at_temperature`, owns the physics: it instantiates the simulation, runs the two-start convergence equilibration, executes the measurement sweep, and returns a `RawThermoData` named tuple containing the raw magnetization and energy arrays alongside equilibration diagnostics. Layer 2, `compute_thermo_observables`, owns the statistics: it calls the appropriate `utils/statistics.py` summarizers on the raw arrays and returns a flat `dict[str, float]` in the standard schema format. This separation means that the measurement and summarization concerns can be tested, profiled, and reasoned about independently. The `simulate_thermo_point` function chains both layers and is the function passed to `parallel_sweep` in all three sweep scripts.

Two post-processing helpers, `build_uncertainty_bundle` and `build_quality_flags`, aggregate per-seed per-temperature scalar results into the full `(T,)` and `(T, S)` arrays required by the NPZ schema. These were previously duplicated identically in all three scripts and are now the shared canonical implementations.

### Ordering Kinetics and Evolution Helper Infrastructure

The three ordering-kinetics scripts (Ising, XY, Clock) and the three ordering-evolution scripts share simulation loops that were previously duplicated across each model. These loops are now centralized in two utility modules.

`utils/kinetics_helpers.py` exports `compute_mean_intercept_length` and `run_ordering_kinetics`. `compute_mean_intercept_length` estimates the domain size of a scalar-spin lattice via the stereological mean intercept length: it counts domain-wall crossings along every row and column (including the periodic wrap-around) and divides the total line length by the crossing count. `run_ordering_kinetics` owns the full kinetics loop: it builds logarithmically spaced step targets, instantiates the model, steps the simulation forward, calls `compute_kinetics_metrics` and a caller-supplied `third_metric_fn` at each target, fits power laws via `power_fit`, and writes the two-panel figure with `plot_ordering_kinetics`. The `third_metric_fn` parameter is a `Callable[[Any], float]` that allows each script to supply a model-specific scalar metric (mean intercept length for Ising, vortex density for XY and Clock) without any branching inside the helper. `model_kwargs` is forwarded to the constructor, enabling the Clock model to pass `q` and `A` as keyword arguments.

`utils/evolution_helpers.py` exports `run_ordering_evolution`. It accepts a `capture_vorticity` flag that governs whether vorticity maps are collected and forwarded to `plot_ordering_evolution`. When `True` (XY and Clock), the function captures `_calculate_vorticity()` at each snapshot and logs the instantaneous vortex density; when `False` (Ising), those calls are skipped and `vorticity_data=None` is passed to the plotter, which falls back to a structure-factor panel. The `step_targets` list is sorted internally so callers need not pre-sort it.

The three kinetics scripts and three evolution scripts are now thin wrappers: each constructs its argument parser, configures logging, and calls the corresponding helper with model-specific constants and labels.

## Bibliography

The engineering and algorithmic choices in VibeSpin are grounded in standard practices for scientific computing and statistical physics. For a comprehensive list of all references used in the project, see [BIBLIOGRAPHY.md](./BIBLIOGRAPHY.md).

[[1]](#Bibliography) W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications," *Biometrika*, vol. 57, no. 1, pp. 97–109, 1970. [Oxford Academic Open Access](https://academic.oup.com/biomet/article/57/1/97/252073)

[[2]](#Bibliography) M. E. J. Newman and G. T. Barkema, "Monte Carlo Methods in Statistical Physics," Oxford University Press, 1999. [Lecture Notes Summary (H. G. Katzgraber)](https://arxiv.org/abs/0905.1629)

[[3]](#Bibliography) Numba Documentation: [Parallel Acceleration](https://numba.readthedocs.io/en/stable/user/parallel.html)
