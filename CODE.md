# Developer and Architecture Guide

This guide describes the software architecture, design patterns, and engineering principles of the VibeSpin framework. It serves as a technical blueprint for developers and researchers extending the simulation engine or implementing new analysis workflows.

## Software Architecture

The VibeSpin engine follows a strict separation between high-level state management and performance-critical numerical kernels. The system architecture centers on a three-tier hierarchy that organizes code by its proximity to the simulation loop. At the foundation, Numba-accelerated kernels handle the microscopic spin updates and observable calculations. These kernels are stateless functions designed for maximum JIT compatibility. Above the kernels, specialized simulation classes manage the lattice state, handle parameter validation, and provide a stable API for experimentation. The top layer consists of analysis scripts and Jupyter notebooks that drive the simulations and aggregate results for physical interpretation.

The class hierarchy begins with the `MonteCarloSimulation` abstract base class located in `models/simulation_base.py`. This base class defines the interface for all models and implements shared infrastructure for periodic boundaries, radial distance binning, and Fourier-based spatial analysis. Specialized implementations like `IsingSimulation`, `XYSimulation`, and `ClockSimulation` inherit from this base and provide model-specific logic. Each simulation class encapsulates its own lattice data and configuration while delegating the inner loops to JIT-compiled functions. This design ensures that the high-level API remains expressive and easy to test without sacrificing the execution speed required for large-scale Monte Carlo sampling.

## Performance Engineering with Numba

High simulation throughput is achieved through the aggressive use of Numba JIT compilation. Every performance-critical loop in VibeSpin is decorated with `@njit(cache=True, fastmath=True)` to ensure that Python overhead is eliminated during the simulation. These kernels operate directly on NumPy arrays and avoid high-level Python abstractions that would trigger object-mode fallback. To maximize hardware utilization, checkerboard kernels utilize `parallel=True` and `prange` to distribute sublattice updates across multiple CPU cores. This parallelism is restricted to equilibrium regimes where the independent update of non-neighboring sites is physically valid and numerically efficient.

Numerical efficiency in the inner loops also relies on the elimination of expensive operations like the modulo operator for periodic boundary conditions. The simulation engine instead uses pre-calculated index arrays named `idx_next` and `idx_prev` to resolve neighbor positions. This approach replaces arithmetic divisions with fast memory lookups, which significantly reduces the computational cost per spin flip. When a simulation is initialized with a random seed, the engine synchronizes the Python random state with Numba's internal generator using the `_seed_numba` helper. This guarantee of determinism is essential for reproducing trajectories and debugging rare stochastic failures in complex energy landscapes.

## Error Handling and Type Safety

The framework maintains a rigorous approach to type safety and error reporting to support scientific reliability. Every source file employs `from __future__ import annotations` and uses comprehensive type hints to enable static analysis with Mypy. Public methods in the simulation classes use keyword-only arguments to prevent parameter ambiguity, which is particularly useful when configuring models with many physical constants. This strict interface design reduces the likelihood of silent failures or misconfiguration during automated parameter sweeps.

Errors are managed through a three-tier exception hierarchy defined in `utils/exceptions.py`. General configuration issues or invalid user inputs trigger standard `ValueError` exceptions. Physical or mathematical failures during data processing, such as an undefined autocorrelation time due to zero variance, raise specialized `NumericalAnalysisError` subclasses. Internal logic violations or impossible states result in a `RuntimeError`. This granular approach to error handling allows analysis scripts to catch and handle known failure modes gracefully while ensuring that fundamental bugs are immediately visible to the developer.

## Testing Strategy and Quality Gates

The VibeSpin verification suite is organized into five conceptual layers to ensure both code quality and physical correctness. Microscopic integrity tests in the algorithm layer validate the fundamental properties of the Monte Carlo kernels, including detailed balance and ergodicity [[1]](#Bibliography). The model layer focuses on API contracts, edge cases at extreme temperatures, and CLI behavior. Utility tests cover physics observables and system helpers. A dedicated style layer enforces documentation standards and docstring compliance. Finally, the integration layer uses reusable infrastructure patterns to verify script behavior, output schemas, and deterministic reproducibility.

Development workflows are protected by multi-stage quality gates implemented via pre-commit and pre-push hooks. These hooks run the full suite of static analysis tools, including Ruff for linting and Mypy for type checking. Documentation consistency is also enforced at push time: the system validates markdown links, ensures that API documentation is synchronized with the source, and builds the Sphinx HTML output with all warnings treated as errors. This rigorous pipeline ensures that every contribution maintains the high engineering standards required for reproducible scientific computing.

## Bibliography
## Uncertainty Data Contract and Pipeline

Analysis scripts and notebooks in VibeSpin share a standardized uncertainty schema for all serialized observables. This contract is defined in `utils/physics_helpers.py` and enforced by the integration test layer; any script that writes an NPZ file should conform to it.

### Schema Fields

For each observable `<obs>` saved by a temperature-sweep script, the NPZ file contains the following keys. The suffix `<obs>` takes values such as `avg_m`, `avg_e`, `susc`, and `spec_h`.

| Key | Shape | Meaning |
|-----|-------|---------|
| `<obs>_value` | `(T,)` | Point estimate (mean across seeds, or single-seed value) |
| `<obs>_err` | `(T,)` | Autocorrelation-aware standard error; `NaN` for single-seed runs |
| `<obs>_ci_low` | `(T,)` | Lower confidence-interval bound |
| `<obs>_ci_high` | `(T,)` | Upper confidence-interval bound |
| `<obs>_tau_int` | `(T,)` | Integrated autocorrelation time per temperature point |
| `<obs>_n_eff` | `(T,)` | Effective sample size; `NaN` where `tau_int` is undefined |
| `<obs>_samples` | `(T, S)` | Raw per-seed samples (S = 1 for single-seed runs) |

Global metadata keys are written once per file: `uncertainty_method` (string, currently `'blocking'`), `confidence_level` (float, default `0.68`), `n_seeds` (int), `bootstrap_resamples` (int, 0 if unused), and `nan_or_undefined_count` (float, count of temperature points with undefined `tau_int`).

Legacy keys (`avg_m`, `avg_e`, `susc`, `spec_h`, `entropy`, `tau_int`, `temperatures`) are preserved alongside the new keys for backward compatibility. Consumers should prefer the standardized keys where available.

### Utility API

The canonical implementation lives in `utils/physics_helpers.py`. The key public functions are as follows. `estimate_effective_sample_size` computes $N_\mathrm{eff} = N / (2\tau_{\mathrm{int}})$ from a time series, optionally accepting a pre-computed `tau_int` to avoid redundant autocorrelation calculation. `blocking_error` applies the plateau-selection blocking method and returns the plateau standard error alongside the estimated `tau_int` and `n_eff`. `summarize_primary_observable` wraps blocking into a single dict conforming to the schema fields above, handling the zero-variance edge case by storing `NaN` for the undefined fields. `summarize_derived_observable` extends this to nonlinear quadratic estimators like susceptibility and specific heat, supporting both the blocking propagation (default) and an optional block-bootstrap. `summarize_replicate_samples` aggregates a 2D `(T, S)` array of per-seed samples into a schema-consistent dict using the inter-seed distribution as the uncertainty source.

### Design Rationale

The schema is written additively: new standardized keys are always appended rather than replacing the legacy layout. This allows existing notebooks and downstream loaders to continue reading the pre-schema keys while new code targets the standardized ones. All constants (`DEFAULT_CONFIDENCE_LEVEL`, `UNCERTAINTY_METHOD_BLOCKING`, `UNCERTAINTY_METHOD_BOOTSTRAP`, `UNCERTAINTY_FIELDS`, `UNCERTAINTY_METADATA_FIELDS`) are defined at module level in `utils/physics_helpers.py` and imported by all scripts, ensuring a single authoritative source for the contract.

## Bibliography

The engineering and algorithmic choices in VibeSpin are grounded in standard practices for scientific computing and statistical physics. For a comprehensive list of all references used in the project, see [BIBLIOGRAPHY.md](./bibliography.md).

[[1]](#Bibliography) W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications," *Biometrika*, vol. 57, no. 1, pp. 97–109, 1970. [Oxford Academic Open Access](https://academic.oup.com/biomet/article/57/1/97/252073)

[[2]](#Bibliography) M. E. J. Newman and G. T. Barkema, "Monte Carlo Methods in Statistical Physics," Oxford University Press, 1999. [Lecture Notes Summary (H. G. Katzgraber)](https://arxiv.org/abs/0905.1629)

[[3]](#Bibliography) Numba Documentation: [Parallel Acceleration](https://numba.readthedocs.io/en/stable/user/parallel.html)
