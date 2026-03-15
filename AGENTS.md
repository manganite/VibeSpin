# Agent Instruction Guide: VibeSpin

This document provides mandatory procedural context and technical constraints for AI Agents working on this codebase.

## Mandatory Development Policies

### 1. High Performance Computing (Numba JIT)
- **Constraint**: All simulation loops and kernels MUST be JIT-compiled. Use `@njit(cache=True, fastmath=True)`.
- **Constraint**: Optional multi-core parallelization should be implemented for checkerboard kernels using `parallel=True` and `prange`.
- **Constraint**: Do not use `np.mod` or `%` for Periodic Boundary Conditions (PBCs). Use pre-calculated indices `self.idx_next` and `self.idx_prev`.
- **Constraint**: Minimize memory allocation inside JIT loops; update arrays in-place whenever possible.

### 2. Code Quality & Type Safety
- **Type Hints**: Every source file MUST include `from __future__ import annotations` as the first import.
- **API Safety**: Use `*` to force **keyword-only arguments** for all public simulation and analysis methods.
- **CLI Patterns**: Simulation models MUST include a `main()` entry point refactoring the CLI logic to support unit testing via mocking.
- **Import Strategy**: Use **relative imports** within the same package namespace. Use **absolute imports** for cross-package and script/test imports.

### 3. Physical Fidelity & Algorithm Integrity
- **Metropolis Prerequisites**: All update algorithms MUST strictly fulfill the conditions for the Metropolis-Hastings algorithm: **Detailed Balance**, **Ergodicity**, and **Symmetric Proposals**.
- **Dynamics/Kinetics Mandate**: Use **Random Site Selection** for non-equilibrium studies. Sequential/checkerboard updates are physically invalid for these regimes.
- **Thermodynamics/Equilibrium Mandate**: Use **Checkerboard Updates** for steady-state measurements to maximize SIMD and multi-core throughput.
- **Discrete Speedup**: Prefer discrete state representations (integer state indices) for q-state models to avoid per-site trigonometric evaluations.

### 4. Verification & Testing
- **Comprehensive Testing**: Any modification to simulation kernels or observables must be verified by running the full test suite:
  ```bash
  pytest
  ```
- **Integrity Probes**: Ensure new physical logic is covered by microscopic integrity tests (e.g., in `tests/test_algorithm_integrity.py`) and parameter validation (e.g., in `tests/test_model_extremes.py`).
- **Static Analysis**: Maintain strict quality standards:
  ```bash
  ruff check .
  mypy --explicit-package-bases models/ utils/ scripts/
  ```

### 5. Source Control & Delivery
- **Pre-Commit Check**: Before proposing a commit, you MUST run all tests, linting, and type checking.
- **Commit Format**: Use **Conventional Commits** (`type(scope): description`). Example: `phys(xy): implement helicity modulus calculation`.
- **GitHub Sync**: After a successful local commit, ask the user if they wish to push to the remote repository.

### 6. Documentation & Knowledge Management
- **Docstring Compliance**: All new classes, methods, and kernels MUST include **NumPy-style docstrings**. This is mandatory for automated Sphinx API generation (`sphinx-apidoc`).
- **Theory Updates**: When introducing new physical models or observables, you MUST update **`PHYSICS.md`** with the relevant Hamiltonian definitions, phase behavior, and mathematical formulations.
- **Scripts Catalog**: Any new entry-point script added to `scripts/` MUST be registered in **`SCRIPTS.md`** with a brief description of its purpose and usage.
- **Performance Re-profiling**: If a change significantly impacts simulation throughput or analysis overhead, you MUST re-run the benchmark tool (`benchmark.py`) and update the **`Performance_Benchmarks.ipynb`** summary results.
- **Cross-linking**: Standalone documentation files MUST be cross-linked in the Sphinx hub (`docs/source/index.md`) to ensure they appear in the hosted documentation site.

## Directory Map for Agents
...

- `models/`: Refactored simulation classes with `main()` entry points.
- `utils/`: Physics and system-level helper functions.
- `tests/`: High-coverage test suite including integrity, CLI, and extreme case verification.
- `scripts/`: Physics experiments and equilibrium/kinetics drivers.

## Common Operational Workflows

### Task: Implement a New Physical Observable
1. Add the `@njit` kernel to `models/simulation_base.py` or the specific model.
2. Add a `_get_<name>` method to the Simulation class.
3. Add a test case in `tests/test_coverage_enhancement.py` or a specialized test file.
4. Verify the physical limits (e.g., ground state) in `tests/test_model_extremes.py`.

### Task: Investigate Performance Regression
1. Run the benchmark tool: `python benchmark.py --sizes 512 1024 --sweeps 100`.
2. Check the **Pure Simulation Time** vs. overhead in the summary table.
3. Profile the kernel for unexpected allocations or `object mode` fallbacks.

## Explanatory Writing Style

### Role
You are an excellent human writer with a high-level scientific background. Write explanatory text in a human voice with clarity, precision, and conciseness.

### Instructions
- **Destroy the List**: In explanatory prose, do not use bullet points unless procedural. Use continuous, flowing prose.
- **Vary Sentence Length**: Avoid a monotonous rhythm of medium-length sentences.
- **Mechanism over Slogan**: Favor technical interpretation over abstract praise. Plainly describe crossovers, plateaus, and decay laws.
- **Regime Awareness**: Always specify whether a claim concerns equilibrium, kinetics, topological defects, or numerical cost.
- **No Conversational Filler**: Adopt a direct, professional tone suitable for a CLI environment. Fulfill the user's request thoroughly while maintaining simplicity.
