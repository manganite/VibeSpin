# Agent Instruction Guide: VibeSpin

This document provides mandatory procedural context and technical constraints for AI Agents working on this codebase.

## Mandatory Development Policies

### 1. High Performance Computing (Numba JIT)
- **Constraint**: All simulation loops and kernels MUST be JIT-compiled. Use `@njit(cache=True, fastmath=True)`.
- **Constraint**: Do not use `np.mod` or `%` for Periodic Boundary Conditions (PBCs). Use pre-calculated indices `self.idx_next` and `self.idx_prev`.
- **Constraint**: Minimize memory allocation inside JIT loops; update arrays in-place whenever possible.

### 2. Code Quality & Type Safety
- **Type Hints**: Every source file MUST include `from __future__ import annotations` as the first import to support modern type hinting and forward references.
- **API Safety**: Use `*` to force **keyword-only arguments** for all public simulation and analysis methods (e.g., `def run(self, *, steps: int, temp: float) -> None:`). This prevents transposition errors in complex physical APIs.
- **Import Strategy**: Use **relative imports** within the package (e.g., `from ..utils import ...`) and **absolute imports** in `tests/` and `scripts/`.

### 3. Physical Fidelity & Update Algorithms
- **Dynamics/Kinetics Mandate**: When simulating non-equilibrium kinetics (coarsening, aging), you MUST use **Random Site Selection** (e.g., `xy_step_random_numba`). Sequential or checkerboard updates are physically invalid for these studies.
- **Thermodynamics/Equilibrium Mandate**: For steady-state measurements or temperature sweeps, you SHOULD use **Checkerboard Updates** for higher throughput.
- **Vorticity**: Calculate using directed phase differences around plaquettes as implemented in `models/simulation_base.py`.

### 4. Verification & Testing
- **Interface Testing**: Focus tests on **physical observables** (magnetization, energy, correlations) rather than internal implementation details.
- **Reproducibility**: Models must sync Numba's internal RNG with the global seed using `models.simulation_base._seed_numba(seed)`.
- **Unit Testing**: Any change to `models/` or `utils/physics_helpers.py` must be verified by running:
  ```bash
  pytest tests/test_models.py tests/test_reproducibility.py
  ```
- **Static Analysis**: Maintain type safety and linting quality:
  ```bash
  ruff check .
  mypy --explicit-package-bases models/ utils/ scripts/
  ```

### 5. Source Control & Delivery
- **Pre-Commit Check**: Before proposing a commit, you MUST run the full test suite and linting (`pytest`, `ruff`, `mypy`).
- **Commit Format**: Use **Conventional Commits** (`type(scope): description`). Valid types: `phys` (physics logic), `feat`, `fix`, `perf`, `docs`, `test`, `chore`. Example: `phys(xy): implement helicity modulus calculation`.
- **GitHub Sync**: After a successful local commit, always ask the user if they wish to push to the remote repository.

## Directory Map for Agents
...

- `models/`: Implementations of Hamiltonian dynamics.
  - `simulation_base.py`: Abstract base class `MonteCarloSimulation`.
- `utils/`: Core shared logic.
  - `physics_helpers.py`: Math-heavy analysis (power-law fitting, correlation functions).
  - `system_helpers.py`: I/O, CLI, and parallelization.
- `scripts/`: Entry points for specific physics experiments.
  - `*_kinetics.py`: Non-equilibrium studies ($T=0$ or $T < T_c$).
  - `*_evolution.py`: Visual/Snapshot generation.
  - `*_sweep.py`: Thermodynamic equilibrium sweeps.

## Common Operational Workflows

### Task: Implement a New Physical Observable
1. Identify if the observable requires a new JIT kernel.
2. If yes, add it as a `@njit` helper in `models/simulation_base.py` (if shared) or the specific model file.
3. Add a `_get_<name>` method to the Simulation class.
4. Add a test case in `tests/test_models.py` to verify the calculation against a known configuration (e.g., ground state).

### Task: Investigate Performance Regression
1. Run the benchmark tool: `python benchmark.py --sizes 128 256 --sweeps 500`.
2. Compare results with `results/benchmarks/`.
3. Check for `object mode` fallbacks in Numba (ensure no `np.random` calls or unsupported Python objects inside kernels).

### Task: Analyze Growth Law Exponents
1. Use `utils/physics_helpers.py:power_fit` for robust extraction.
2. Ensure the time range for fitting avoids the initial transient and final saturation regimes.
