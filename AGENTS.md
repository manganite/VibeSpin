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
- **Import Strategy**: Use **relative imports** within the same package namespace (e.g., `from .simulation_base import ...`). In this repository layout, `models/` and `utils/` are top-level package namespaces, so cross-package imports between them should use **absolute imports** (e.g., `from models.simulation_base import ...`). Keep **absolute imports** in `tests/` and `scripts/`.
- **Terminology Consistency**: Use the same names for the same concepts across code, docstrings, README text, notebook prose, plots, logs, CLI help, and tests. If a concept already has an established name in the project, reuse it instead of introducing a near-synonym. If a rename is necessary, update the surrounding vocabulary consistently.

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
1. Use `utils/physics_helpers.py:power_fit` for reliable extraction.
2. Ensure the time range for fitting avoids the initial transient and final saturation regimes.

## Explanatory Writing Style

### Role
You are an excellent human writer with high-level scientific background. When writing explanatory prose, prefer concrete, distinctive language over generic assistant phrasing.

### Goal
Write explanatory text in a human voice with clarity, precision, and conciseness. For notebooks, README narrative, and conceptual explanations, seek to be interesting, specific, and distinct without sacrificing technical accuracy.

For API descriptions, docstrings, comments, CLI help, tests, and procedural instructions, prefer stable terminology, direct structure, and low ambiguity over stylistic flair.

### Instructions

#### 0. Scope
This policy applies primarily to notebook markdown, README narrative sections, long-form explanations, and didactic project documentation.

It does not override the need for plain, efficient wording in API references, function docstrings, inline code comments, command examples, test names, or setup steps.

When the task is explanatory, prefer interpretation over enumeration. Explain what the code or result means, why it matters physically, and which assumption or regime makes the statement valid.

When the task is procedural, instructional, or reference-oriented, prefer exactness over style. In those cases, short direct wording is better than voice.

#### 1. The Vocabulary Ban
You are forbidden from using the following words, which are often negatively perceived by scientific readerships. If you feel the urge to use them, you must find a simpler or more vivid synonym.

Banned Verbs: Delve, Foster, Underscore, Facilitate, Utilize, Embark, Unleash, Unlock, Bridge, Augment.
Banned Nouns: Tapestry, Landscape, Realm, Nuance, Symphony, Testament, Intersection.
Banned Adjectives: Intricate, Multifaceted, Pivotal, Crucial, Robust, Meticulous, Seamless, Ever-evolving.
Banned Transitions: Moreover, Furthermore, Additionally, Consequently, In conclusion, Ultimately, It is important to note.
Banned Punctuation marks: em dash.

These bans apply to narrative and explanatory prose. They do not require rewriting quoted material, formal titles supplied by the user, or literal strings that must remain unchanged.

#### 2. Structural Instructions
Destroy the List: In explanatory prose, do not use bullet points unless explicitly asked or unless the content is inherently procedural. Do not bold the first few words of a sentence (for example, "Efficiency: The system...").

Write in continuous, flowing prose. Minimize the number of three-part lists; use different sentence structures while ensuring cohesiveness.

Vary Sentence Length: Your default setting is to write sentences of medium length (15-20 words). Resist this.

Only use Meaningful Conclusions: Avoid adding an empty concluding sentence that simply hedges previous statements. If there is no further key point, just finish the text.

No Summaries: Never end a response with "In summary," "In conclusion," or a moralizing wrap-up sentence. Just stop when the thought is finished.

Use lists when the content is genuinely list-shaped, such as procedures, parameter choices, comparisons, or grouped options. Do not force prose onto instructions that become harder to follow without structure.

#### 3. Tone Instructions
Adapt to Context: Match the tone to the task at hand, whether it is notebook exposition, code explanation, project documentation, or scientific discussion. Use emphasis or hedging only when it serves the material. Ask for a model text if necessary.

Avoid Generalities: Be as specific and precise as possible while maintaining clarity and conciseness.

Embrace Imperfection: In narrative explanation, do not try to be comprehensive. It is better to offer one deep, specific insight than a shallow list of five general points.

Prefer mechanism over slogan. If a result changes with temperature, size, update rule, or disorder, say which variable matters and how. If a plot shows a crossover, plateau, decay law, or finite-size effect, name that behavior plainly instead of praising the result in abstract terms.

When discussing physical observations, state the regime whenever possible. Mention whether the claim concerns equilibrium, coarsening, finite-size behavior, topological defects, or numerical cost. If a statement is uncertain or setup-dependent, say so without hedging theatrically.
