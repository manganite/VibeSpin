# **VibeSpin**
### *A Vibe-Coded Journey through Statistical Mechanics*

**VibeSpin** is a Python framework designed to demonstrate how **vibe coding**—the high-level orchestration of AI to manifest complex logic—can be used to build and explore the deep landscapes of **statistical physics**. 

By bridging the gap between natural language intent and computational rigor, this project serves as both a showcase for modern AI-native development and an instructive tool for mastering discrete spin models.

---

### **Core Capabilities**
* **Lattice Simulations:** High-performance Python framework for simulating 2D lattice models, including **Ising**, **XY**, and **$q$-state Clock** models.
* **Phase Transition Analysis:** Built-in tools for analyzing critical phenomena, from standard ferromagnetic transitions to complex topological events like the **BKT (Berezinskii–Kosterlitz–Thouless)** transition.
* **Dynamic Kinetics:** Designed for studying **phase ordering kinetics** and the evolution of spatial correlation over time.
* **Instructive Workflow:** A transparent "vibe-coded" codebase that helps students move from the mathematical **Hamiltonian** to a functional **Monte Carlo** simulation using the power of Generative AI.

---

### **The Equilibrium**

| The Vibe (Methodology) | The Spin (Physics) |
| :--- | :--- |
| **AI-Native Orchestration** | **Monte Carlo / Metropolis** |
| **Rapid Iteration & Prompting** | **Finite-Size Scaling** |
| **Literate Vibe Coding** | **Statistical Equilibrium** |

---

## Technical Features

- **High Performance**: Numba JIT-compiled kernels achieving C-like simulation speeds.
- **Scientific Rigor**: Deterministic seed management for perfect reproducibility.
- **Deep Analysis**: Specialized tools for verifying growth laws ($L(t) \sim t^{1/2}$) and defect decay.
- **Visualization**: High-resolution snapshots of spatial order, vorticity maps, and correlations.
- **Extensive Metrics**: Calculates magnetization, energy, susceptibility, helicity modulus, and more.

## Core Technologies

- **Python 3.x** (≥ 3.9)
- **NumPy & Numba**: Numerical computation and JIT acceleration.
- **Matplotlib**: Data visualization and physical snapshots.
- **pytest**: Comprehensive testing suite.
- **Ruff & Mypy**: Code quality and type safety.

---

## **Getting Started**

### **Installation**
```bash
pip install -e .
```

> **Vibe Coding Tip:** Use natural language to describe new physical observables (e.g., *"Calculate the staggered magnetization for an antiferromagnet"*) and let your AI collaborator generate the optimized Numba kernel following the patterns in `models/simulation_base.py`.

### **Phase Transition & Equilibrium Analysis**
Analyze steady-state behavior and critical phenomena across a range of temperatures.

**1. Temperature Sweeps (XY Model)**
```bash
python scripts/xy/temperature_sweep.py --L 32 --T-min 0.2 --T-max 1.5 --steps 10
```

**2. BKT Transition Analysis**
```bash
python scripts/xy/bkt_transition.py --size 64 --temp 0.89
```

### **Non-Equilibrium Kinetics & Evolution**
Observe the time-dependent growth of order and the decay of topological defects.

**1. Study Phase Ordering (Ising Model)**
```bash
python scripts/ising/ordering_kinetics.py --size 256 --max-steps 1000
```

**2. Visualize Vortex Dynamics (XY Model)**
```bash
python scripts/xy/ordering_evolution.py --size 256 --targets 1 10 100 1000
```

---

## **Development Conventions**

- **Numba**: Always use `@njit(cache=True, fastmath=True)` for simulation kernels.
- **Reproducibility**: Use the `seed` parameter when initializing models.
- **Logging**: Use `utils/system_helpers.py:setup_logging` instead of standard `print()`.
- **Quality**: Run `ruff check .` and `mypy` before submitting changes.

> **Vibe Coding Tip:** If you encounter a Numba `TypingError`, share the error and the kernel code with the AI—it is exceptionally good at identifying unsupported Python objects or non-deterministic types within JIT-compiled blocks.

---

### **A Note on the Journey**
In **VibeSpin**, the code is not just a black box; it is a collaborative artifact. As you navigate from the simple flips of an Ising model to the swirling vortices of the XY model, you are participating in a new era of scientific computing—one where the "vibe" of the researcher and the precision of the physics exist in perfect symmetry.

For detailed developer and AI agent instructions, please refer to [AGENTS.md](AGENTS.md).
