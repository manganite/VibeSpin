"""
Diagnostic: raw two-start equilibration traces at four representative temperatures.

Run from the repo root:
    python scripts/ising/_diag_eq_traces.py
"""
from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from models.ising_model import IsingSimulation as IsingModel

# ── Parameters ───────────────────────────────────────────────────────────────
TC          = 2.0 / np.log(1.0 + np.sqrt(2.0))   # ≈ 2.2692
TEMPERATURES = [0.5, 2.0, TC, 2.5]
T_LABELS     = [r"$T = 0.5$  (deep ordered)",
                r"$T = 2.00$  (below $T_c$)",
                rf"$T = T_c \approx {TC:.3f}$",
                r"$T = 2.50$  (above $T_c$)"]

SIZE         = 64
CHUNK_SIZE   = 50        # sweeps per chunk
MAX_STEPS    = 5000      # total sweeps recorded per sim
N_CHUNKS     = MAX_STEPS // CHUNK_SIZE
# Scan seeds 0-19 at T=0.5 to find ones that get stuck (|M|<0.5 after a short run)
_SCAN_STEPS  = 500  # quick pre-screen length
_STUCK_CHECK_TEMP = TEMPERATURES[0]  # T=0.5

SEED_COLORS  = ["#2166ac", "#d6604d", "#4dac26",
                "#984ea3", "#ff7f00", "#a65628"]   # up to 6 seeds
# ─────────────────────────────────────────────────────────────────────────────


def collect_trace(*, size: int, temp: float, seed: int,
                  init_state: str, n_chunks: int, chunk_size: int) -> np.ndarray:
    """Run sim for n_chunks × chunk_size sweeps; return per-sweep |M| trace."""
    sim = IsingModel(size=size, temp=temp, init_state=init_state,
                     update='checkerboard', seed=seed)
    mags: list[float] = []
    for _ in range(n_chunks):
        m, _ = sim.run(n_steps=chunk_size)
        mags.extend(m)
    return np.array(mags)


# ── Pre-screen: find seeds where random start stays stuck at T=0.5 ────────────
print(f"Scanning seeds 0-24 at T={_STUCK_CHECK_TEMP} L={SIZE} for stuck vs. ordering ...")
stuck_seeds: list[int] = []
ordering_seeds: list[int] = []
for s in range(25):
    tr = collect_trace(size=SIZE, temp=_STUCK_CHECK_TEMP, seed=s,
                       init_state='random', n_chunks=_SCAN_STEPS // CHUNK_SIZE,
                       chunk_size=CHUNK_SIZE)
    final_m = float(np.abs(tr[-50:]).mean())
    if final_m < 0.5:
        stuck_seeds.append(s)
    else:
        ordering_seeds.append(s)

print(f"  Stuck seeds (|M|<0.5 after {_SCAN_STEPS} sweeps): {stuck_seeds[:6]}")
print(f"  Ordering seeds: {ordering_seeds[:6]}")

# Pick up to 3 stuck + 3 ordering seeds for display
SEEDS_STUCK    = stuck_seeds[:3]
SEEDS_ORDERING = ordering_seeds[:3]
SEEDS          = sorted(set(SEEDS_STUCK + SEEDS_ORDERING))
print(f"  Using seeds for T=0.5 panel: stuck={SEEDS_STUCK}, ordering={SEEDS_ORDERING}")
# For other temperatures, just use seeds 1-3
SEEDS_OTHER = [1, 2, 3]
# ─────────────────────────────────────────────────────────────────────────────


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes_flat = axes.flat
sweeps = np.arange(1, MAX_STEPS + 1)

for ax, temp, label in zip(axes_flat, TEMPERATURES, T_LABELS, strict=True):
    seeds_to_use = SEEDS if temp == TEMPERATURES[0] else SEEDS_OTHER
    for idx, seed in enumerate(seeds_to_use):
        color = SEED_COLORS[idx % len(SEED_COLORS)]
        tr = collect_trace(size=SIZE, temp=temp, seed=seed, init_state='random',
                           n_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE)
        to = collect_trace(size=SIZE, temp=temp, seed=seed, init_state='ordered',
                           n_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE)

        stuck_tag = " ★stuck" if (temp == TEMPERATURES[0] and seed in SEEDS_STUCK) else ""
        ax.plot(sweeps, np.abs(tr), color=color, lw=0.7, alpha=0.65,
                label=f"seed {seed}{stuck_tag} (rand)" if temp == TEMPERATURES[0] else None)
        ax.plot(sweeps, to, color=color, lw=0.7, alpha=0.65,
                linestyle='--',
                label=f"seed {seed}{stuck_tag} (ord)" if temp == TEMPERATURES[0] else None)

    ax.set_title(label, fontsize=11)
    ax.set_xlabel("MC sweep", fontsize=9)
    ax.set_ylabel(r"$|M|$", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, MAX_STEPS)

    def x_fmt(x: float, _: Any) -> str:
        return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_fmt))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(True, linewidth=0.4, alpha=0.5)

# Legend from the first panel only
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8,
           title="solid = random start  |  dashed = ordered start",
           title_fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

beta_c_str = r"$\beta_c = \ln(1+\sqrt{2})/2$"
fig.suptitle(
    rf"Two-start equilibration traces  ·  Ising $L={SIZE}$  ·  3 seeds",
    fontsize=13, y=1.01
)

plt.tight_layout()
outpath = "results/ising/diagnostic_eq_traces_L64.png"
os.makedirs("results/ising", exist_ok=True)
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved → {outpath}")
