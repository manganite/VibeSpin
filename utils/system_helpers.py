"""
Technical utility functions for file system operations, plotting, and parallel execution.
"""

import logging
import os
import sys
from collections.abc import Callable, Iterable, Sequence, Sized
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# No typing import needed for modern syntax

def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """
    Configure project-wide logging.

    Args:
        level: Logging level (e.g., logging.INFO).
        log_file: Optional path to a file to save logs to.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger('multiferroic')
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            ensure_results_dir(os.path.dirname(log_file))
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger

def ensure_results_dir(directory: str = 'results') -> str:
    """
    Ensure the results directory exists.

    Args:
        directory: Name of the directory to create.

    Returns:
        The path to the directory.
    """
    if directory:
        os.makedirs(directory, exist_ok=True)
    return directory

def save_plot(filename: str, directory: str = 'results', tight_layout: bool = True) -> None:
    """
    Save the current matplotlib plot to the results directory.

    Args:
        filename: Name of the output file (e.g., 'plot.png').
        directory: Output directory name.
        tight_layout: Whether to apply plt.tight_layout() before saving.
    """
    logger = logging.getLogger('multiferroic')
    ensure_results_dir(directory)
    if tight_layout:
        try:
            plt.tight_layout()
        except UserWarning:
            pass
    path = os.path.join(directory, filename)
    plt.savefig(path)
    logger.info(f"Plot saved to {path}")

# tqdm bar format that always shows rate as iterations/s (never inverts to s/it).
_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]'

def parallel_sweep(
    worker_func: Callable, params: Iterable, num_processes: int | None = None
) -> list:
    """
    Run a parallel sweep over a set of parameters using a worker function.
    Uses multiprocessing.Pool and tqdm for progress tracking.

    Args:
        worker_func: Function to execute in parallel.
        params: Iterable of parameters to pass to the worker function.
        num_processes: Number of processes to use. Defaults to CPU count.

    Returns:
        List of results from the worker function.
    """
    # Try to get the length of params for tqdm without converting to list if possible
    total_len = len(params) if isinstance(params, Sized) else None

    with Pool(processes=num_processes) as pool:
        return list(tqdm(pool.imap(worker_func, params), total=total_len, bar_format=_BAR_FORMAT))


def plot_temperature_sweep(
    temperatures: np.ndarray,
    avg_m: Sequence[float],
    avg_e: Sequence[float],
    susc: Sequence[float],
    spec_h: Sequence[float],
    title: str,
    filename: str,
    directory: str,
) -> None:
    """
    Generate and save a standardized 4-panel temperature sweep plot.

    Displays magnetization, energy, susceptibility, and specific heat as
    functions of temperature. Saves the figure via :func:`save_plot`.

    Args:
        temperatures: Array of temperature values (x-axis).
        avg_m: Average absolute magnetization per temperature point.
        avg_e: Average energy per temperature point.
        susc: Magnetic susceptibility per temperature point.
        spec_h: Specific heat per temperature point.
        title: Figure-level suptitle string (e.g. '2D Ising Model: Temperature Sweep (L=50)').
        filename: Output filename passed to :func:`save_plot` (e.g. 'temperature_sweep.png').
        directory: Output directory passed to :func:`save_plot` (e.g. 'results/ising').
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(temperatures, avg_m, 'o-', markersize=4)
    ax1.set_ylabel('Average Magnetization |M|')
    ax1.set_title('Magnetization')
    ax1.grid(True)

    ax2.plot(temperatures, avg_e, 'o-', color='orange', markersize=4)
    ax2.set_ylabel('Average Energy')
    ax2.set_title('Energy')
    ax2.grid(True)

    ax3.plot(temperatures, susc, 'o-', color='green', markersize=4)
    ax3.set_ylabel(r'Susceptibility $\chi$')
    ax3.set_title('Magnetic Susceptibility')
    ax3.grid(True)

    ax4.plot(temperatures, spec_h, 'o-', color='red', markersize=4)
    ax4.set_ylabel(r'Specific Heat $C_v$')
    ax4.set_title('Specific Heat')
    ax4.grid(True)

    for ax in axes.flatten():
        ax.set_xlabel('Temperature (T)')

    save_plot(filename, directory=directory)


def plot_domain_evolution(
    targets: Sequence[int],
    snapshots: Sequence[np.ndarray],
    gr_data: Sequence[tuple[np.ndarray, np.ndarray]],
    vorticity_data: Sequence[np.ndarray] | None,
    title: str,
    filename: str,
    directory: str,
    is_vector: bool = False,
) -> None:
    """
    Generate and save a multi-row figure showing domain evolution over time.

    Row 0: Spin configurations (binary for Ising, HSV for XY/Clock).
    Row 1: Circularly-averaged structure factor S(|k|) or Vorticity maps.
    Row 2: Radially averaged correlation functions G(r) with xi estimates.

    Args:
        targets: List of MC steps for each snapshot.
        snapshots: List of spin arrays.
        gr_data: List of (r, G) tuples.
        vorticity_data: Optional list of vorticity arrays.
        title: Figure-level suptitle.
        filename: Output filename.
        directory: Output directory.
        is_vector: Whether the spins are 2D vectors (True) or scalars (False).
    """
    n_cols = len(targets)
    # Determine rows: 3 rows if vorticity is provided, otherwise 3 rows (with Sk)
    fig, axes = plt.subplots(3, n_cols,
                             figsize=(n_cols * 4, 12.5),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.25})
    fig.suptitle(title, fontsize=14, y=0.98)

    for col in range(n_cols):
        t = targets[col]
        spins = snapshots[col]
        
        # --- Row 0: Spin Configuration ---
        ax_spin = axes[0, col]
        if is_vector:
            angles = np.arctan2(spins[..., 1], spins[..., 0])
            im = ax_spin.imshow(angles, cmap='hsv', interpolation='none',
                                vmin=-np.pi, vmax=np.pi)
            if col == n_cols - 1:
                plt.colorbar(im, ax=ax_spin, label='Phase (rad)', shrink=0.8)
        else:
            ax_spin.imshow(spins, cmap='binary', interpolation='none',
                           vmin=-1, vmax=1)
        ax_spin.set_title(f't = {t} sweep{"s" if t != 1 else ""}', fontsize=12)
        ax_spin.axis('off')

        # --- Row 1: Mid-row (Vorticity or Sk) ---
        ax_mid = axes[1, col]
        if vorticity_data is not None:
            vort = vorticity_data[col]
            im_v = ax_mid.imshow(vort, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
            ax_mid.axis('off')
            v_count = int(np.sum(np.abs(vort)))
            ax_mid.set_title(f'Vortices: {v_count}', fontsize=10)
            if col == n_cols - 1:
                plt.colorbar(im_v, ax=ax_mid, ticks=[-1, 0, 1], label='Winding No.', shrink=0.8)
        else:
            # Fallback to structure factor Sk if no vorticity (Ising style)
            # Importing here to avoid circular dependencies
            from .physics_helpers import radial_average_sk
            k_vals, S_radial = radial_average_sk(spins)
            ax_mid.plot(k_vals[1:], S_radial[1:], linewidth=1.2)
            ax_mid.set_xscale('log')
            ax_mid.set_yscale('log')
            ax_mid.set_xlabel('$|k|$', fontsize=9)
            if col == 0:
                ax_mid.set_ylabel('$S(|k|)$', fontsize=9)
            ax_mid.grid(True, which='both', alpha=0.25)

        # --- Row 2: Correlation G(r) ---
        ax_gr = axes[2, col]
        r, G = gr_data[col]
        ax_gr.plot(r[1:], G[1:], linewidth=1.5)
        ax_gr.axhline(0, color='tab:gray', linewidth=0.7, linestyle='--')
        
        inv_e = 1.0 / np.e
        ax_gr.axhline(inv_e, color='tab:red', linewidth=0.8,
                      linestyle=':', alpha=0.7, label='$1/e$')
        
        ax_gr.set_xscale('log')
        ax_gr.set_xlabel('Distance r', fontsize=10)
        if col == 0:
            ax_gr.set_ylabel('$G(r)$ / $G(0)$', fontsize=10)
        ax_gr.grid(True, which='both', alpha=0.3)
        ax_gr.set_ylim(-0.1, 1.1)

        # Find xi where G(r) first drops below 1/e
        r_plot = r[1:]
        G_plot = G[1:]
        below = np.where(G_plot < inv_e)[0]
        if len(below) > 0:
            idx = below[0]
            if idx > 0:
                r0, r1 = float(r_plot[idx - 1]), float(r_plot[idx])
                g0, g1 = float(G_plot[idx - 1]), float(G_plot[idx])
                xi = r0 + (inv_e - g0) * (r1 - r0) / (g1 - g0)
            else:
                xi = float(r_plot[idx])
            ax_gr.axvline(xi, color='tab:red', linewidth=1.0, linestyle='--', alpha=0.8)
            ax_gr.text(xi * 1.15, inv_e + 0.04,
                       f'$\\xi = {xi:.1f}$', fontsize=9, color='tab:red')

    save_plot(filename, directory=directory)
