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


def plot_ordering_kinetics(
    t: np.ndarray,
    R_sk: np.ndarray,
    R_xi: np.ndarray,
    third_metric: np.ndarray | None,
    third_metric_label: str | None,
    exponents: dict[str, float | None],
    prefactors: dict[str, float | None],
    fit_mask: np.ndarray,
    title: str,
    filename: str,
    directory: str,
    y_label: str = 'Characteristic Length Scale $L(t)$ (lattice units)',
    left_title: str = 'Domain Coarsening',
    right_title: str = 'Defect/Boundary Evolution',
) -> None:
    """
    Generate and save a 2-panel figure showing ordering kinetics.

    Left Panel: Growth of length scales R_sk and xi.
    Right Panel: Growth/Decay of a third metric (Vortex density or MIL).

    Args:
        t: Time array (Monte Carlo sweeps).
        R_sk: Domain size from structure factor.
        R_xi: Correlation length from G(r).
        third_metric: Optional third metric array.
        third_metric_label: Label for the third metric.
        exponents: Dict of fitted exponents.
        prefactors: Dict of fitted prefactors.
        fit_mask: Mask used for fitting.
        title: Figure-level suptitle.
        filename: Output filename.
        directory: Output directory.
        y_label: Y-axis label for the left panel.
        left_title: Title for the left subplot.
        right_title: Title for the right subplot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13)

    # 1. Length Scale Growth (Log-Log)
    ax1.loglog(t, R_sk, 'o', label='$R_{S(k)}$ (Structure Factor)')
    if exponents.get('R_sk') is not None:
        exp, pre = exponents['R_sk'], prefactors['R_sk']
        ax1.loglog(t[fit_mask], pre * t[fit_mask] ** exp, '--', color='tab:blue',
                   label=f'Fit $R_{{sk}}$: $t^{{{exp:.2f}}}$')

    ax1.loglog(t, R_xi, 's', label='$\\xi$ (Correlation length)')
    if exponents.get('xi') is not None:
        exp, pre = exponents['xi'], prefactors['xi']
        ax1.loglog(t[fit_mask], pre * t[fit_mask] ** exp, '--', color='tab:orange',
                   label=f'Fit $\\xi$: $t^{{{exp:.2f}}}$')

    ax1.set_xlabel('Time t (Monte Carlo sweeps)')
    ax1.set_ylabel(y_label)
    ax1.set_title(left_title)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    # 2. Third Metric (Log-Log)
    if third_metric is not None:
        ax2.loglog(t, third_metric, 'D', color='tab:purple', label=third_metric_label)
        if exponents.get('third') is not None:
            exp, pre = exponents['third'], prefactors['third']
            ax2.loglog(t[fit_mask], pre * t[fit_mask] ** exp, 'k--',
                       label=f'Fit: $t^{{{exp:.2f}}}$')
        ax2.set_xlabel('Time t (Monte Carlo sweeps)')
        ax2.set_ylabel(third_metric_label)
        ax2.set_title(right_title)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend()
    else:
        ax2.axis('off')

    save_plot(filename, directory=directory)
