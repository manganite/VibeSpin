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
        plt.tight_layout()
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
