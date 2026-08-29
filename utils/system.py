"""
Technical infrastructure for logging, parallel execution, and CLI argument parsing.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterable, Sized
from multiprocessing import Pool
from typing import cast

from tqdm import tqdm


def setup_logging(*, level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """
    Configure project-wide logging.

    Parameters
    ----------
        level: Logging level (e.g., logging.INFO).
        log_file: Optional path to a file to save logs to.

    Returns
    -------
        The configured logger instance.
    """
    logger = logging.getLogger('vibespin')
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
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# tqdm bar format that always shows rate as iterations/s (never inverts to s/it).
_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]'


def parallel_sweep(
    *, worker_func: Callable, params: Iterable, num_processes: int | None = None
) -> list:
    """
    Run a parallel sweep over a set of parameters using a worker function.
    Uses multiprocessing.Pool and tqdm for progress tracking.

    Parameters
    ----------
        worker_func: Function to execute in parallel.
        params: Iterable of parameters to pass to the worker function.
        num_processes: Number of processes to use. Defaults to CPU count.

    Returns
    -------
        List of results from the worker function.
    """
    # Try to get the length of params for tqdm without converting to list if possible
    total_len = len(params) if isinstance(params, Sized) else None

    with Pool(processes=num_processes) as pool:
        return list(tqdm(pool.imap(worker_func, params), total=total_len, bar_format=_BAR_FORMAT))


def parse_args_compat(*, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse CLI args with compatibility for parser wrappers used by some runners."""
    parse_arguments = getattr(parser, 'parse_arguments', None)
    if callable(parse_arguments):
        return cast(Callable[[], argparse.Namespace], parse_arguments)()
    return parser.parse_args()
