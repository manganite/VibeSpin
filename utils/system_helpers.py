"""
Technical utility functions for file system operations, plotting, and parallel execution.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from multiprocessing import Pool
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.exceptions import ZeroVarianceAutocorrelationError


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
            ensure_results_dir(directory=os.path.dirname(log_file))
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def ensure_results_dir(*, directory: str = 'results') -> str:
    """
    Ensure the results directory exists.

    Parameters
    ----------
        directory: Name of the directory to create.

    Returns
    -------
        The path to the directory.
    """
    if directory:
        os.makedirs(directory, exist_ok=True)
    return directory


def save_plot(*, filename: str, directory: str = 'results', tight_layout: bool = True) -> None:
    """
    Save the current matplotlib plot to the results directory.

    Parameters
    ----------
        filename: Name of the output file (e.g., 'plot.png').
        directory: Output directory name.
        tight_layout: Whether to apply plt.tight_layout() before saving.
    """
    logger = logging.getLogger('vibespin')
    ensure_results_dir(directory=directory)
    if tight_layout:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            plt.tight_layout()
    path = os.path.join(directory, filename)
    plt.savefig(path)
    logger.info(f'Plot saved to {path}')


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


class _Sim(Protocol):
    """Structural type for any MonteCarloSimulation; avoids a models/ → utils/ import."""

    def equilibrate(self, *, n_steps: int) -> None: ...

    def run(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray]: ...


def adaptive_equilibrate(
    sim: _Sim,
    *,
    min_steps: int,
    probe_steps: int = 500,
    factor: float = 50.0,
    max_steps: int = 200_000,
) -> int:
    """
    Equilibrate a simulation adaptively, extending the burn-in until the probe
    window covers ``factor`` integrated autocorrelation times.

    After the mandatory ``min_steps`` burn-in, the function repeatedly runs a
    ``probe_steps`` measurement via ``sim.run()`` and computes ``tau_int`` from
    the resulting magnetization series. If ``probe_steps >= factor * tau_int``,
    the probe spans enough correlation times for the initial state to be
    forgotten and equilibration is declared complete. Otherwise the probe has
    advanced the system state and the loop continues. In the ordered phase the
    magnetization series has zero variance; this is treated as full
    equilibration and the function returns immediately.

    Parameters
    ----------
        sim: Any simulation object implementing ``equilibrate`` and ``run``.
        min_steps: Mandatory burn-in passed to ``sim.equilibrate`` before probing.
        probe_steps: MC steps per probe run (default 500).
        factor: Required ratio ``probe_steps / tau_int`` (default 50.0).
        max_steps: Hard cap on total steps to prevent unbounded runtime near
            criticality (default 200 000).

    Returns
    -------
        Total number of MC steps run (burn-in + probes).

    Raises
    ------
        ValueError: If the adaptive-equilibration parameters are invalid.
    """
    from .physics_helpers import calculate_autocorr  # lazy import; avoids pickle issues

    if min_steps < 0:
        raise ValueError(f'min_steps must be non-negative, got {min_steps}')
    if probe_steps < 3:
        raise ValueError(f'probe_steps must be >= 3, got {probe_steps}')
    if factor <= 0.0:
        raise ValueError(f'factor must be positive, got {factor}')
    if max_steps < min_steps:
        raise ValueError(
            f'max_steps must be >= min_steps, got max_steps={max_steps} and min_steps={min_steps}'
        )

    logger = logging.getLogger('vibespin')
    sim.equilibrate(n_steps=min_steps)
    total = min_steps

    while total < max_steps:
        mags, _ = sim.run(n_steps=probe_steps)
        total += probe_steps
        try:
            _, tau_int = calculate_autocorr(time_series=mags)
        except ZeroVarianceAutocorrelationError:
            # Zero variance: fully ordered phase, equilibration is trivially satisfied.
            return total
        if probe_steps >= factor * tau_int:
            return total

    logger.warning(
        f'adaptive_equilibrate: reached max_steps={max_steps} without satisfying '
        f'criterion probe_steps({probe_steps}) >= factor({factor}) * tau_int; '
        'proceeding anyway.'
    )
    return total


def convergence_equilibrate(
    sim_random: _Sim,
    sim_ordered: _Sim,
    *,
    chunk_size: int = 500,
    max_steps: int = 200_000,
    **kwargs: Any,
) -> int:
    """
    Equilibrate two simulations (random- and ordered-start) until they converge.

    Uses ``estimate_relaxation_time_two_start`` to detect when the two
    trajectories have entered the same equilibrium band. This is more robust
    than one-start adaptive methods for complex energy landscapes.

    Parameters
    ----------
        sim_random: Simulation instance started from a random state.
        sim_ordered: Simulation instance started from an ordered state.
        chunk_size: Number of steps to run between convergence checks.
        max_steps: Hard cap on total steps.
        **kwargs: Passed to ``estimate_relaxation_time_two_start`` (k, smooth, dwell, etc.).

    Returns
    -------
        Total number of MC steps run per simulation.
    """
    from .physics_helpers import estimate_relaxation_time_two_start  # lazy import

    logger = logging.getLogger('vibespin')
    mags_r: list[float] = []
    mags_o: list[float] = []
    total = 0

    while total < max_steps:
        mr, _ = sim_random.run(n_steps=chunk_size)
        mo, _ = sim_ordered.run(n_steps=chunk_size)
        mags_r.extend(mr)
        mags_o.extend(mo)
        total += chunk_size

        # Only check if we have enough data for a meaningful estimate
        # smooth_window defaults to 30, dwell_window to 30.
        if total >= 100:
            tau = estimate_relaxation_time_two_start(
                trace_random=np.array(mags_r),
                trace_ordered=np.array(mags_o),
                **kwargs,
            )
            # If estimate_relaxation_time_two_start returns a value < total,
            # it means convergence was detected at some point in the past.
            if tau < total:
                return total

    logger.warning(
        f'convergence_equilibrate: reached max_steps={max_steps} without convergence; '
        'proceeding anyway.'
    )
    return total


def plot_temperature_sweep(
    *,
    temperatures: np.ndarray,
    avg_m: Sequence[float],
    avg_e: Sequence[float],
    susc: Sequence[float],
    spec_h: Sequence[float],
    title: str,
    filename: str,
    directory: str,
    avg_m_err: np.ndarray | Sequence[float] | None = None,
    avg_e_err: np.ndarray | Sequence[float] | None = None,
    susc_err: np.ndarray | Sequence[float] | None = None,
    spec_h_err: np.ndarray | Sequence[float] | None = None,
    entropy: np.ndarray | Sequence[float] | None = None,
    entropy_err: np.ndarray | Sequence[float] | None = None,
    entropy_ci_low: np.ndarray | Sequence[float] | None = None,
    entropy_ci_high: np.ndarray | Sequence[float] | None = None,
    tau_int: np.ndarray | Sequence[float] | None = None,
    tau_int_ci_low: np.ndarray | Sequence[float] | None = None,
    tau_int_ci_high: np.ndarray | Sequence[float] | None = None,
    tau_unstable_flag: np.ndarray | Sequence[float] | None = None,
    low_effective_sample_flag: np.ndarray | Sequence[float] | None = None,
    diagnostics_note: str | None = None,
    run_metadata_note: str | None = None,
    quality_summary: Mapping[str, int | float] | None = None,
    transition_temperatures: dict[str, float] | None = None,
    transition_window: tuple[float, float] | None = None,
    annotate_peaks: bool = True,
    min_visible_rel_error: float = 0.01,
    mark_invalid_uncertainty: bool = True,
) -> None:
    """
    Generate and save a standardized temperature sweep plot.

    Displays magnetization, energy, susceptibility, and specific heat as
    functions of temperature in a dedicated thermodynamics figure. When
    *entropy* or *tau_int* are provided a companion diagnostics figure is
    saved alongside the main figure.

    Parameters
    ----------
        temperatures: Array of temperature values (x-axis).
        avg_m: Average absolute magnetization per temperature point.
        avg_e: Average energy per temperature point.
        susc: Magnetic susceptibility per temperature point.
        spec_h: Specific heat per temperature point.
        title: Figure-level suptitle string (e.g. '2D Ising Model: Temperature Sweep (L=50)').
        filename: Output filename passed to :func:`save_plot` (e.g. 'temperature_sweep.png').
        directory: Output directory passed to :func:`save_plot` (e.g. 'results/ising').
        entropy: Optional entropy per temperature point.
        entropy_err: Optional symmetric uncertainty for entropy.
        entropy_ci_low: Optional lower confidence band for entropy.
        entropy_ci_high: Optional upper confidence band for entropy.
        tau_int: Optional integrated autocorrelation time per temperature point.
            Reveals critical slowing down as a peak near T_c.
        tau_int_ci_low: Optional lower confidence band for tau_int.
        tau_int_ci_high: Optional upper confidence band for tau_int.
        tau_unstable_flag: Optional per-temperature mask flagging unstable
            tau_int intervals.
        low_effective_sample_flag: Optional per-temperature mask identifying
            points with low effective sample size.
        diagnostics_note: Optional text shown on diagnostics figure summarizing
            uncertainty quality metrics.
        run_metadata_note: Optional concise metadata line for the diagnostics
            header (for example lattice size, seeds, confidence level, method).
        quality_summary: Optional dictionary containing quality counts. Expected
            keys are ``total_points``, ``well_conditioned_count``,
            ``low_effective_count``, ``unstable_interval_count``, and
            ``undefined_count``.
        transition_temperatures: Optional mapping of transition-marker labels
            to temperatures rendered as vertical dashed guide lines.
        transition_window: Optional (low, high) temperature window highlighted
            with a light background band.
        annotate_peaks: If True, annotate peak temperatures for susceptibility,
            specific heat, and tau_int when present.
        min_visible_rel_error: Minimum relative error bar size used for visibility
            when finite uncertainties are extremely small.
        mark_invalid_uncertainty: If True, points with non-finite uncertainty are
            marked with ``x`` markers and a legend label.
    """
    temperatures_arr = np.asarray(temperatures, dtype=np.float64)
    use_extra = entropy is not None or tau_int is not None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    flat = axes.flatten()
    ax1, ax2, ax3, ax4 = flat

    fig.suptitle(title)

    def _visible_yerr(
        y: np.ndarray | Sequence[float],
        yerr: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        e_arr = np.asarray(yerr, dtype=np.float64)
        min_abs = np.maximum(1e-12, min_visible_rel_error * np.maximum(np.abs(y_arr), 1e-12))
        out = np.array(e_arr, copy=True)
        finite = np.isfinite(out)
        out[finite] = np.maximum(out[finite], min_abs[finite])
        return out

    def _mark_invalid(
        *,
        ax: plt.Axes,
        x: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[float],
        yerr: np.ndarray | Sequence[float],
        color: str,
        label: str,
    ) -> None:
        if not mark_invalid_uncertainty:
            return
        e_arr = np.asarray(yerr, dtype=np.float64)
        invalid = ~np.isfinite(e_arr)
        if not np.any(invalid):
            return
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        ax.plot(x_arr[invalid], y_arr[invalid], 'x', color=color, markersize=5, label=label)
        ax.legend(loc='best', fontsize=8)

    def _mark_low_effective_samples(
        *,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray | Sequence[float],
    ) -> None:
        if low_effective_sample_flag is None:
            return
        low_eff = np.asarray(low_effective_sample_flag, dtype=np.float64) > 0.0
        y_arr = np.asarray(y, dtype=np.float64)
        mask = low_eff & np.isfinite(x) & np.isfinite(y_arr)
        if not np.any(mask):
            return
        ax.plot(
            x[mask],
            y_arr[mask],
            'o',
            markerfacecolor='none',
            markeredgecolor='darkorange',
            markersize=7,
            linewidth=1.0,
            label='low effective samples',
        )
        ax.legend(loc='best', fontsize=8)

    def _annotate_peak(
        *,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray | Sequence[float],
        color: str,
        label: str,
        text_offset: tuple[float, float] = (6.0, 8.0),
    ) -> None:
        if not annotate_peaks:
            return
        y_arr = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y_arr)
        if not np.any(valid):
            return
        x_valid = x[valid]
        y_valid = y_arr[valid]
        idx = int(np.argmax(y_valid))
        t_peak = float(x_valid[idx])
        y_peak = float(y_valid[idx])
        ax.plot(
            [t_peak],
            [y_peak],
            marker='D',
            markerfacecolor='none',
            markeredgecolor=color,
            markersize=6,
            zorder=5,
        )
        ax.annotate(
            f'{label} peak: T={t_peak:.3g}',
            xy=(t_peak, y_peak),
            xytext=text_offset,
            textcoords='offset points',
            fontsize=8,
            color=color,
        )

    # Add global transition context guides to all thermodynamics axes.
    guides_added = False
    if transition_window is not None:
        lo, hi = transition_window
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            for ax in flat:
                ax.axvspan(lo, hi, color='gold', alpha=0.1)
            guides_added = True

    if transition_temperatures:
        guide_colors = ['gray', 'dimgray', 'slategray', 'black']
        for idx, (label, t_val) in enumerate(transition_temperatures.items()):
            if not np.isfinite(t_val):
                continue
            color = guide_colors[idx % len(guide_colors)]
            for ax in flat:
                ax.axvline(float(t_val), linestyle='--', linewidth=1.0, color=color, alpha=0.7)
            ax1.text(
                float(t_val),
                0.98,
                label,
                transform=ax1.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=7,
                color=color,
                bbox={
                    'boxstyle': 'round,pad=0.15',
                    'facecolor': 'white',
                    'alpha': 0.65,
                    'edgecolor': 'none',
                },
            )
            guides_added = True

    if guides_added:
        ax1.text(
            0.01,
            0.04,
            'Transition guides',
            transform=ax1.transAxes,
            fontsize=7,
            color='dimgray',
            ha='left',
            va='bottom',
        )

    if avg_m_err is None:
        ax1.plot(temperatures, avg_m, 'o-', markersize=4)
    else:
        ax1.errorbar(
            temperatures_arr,
            avg_m,
            yerr=_visible_yerr(avg_m, avg_m_err),
            fmt='o-',
            markersize=4,
            capsize=2,
        )
        _mark_invalid(
            ax=ax1,
            x=temperatures_arr,
            y=avg_m,
            yerr=avg_m_err,
            color='C0',
            label='uncertainty unavailable',
        )
    ax1.set_ylabel('Average Magnetization |M|')
    ax1.set_title('Magnetization')
    ax1.grid(True)
    _mark_low_effective_samples(ax=ax1, x=temperatures_arr, y=avg_m)

    if avg_e_err is None:
        ax2.plot(temperatures_arr, avg_e, 'o-', color='orange', markersize=4)
    else:
        ax2.errorbar(
            temperatures_arr,
            avg_e,
            yerr=_visible_yerr(avg_e, avg_e_err),
            fmt='o-',
            color='orange',
            markersize=4,
            capsize=2,
        )
        _mark_invalid(
            ax=ax2,
            x=temperatures_arr,
            y=avg_e,
            yerr=avg_e_err,
            color='orange',
            label='uncertainty unavailable',
        )
    ax2.set_ylabel('Average Energy')
    ax2.set_title('Energy')
    ax2.grid(True)
    _mark_low_effective_samples(ax=ax2, x=temperatures_arr, y=avg_e)

    if susc_err is None:
        ax3.plot(temperatures_arr, susc, 'o-', color='green', markersize=4)
    else:
        ax3.errorbar(
            temperatures_arr,
            susc,
            yerr=_visible_yerr(susc, susc_err),
            fmt='o-',
            color='green',
            markersize=4,
            capsize=2,
        )
        _mark_invalid(
            ax=ax3,
            x=temperatures_arr,
            y=susc,
            yerr=susc_err,
            color='green',
            label='uncertainty unavailable',
        )
    ax3.set_ylabel(r'Susceptibility $\chi$')
    ax3.set_title('Magnetic Susceptibility')
    ax3.grid(True)
    _annotate_peak(ax=ax3, x=temperatures_arr, y=susc, color='green', label='$\\chi$')
    _mark_low_effective_samples(ax=ax3, x=temperatures_arr, y=susc)

    if spec_h_err is None:
        ax4.plot(temperatures_arr, spec_h, 'o-', color='red', markersize=4)
    else:
        ax4.errorbar(
            temperatures_arr,
            spec_h,
            yerr=_visible_yerr(spec_h, spec_h_err),
            fmt='o-',
            color='red',
            markersize=4,
            capsize=2,
        )
        _mark_invalid(
            ax=ax4,
            x=temperatures_arr,
            y=spec_h,
            yerr=spec_h_err,
            color='red',
            label='uncertainty unavailable',
        )
    ax4.set_ylabel(r'Specific Heat $C_v$')
    ax4.set_title('Specific Heat')
    ax4.grid(True)
    _annotate_peak(ax=ax4, x=temperatures_arr, y=spec_h, color='red', label='$C_v$')
    _mark_low_effective_samples(ax=ax4, x=temperatures_arr, y=spec_h)

    for ax in flat:
        if ax.get_visible():
            ax.set_xlabel('Temperature (T)')

    save_plot(filename=filename, directory=directory)

    if not use_extra:
        return

    stem, ext = os.path.splitext(filename)
    diagnostics_filename = f'{stem}_diagnostics{ext or ".png"}'

    diag_fig, diag_axes = plt.subplots(1, 2, figsize=(12, 5.5))
    diag_flat = np.ravel(diag_axes)
    ax5, ax6 = diag_flat
    diag_fig.suptitle(f'{title} Diagnostics', y=0.99)
    if diagnostics_note:
        diag_fig.text(0.02, 0.94, diagnostics_note, ha='left', va='top', fontsize=8, wrap=True)
    if run_metadata_note:
        diag_fig.text(0.02, 0.915, run_metadata_note, ha='left', va='top', fontsize=7, wrap=True)
    if quality_summary is not None:
        total = int(quality_summary.get('total_points', 0))
        if total > 0:
            well = int(quality_summary.get('well_conditioned_count', 0))
            low_eff = int(quality_summary.get('low_effective_count', 0))
            unstable_count = int(quality_summary.get('unstable_interval_count', 0))
            undefined = int(quality_summary.get('undefined_count', 0))

            def _pct(v: int) -> float:
                return 100.0 * float(v) / float(total)

            quality_text = (
                f'quality: well={well}/{total} ({_pct(well):.1f}%), '
                f'low Neff={low_eff}/{total} ({_pct(low_eff):.1f}%), '
                f'unstable={unstable_count}/{total} ({_pct(unstable_count):.1f}%), '
                f'undefined={undefined}/{total} ({_pct(undefined):.1f}%)'
            )
            diag_fig.text(0.02, 0.89, quality_text, ha='left', va='top', fontsize=7, wrap=True)

    if entropy is not None:
        entropy_arr = np.asarray(entropy, dtype=np.float64)
        ax5.plot(temperatures_arr, entropy_arr, 'o-', color='purple', markersize=4)
        if entropy_ci_low is not None and entropy_ci_high is not None:
            ent_lo = np.asarray(entropy_ci_low, dtype=np.float64)
            ent_hi = np.asarray(entropy_ci_high, dtype=np.float64)
            ent_valid = np.isfinite(entropy_arr) & np.isfinite(ent_lo) & np.isfinite(ent_hi)
            if np.any(ent_valid):
                ax5.fill_between(
                    temperatures_arr[ent_valid],
                    ent_lo[ent_valid],
                    ent_hi[ent_valid],
                    color='purple',
                    alpha=0.18,
                    linewidth=0.0,
                )
            if mark_invalid_uncertainty:
                invalid_entropy = np.isfinite(entropy_arr) & ~ent_valid
                if np.any(invalid_entropy):
                    ax5.plot(
                        temperatures_arr[invalid_entropy],
                        entropy_arr[invalid_entropy],
                        'x',
                        color='purple',
                        markersize=5,
                        label='uncertainty unavailable',
                    )
                    ax5.legend(loc='best', fontsize=8)
        elif entropy_err is not None:
            ax5.errorbar(
                temperatures_arr,
                entropy_arr,
                yerr=_visible_yerr(entropy_arr, entropy_err),
                fmt='none',
                ecolor='purple',
                alpha=0.6,
                capsize=2,
            )
            _mark_invalid(
                ax=ax5,
                x=temperatures_arr,
                y=entropy_arr,
                yerr=entropy_err,
                color='purple',
                label='uncertainty unavailable',
            )
        ax5.set_ylabel(r'Entropy $S\,/\,k_B$')
        ax5.set_title('Entropy')
        ax5.grid(True)
        ax5.set_xlabel('Temperature (T)')
    else:
        ax5.set_visible(False)

    if tau_int is not None:
        tau_arr = np.asarray(tau_int, dtype=np.float64)
        ax6.plot(temperatures_arr, tau_arr, 'o-', color='saddlebrown', markersize=4)
        if tau_int_ci_low is not None and tau_int_ci_high is not None:
            tau_lo = np.asarray(tau_int_ci_low, dtype=np.float64)
            tau_hi = np.asarray(tau_int_ci_high, dtype=np.float64)
            band_valid = np.isfinite(tau_arr) & np.isfinite(tau_lo) & np.isfinite(tau_hi)
            if np.any(band_valid):
                ax6.fill_between(
                    temperatures_arr[band_valid],
                    tau_lo[band_valid],
                    tau_hi[band_valid],
                    color='saddlebrown',
                    alpha=0.22,
                    linewidth=0.0,
                )
            if mark_invalid_uncertainty:
                invalid_band = np.isfinite(tau_arr) & ~band_valid
                if np.any(invalid_band):
                    ax6.plot(
                        temperatures_arr[invalid_band],
                        tau_arr[invalid_band],
                        'x',
                        color='saddlebrown',
                        markersize=5,
                        label='uncertainty unavailable',
                    )
                    ax6.legend(loc='best', fontsize=8)
        elif mark_invalid_uncertainty:
            finite_tau = np.isfinite(tau_arr)
            if np.any(finite_tau):
                ax6.plot(
                    temperatures_arr[finite_tau],
                    tau_arr[finite_tau],
                    'x',
                    color='saddlebrown',
                    markersize=5,
                    label='uncertainty unavailable',
                )
                ax6.legend(loc='best', fontsize=8)
        if tau_unstable_flag is not None:
            unstable_mask = np.asarray(tau_unstable_flag, dtype=np.float64) > 0.0
            unstable_points = unstable_mask & np.isfinite(tau_arr)
            if np.any(unstable_points):
                ax6.plot(
                    temperatures_arr[unstable_points],
                    tau_arr[unstable_points],
                    'o',
                    markerfacecolor='none',
                    markeredgecolor='darkorange',
                    markersize=7,
                    linewidth=1.0,
                    label='unstable interval',
                )
                ax6.legend(loc='best', fontsize=8)
        ax6.set_ylabel(r'Integrated Autocorr. Time $\tau_\mathrm{int}$')
        ax6.set_title('Critical Slowing Down')
        ax6.grid(True)
        ax6.set_xlabel('Temperature (T)')
        _annotate_peak(
            ax=ax6,
            x=temperatures_arr,
            y=tau_arr,
            color='saddlebrown',
            label='$\\tau_\\mathrm{int}$',
            text_offset=(6.0, -12.0),
        )
        _mark_low_effective_samples(ax=ax6, x=temperatures_arr, y=tau_arr)
    else:
        ax6.set_visible(False)

    save_plot(filename=diagnostics_filename, directory=directory)


def plot_ordering_kinetics(
    *,
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

    Parameters
    ----------
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
        ax1.loglog(
            t[fit_mask],
            pre * t[fit_mask] ** exp,
            '--',
            color='tab:blue',
            label=f'Fit $R_{{sk}}$: $t^{{{exp:.2f}}}$',
        )

    ax1.loglog(t, R_xi, 's', label='$\\xi$ (Correlation length)')
    if exponents.get('xi') is not None:
        exp, pre = exponents['xi'], prefactors['xi']
        ax1.loglog(
            t[fit_mask],
            pre * t[fit_mask] ** exp,
            '--',
            color='tab:orange',
            label=f'Fit $\\xi$: $t^{{{exp:.2f}}}$',
        )

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
            ax2.loglog(
                t[fit_mask], pre * t[fit_mask] ** exp, 'k--', label=f'Fit: $t^{{{exp:.2f}}}$'
            )
        ax2.set_xlabel('Time t (Monte Carlo sweeps)')
        ax2.set_ylabel(third_metric_label)
        ax2.set_title(right_title)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend()
    else:
        ax2.axis('off')

    save_plot(filename=filename, directory=directory)


def plot_ordering_evolution(
    *,
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
    Generate and save a multi-row figure showing order evolution over time.

    Row 0: Spin configurations (binary for Ising, HSV for XY/Clock).
    Row 1: Circularly-averaged structure factor :math:`S(|k|)` or vorticity maps.
    Row 2: Radially averaged correlation functions G(r) with xi estimates.

    Parameters
    ----------
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
    fig, axes = plt.subplots(
        3, n_cols, figsize=(n_cols * 4, 12.5), gridspec_kw={'hspace': 0.45, 'wspace': 0.25}
    )
    fig.suptitle(title, fontsize=14, y=0.98)

    for col in range(n_cols):
        t = targets[col]
        spins = snapshots[col]

        # --- Row 0: Spin Configuration ---
        ax_spin = axes[0, col]
        if is_vector:
            angles = np.arctan2(spins[..., 1], spins[..., 0])
            im = ax_spin.imshow(angles, cmap='hsv', interpolation='none', vmin=-np.pi, vmax=np.pi)
            if col == n_cols - 1:
                plt.colorbar(im, ax=ax_spin, label='Phase (rad)', shrink=0.8)
        else:
            ax_spin.imshow(spins, cmap='binary', interpolation='none', vmin=-1, vmax=1)
        ax_spin.set_title(f't = {t} sweep{"s" if t != 1 else ""}', fontsize=12)
        ax_spin.axis('off')

        # --- Row 1: Mid-row (Vorticity or Sk) ---
        ax_mid = axes[1, col]
        if vorticity_data is not None:
            vort = vorticity_data[col]
            im_v = ax_mid.imshow(vort, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
            ax_mid.axis('off')
            v_count = int(np.count_nonzero(np.abs(vort) > 0.0))
            n_v = v_count / vort.size
            ax_mid.set_title(f'Vortices: {v_count}, $n_v={n_v:.3f}$', fontsize=10)
            if col == n_cols - 1:
                plt.colorbar(im_v, ax=ax_mid, ticks=[-1, 0, 1], label='Winding No.', shrink=0.8)
        else:
            from .physics_helpers import radial_average_sk

            k_vals, S_radial = radial_average_sk(spins=spins)
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
        ax_gr.axhline(
            inv_e, color='tab:red', linewidth=0.8, linestyle=':', alpha=0.7, label='$1/e$'
        )

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
            ax_gr.text(xi * 1.15, inv_e + 0.04, f'$\\xi = {xi:.1f}$', fontsize=9, color='tab:red')

    save_plot(filename=filename, directory=directory)
