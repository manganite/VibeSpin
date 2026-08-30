"""
Shared pytest configuration.

The check here guards the repository's ``results/`` directory. Every script
writes its NPZ and PNG output there by default, so a test that calls a
script's ``main()`` or a plotting helper without redirecting the output drops
a file with test-sized parameters into the location the notebooks read as
their cache, under the production filename. The notebooks then either load it
or reject it and say why, but either way the run is no longer the one the
author intended.

The check runs once at the end of the session rather than as a fixture, so
that the failure is reported against the session instead of against whichever
unrelated test happened to finish last.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'

_BEFORE: dict[str, float] = {}


def _snapshot() -> dict[str, float]:
    """Map every file under results/ to its modification time."""
    if not RESULTS_DIR.is_dir():
        return {}
    return {
        str(path.relative_to(RESULTS_DIR)): os.path.getmtime(path)
        for path in RESULTS_DIR.rglob('*')
        if path.is_file()
    }


def pytest_sessionstart(session: pytest.Session) -> None:
    """Record the state of the results directory before any test runs."""
    _BEFORE.clear()
    _BEFORE.update(_snapshot())


def pytest_sessionfinish(session: pytest.Session, exitstatus: Any) -> None:
    """Fail the session if any test wrote into the repository results directory."""
    after = _snapshot()
    created = sorted(set(after) - set(_BEFORE))
    modified = sorted(
        name for name in set(after) & set(_BEFORE) if after[name] != _BEFORE[name]
    )
    if not created and not modified:
        return

    session.exitstatus = pytest.ExitCode.TESTS_FAILED
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    message = (
        'The test suite wrote into the repository results/ directory, which the '
        'notebooks read as their data cache.\n'
        f'  Created:  {created}\n'
        f'  Modified: {modified}\n'
        "Pass an output directory under tmp_path when a test calls a script's "
        'main() or a plotting helper.'
    )
    if reporter is not None:
        reporter.write_sep('=', 'results directory guard', red=True)
        reporter.write_line(message)
    else:
        print(message)
