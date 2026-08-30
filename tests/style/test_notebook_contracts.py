"""Static contract checks for the Jupyter notebooks.

Ruff lints notebook cells, but it cannot resolve imports across modules or
know which project functions are keyword-only. Both gaps have bitten this
repository: a module split left five notebooks importing a deleted module,
and a keyword-only refactor left positional call sites behind. Executing
every notebook would catch these but is far too slow for the test suite, so
these checks close the gap statically.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((ROOT / 'notebooks').glob('*.ipynb'))
PROJECT_MODULES = (
    'utils.statistics', 'utils.observables', 'utils.equilibration', 'utils.plotting',
    'utils.kinetics_helpers', 'utils.evolution_helpers', 'utils.sweep_helpers',
    'utils.system', 'utils.exceptions',
)


def _code_cells(path: Path) -> list[tuple[int, str]]:
    """Return (index, source) for each code cell, with Jupyter magics blanked."""
    nb = json.loads(path.read_text(encoding='utf-8'))
    cells = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        src = '\n'.join('' if line.lstrip()[:1] in '%!' else line for line in src.split('\n'))
        cells.append((i, src))
    return cells


def _keyword_only_functions() -> dict[str, list[str]]:
    """Map function name to its keyword-only parameters, for fully keyword-only callables."""
    out: dict[str, list[str]] = {}
    for mod_name in PROJECT_MODULES:
        mod = importlib.import_module(mod_name)
        for name in dir(mod):
            fn = getattr(mod, name)
            if not callable(fn) or getattr(fn, '__module__', '') != mod_name:
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            kw_only = [p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY]
            positional = [
                p.name for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if kw_only and not positional:
                out[name] = kw_only
    return out


def _import_aliases(src: str) -> dict[str, str]:
    """Map local name to canonical name for `from utils... import x as y`."""
    aliases: dict[str, str] = {}
    pattern = r'from\s+(?:utils|models)[\w.]*\s+import\s+([^\n(]+|\([^)]*\))'
    for match in re.finditer(pattern, src):
        for part in match.group(1).strip().strip('()').split(','):
            part = part.strip()
            if not part:
                continue
            if ' as ' in part:
                original, alias = (x.strip() for x in part.split(' as '))
                aliases[alias] = original
            else:
                aliases[part] = part
    return aliases


def test_notebook_project_imports_resolve() -> None:
    """Every `from utils/models ... import name` in a notebook must resolve."""
    failures: list[str] = []
    for path in NOTEBOOKS:
        for index, src in _code_cells(path):
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                failures.append(f'{path.name} [cell {index}]: syntax error: {exc}')
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                if not node.module.startswith(('utils', 'models')):
                    continue
                try:
                    mod = importlib.import_module(node.module)
                except ImportError as exc:
                    failures.append(f'{path.name} [cell {index}]: {node.module!r}: {exc}')
                    continue
                for alias in node.names:
                    if not hasattr(mod, alias.name):
                        failures.append(
                            f'{path.name} [cell {index}]: '
                            f'{node.module}.{alias.name} does not exist'
                        )
    assert not failures, 'Unresolvable notebook imports:\n' + '\n'.join(failures)


def test_notebooks_call_keyword_only_functions_by_keyword() -> None:
    """Notebooks must not pass positional arguments to keyword-only helpers."""
    kw_only = _keyword_only_functions()
    failures: list[str] = []
    for path in NOTEBOOKS:
        for index, src in _code_cells(path):
            aliases = _import_aliases(src)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue  # reported by the import test
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                if not node.args:
                    continue
                canonical = aliases.get(node.func.id, node.func.id)
                if canonical in kw_only:
                    failures.append(
                        f'{path.name} [cell {index}]: {node.func.id}() called with '
                        f'{len(node.args)} positional argument(s); '
                        f'keyword-only parameters are {kw_only[canonical]}'
                    )
    assert not failures, 'Positional calls to keyword-only helpers:\n' + '\n'.join(failures)
