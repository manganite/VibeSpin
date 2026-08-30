"""
Style guards for notebook markdown.

AGENTS.md applies the project's writing rules to notebook markdown as well as
to documentation, and requires every code cell to be introduced by prose. The
notebooks drifted from all three rules while nothing read them, so the rules
that can be checked mechanically are checked here.

What is deliberately not checked: AGENTS.md permits a bullet list when it is
procedural, and no test can decide whether a numbered list describes a
procedure or is a roadmap that should have been a paragraph. Ordered lists are
therefore left to review, while unordered ones, which are almost never
procedural, are rejected outright.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / 'notebooks'

#: Unordered list items at the start of a line, the form AGENTS.md rules out.
_UNORDERED_ITEM = re.compile(r'^\s*[-*+]\s+\S')

#: Fenced code inside markdown may legitimately contain anything.
_FENCE = re.compile(r'^\s*```')


def _notebooks() -> list[Path]:
    paths = sorted(NOTEBOOK_DIR.glob('*.ipynb'))
    assert paths, f'No notebooks found under {NOTEBOOK_DIR}'
    return paths


def _markdown_cells(path: Path) -> list[tuple[int, str]]:
    cells = json.loads(path.read_text(encoding='utf-8'))['cells']
    return [
        (index, ''.join(cell['source']))
        for index, cell in enumerate(cells)
        if cell['cell_type'] == 'markdown'
    ]


def _prose_lines(source: str) -> list[tuple[int, str]]:
    """Yield numbered lines of a markdown cell, skipping fenced code blocks."""
    lines: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(source.split('\n'), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append((number, line))
    return lines


@pytest.mark.parametrize('path', _notebooks(), ids=lambda p: p.name)
def test_no_em_dash_in_notebook_prose(path: Path) -> None:
    """AGENTS.md forbids the em dash character in user-facing prose."""
    offenders = [
        f'{path.name} cell {index} line {number}: {line.strip()[:90]}'
        for index, source in _markdown_cells(path)
        for number, line in _prose_lines(source)
        if '—' in line
    ]
    assert not offenders, 'Em dash in notebook markdown:\n' + '\n'.join(offenders)


@pytest.mark.parametrize('path', _notebooks(), ids=lambda p: p.name)
def test_no_unordered_bullet_lists_in_notebook_prose(path: Path) -> None:
    """AGENTS.md asks for flowing prose rather than bullet points."""
    offenders = [
        f'{path.name} cell {index} line {number}: {line.strip()[:90]}'
        for index, source in _markdown_cells(path)
        for number, line in _prose_lines(source)
        if _UNORDERED_ITEM.match(line)
    ]
    assert not offenders, (
        'Unordered bullet list in notebook markdown; rewrite as prose:\n'
        + '\n'.join(offenders)
    )


@pytest.mark.parametrize('path', _notebooks(), ids=lambda p: p.name)
def test_every_code_cell_is_introduced_by_markdown(path: Path) -> None:
    """AGENTS.md requires a markdown cell immediately before every code cell."""
    cells = json.loads(path.read_text(encoding='utf-8'))['cells']
    offenders = [
        index
        for index, cell in enumerate(cells)
        if cell['cell_type'] == 'code'
        and (index == 0 or cells[index - 1]['cell_type'] != 'markdown')
    ]
    assert not offenders, (
        f'{path.name}: code cells {offenders} are not preceded by a markdown cell. '
        'A shared intro is allowed only when it names each code cell it covers, '
        'so add the missing prose rather than relying on proximity.'
    )
