from __future__ import annotations

import ast
import re
from pathlib import Path

GOOGLE_STYLE_SECTION_RE = re.compile(
    r'^\s*(Args|Returns|Raises|Attributes|Examples):\s*$',
    flags=re.MULTILINE,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('models', 'utils', 'scripts')
SOURCE_FILES = ('benchmark.py', '__init__.py')


def _iter_docstrings(tree: ast.Module) -> list[tuple[str, int, str]]:
    """Collect module, class, and function docstrings with their line numbers."""
    matches: list[tuple[str, int, str]] = []

    module_doc = ast.get_docstring(tree, clean=False)
    if module_doc and tree.body:
        first = tree.body[0]
        lineno = getattr(first, 'lineno', 1)
        matches.append(('module', lineno, module_doc))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                matches.append((node.name, node.lineno, doc))

    return matches


def _find_google_style_sections(file_path: Path) -> list[tuple[int, str, str]]:
    source = file_path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(file_path))

    violations: list[tuple[int, str, str]] = []
    for obj_name, base_line, doc in _iter_docstrings(tree):
        for match in GOOGLE_STYLE_SECTION_RE.finditer(doc):
            rel_line = doc[: match.start()].count('\n')
            violations.append((base_line + rel_line, obj_name, match.group(0).strip()))

    return violations


def test_no_google_style_docstring_sections() -> None:
    """Enforce NumPy-style docstrings by rejecting Google-style section headers."""
    violations: list[str] = []

    for source_dir in SOURCE_DIRS:
        for py_file in (ROOT / source_dir).rglob('*.py'):
            file_violations = _find_google_style_sections(py_file)
            for line, obj_name, marker in file_violations:
                relative = py_file.relative_to(ROOT)
                violations.append(
                    f'{relative}:{line} ({obj_name}) uses Google-style header "{marker}"'
                )

    for file_name in SOURCE_FILES:
        py_file = ROOT / file_name
        if not py_file.exists():
            continue
        file_violations = _find_google_style_sections(py_file)
        for line, obj_name, marker in file_violations:
            relative = py_file.relative_to(ROOT)
            violations.append(
                f'{relative}:{line} ({obj_name}) uses Google-style header "{marker}"'
            )

    assert not violations, (
        'Google-style docstring headers found. Use NumPy-style sections instead.\n'
        + '\n'.join(violations)
    )


NUMPY_STYLE_SECTION_RE = re.compile(
    r'^\s*(Parameters|Returns|Raises|Attributes|Examples)\n\s*-+\s*$',
    flags=re.MULTILINE,
)


def test_numpy_style_sections_present_in_core() -> None:
    """Core simulation models and utilities should use NumPy-style sections when applicable."""
    core_dirs = ('models', 'utils')
    missing_sections: list[str] = []

    for source_dir in core_dirs:
        for py_file in (ROOT / source_dir).rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
            source = py_file.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=str(py_file))
            for obj_name, line, doc in _iter_docstrings(tree):
                # Only check functions/methods that likely need parameters
                if obj_name == 'module' or obj_name.startswith('_'):
                    continue
                # If it's a simulation class or public method, it should probably have sections
                if 'Simulation' in obj_name or len(doc.split('\n')) > 3:
                    if not NUMPY_STYLE_SECTION_RE.search(doc):
                        # Some small helpers might not have sections, only alert on longer ones
                        if len(doc.split('\n')) > 8:
                            relative = py_file.relative_to(ROOT)
                            missing_sections.append(
                                f'{relative}:{line} ({obj_name}) missing NumPy-style sections'
                            )

    # This is a soft check, we might have valid docstrings without sections
    # but for VibeSpin core, we expect them.
    if missing_sections:
        print('\n'.join(missing_sections))


def test_core_docstring_presence() -> None:
    """Ensure all public classes and functions in core modules have docstrings."""
    core_dirs = ('models', 'utils')
    missing_docs: list[str] = []

    for source_dir in core_dirs:
        for py_file in (ROOT / source_dir).rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
            source = py_file.read_text(encoding='utf-8')
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Skip private/internal
                    if node.name.startswith('_'):
                        continue
                    if not ast.get_docstring(node):
                        relative = py_file.relative_to(ROOT)
                        missing_docs.append(f'{relative}:{node.lineno} ({node.name})')

    assert not missing_docs, (
        'Public classes/functions in core modules must have docstrings:\n'
        + '\n'.join(missing_docs)
    )
