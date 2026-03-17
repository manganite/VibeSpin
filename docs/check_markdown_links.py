from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = ROOT / 'docs' / 'source'

MARKDOWN_LINK_RE = re.compile(r'(?<!!)\[(?P<label>[^\]]+)\]\((?P<target>[^)\s]+)(?:\s+"[^"]*")?\)')
FENCED_BLOCK_RE = re.compile(r'```.*?```', re.DOTALL)
INLINE_CODE_RE = re.compile(r'`[^`]*`')
ALLOWED_SCHEMES = ('http://', 'https://', 'mailto:', 'ftp://')


def _strip_non_content(text: str) -> str:
    text = FENCED_BLOCK_RE.sub('', text)
    text = INLINE_CODE_RE.sub('', text)
    return text


def _normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith('<') and target.endswith('>'):
        target = target[1:-1]
    return target.split('#', 1)[0]


def _validate_link(source_file: Path, raw_target: str) -> str | None:
    target = _normalize_target(raw_target)
    if not target:
        return None
    if target.startswith(ALLOWED_SCHEMES):
        return None
    if target.startswith('/'):
        return f'absolute path link is not allowed in docs markdown: {raw_target}'

    resolved = (source_file.parent / target).absolute()

    # Allow missing extension for markdown/notebook targets (Sphinx/MyST style)
    if not resolved.exists():
        for ext in ('.md', '.ipynb'):
            if (resolved.with_suffix(ext)).exists():
                resolved = resolved.with_suffix(ext)
                break

    try:
        resolved.relative_to(DOCS_SOURCE.absolute())
    except ValueError:
        return (
            'relative link resolves outside docs/source; use a GitHub URL or move the file '
            f'under docs/source: {raw_target}'
        )

    if not resolved.exists():
        return f'relative link target does not exist under docs/source: {raw_target}'

    return None


def main() -> int:
    errors: list[str] = []

    for source_file in sorted(DOCS_SOURCE.rglob('*.md')):
        text = _strip_non_content(source_file.read_text(encoding='utf-8'))
        for match in MARKDOWN_LINK_RE.finditer(text):
            error = _validate_link(source_file, match.group('target'))
            if error is not None:
                rel_path = source_file.relative_to(ROOT)
                errors.append(f'{rel_path}: {error}')

    if errors:
        print('Documentation link validation failed:', file=sys.stderr)
        for error in errors:
            print(f'  - {error}', file=sys.stderr)
        return 1

    print('Documentation link validation passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
