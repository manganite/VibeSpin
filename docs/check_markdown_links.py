"""Validate internal markdown links (and optionally external URLs) in docs/source.

Checks every markdown link in ``docs/source/*.md`` for three failure modes:
file targets that do not exist under ``docs/source``, absolute-path links,
and intra-document anchors (``#fragment``, alone or after a file target)
that do not correspond to any heading slug in the target document.

External ``http(s)`` URLs are skipped by default so the pre-push hook stays
offline and deterministic; pass ``--external`` to also probe each external
URL once over the network (best-effort HEAD/GET with a timeout), as required
by the reference-accessibility policy in AGENTS.md section 7.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = ROOT / 'docs' / 'source'

MARKDOWN_LINK_RE = re.compile(r'(?<!!)\[(?P<label>[^\]]+)\]\((?P<target>[^)\s]+)(?:\s+"[^"]*")?\)')
FENCED_BLOCK_RE = re.compile(r'```.*?```', re.DOTALL)
INLINE_CODE_RE = re.compile(r'`[^`]*`')
HEADING_RE = re.compile(r'^#{1,6}\s+(?P<text>.+?)\s*#*\s*$', re.MULTILINE)
EXTERNAL_SCHEMES = ('http://', 'https://')
SKIPPED_SCHEMES = ('mailto:', 'ftp://')

_heading_slug_cache: dict[Path, set[str]] = {}


def _strip_non_content(text: str) -> str:
    text = FENCED_BLOCK_RE.sub('', text)
    text = INLINE_CODE_RE.sub('', text)
    return text


def _split_target(raw_target: str) -> tuple[str, str | None]:
    """Return ``(path_part, fragment_or_None)`` for a raw link target."""
    target = raw_target.strip()
    if target.startswith('<') and target.endswith('>'):
        target = target[1:-1]
    if '#' in target:
        path_part, fragment = target.split('#', 1)
        return path_part, fragment
    return target, None


def _slugify(heading: str) -> str:
    """Approximate the GitHub/MyST heading slug for anchor matching.

    Lowercases, strips markdown emphasis and inline-code markers, drops
    punctuation, and converts spaces to hyphens. This mirrors how both
    GitHub and myst-parser (``myst_heading_anchors``) derive heading ids
    closely enough for validation purposes.
    """
    text = heading.strip().lower()
    text = re.sub(r'[`*_]', '', text)
    # Drop markdown links inside headings, keeping the label.
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text.strip())
    return text


def _heading_slugs(markdown_file: Path) -> set[str]:
    """Return the set of heading anchor slugs defined in a markdown file."""
    resolved = markdown_file.resolve()
    if resolved not in _heading_slug_cache:
        text = FENCED_BLOCK_RE.sub('', resolved.read_text(encoding='utf-8'))
        _heading_slug_cache[resolved] = {
            _slugify(match.group('text')) for match in HEADING_RE.finditer(text)
        }
    return _heading_slug_cache[resolved]


def _check_external_url(url: str, *, timeout: float) -> str | None:
    """Probe an external URL; return an error message when unreachable."""
    request = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'vibespin-docs'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        # Some hosts reject HEAD; retry once with GET before reporting.
        if exc.code in (403, 405):
            get_request = urllib.request.Request(url, headers={'User-Agent': 'vibespin-docs'})
            try:
                with urllib.request.urlopen(get_request, timeout=timeout) as response:
                    status = int(response.status)
            except (urllib.error.URLError, OSError) as get_exc:
                return f'external link unreachable ({get_exc}): {url}'
        else:
            return f'external link returned HTTP {exc.code}: {url}'
    except (urllib.error.URLError, OSError) as exc:
        return f'external link unreachable ({exc}): {url}'
    if status >= 400:
        return f'external link returned HTTP {status}: {url}'
    return None


def _validate_link(
    source_file: Path,
    raw_target: str,
    *,
    check_external: bool,
    external_timeout: float,
) -> str | None:
    path_part, fragment = _split_target(raw_target)

    if path_part.startswith(EXTERNAL_SCHEMES):
        if check_external:
            return _check_external_url(path_part, timeout=external_timeout)
        return None
    if path_part.startswith(SKIPPED_SCHEMES):
        return None
    if path_part.startswith('/'):
        return f'absolute path link is not allowed in docs markdown: {raw_target}'

    # Pure-fragment link: anchor must exist in the source document itself.
    if not path_part:
        if fragment and _slugify(fragment) not in _heading_slugs(source_file):
            return f'anchor does not match any heading in this document: #{fragment}'
        return None

    resolved = (source_file.parent / path_part).absolute()

    # Allow missing extension for markdown/notebook targets (Sphinx/MyST style)
    # OR correctly handle existing .md/.ipynb extensions
    if not resolved.exists():
        for ext in ('.md', '.ipynb'):
            if not path_part.endswith(ext):
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

    # Cross-file anchors are only validated for markdown targets; notebook
    # anchors are rendered by nbsphinx and follow different id rules.
    if fragment and resolved.suffix == '.md':
        if _slugify(fragment) not in _heading_slugs(resolved):
            return (
                f'anchor does not match any heading in {path_part}: #{fragment}'
            )

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate markdown links in docs/source.')
    parser.add_argument(
        '--external',
        action='store_true',
        help='Also probe external http(s) URLs over the network (off by default '
             'so the pre-push hook stays offline).',
    )
    parser.add_argument(
        '--external-timeout',
        type=float,
        default=10.0,
        help='Per-request timeout in seconds for --external probes.',
    )
    args = parser.parse_args()

    errors: list[str] = []

    for source_file in sorted(DOCS_SOURCE.rglob('*.md')):
        text = _strip_non_content(source_file.read_text(encoding='utf-8'))
        for match in MARKDOWN_LINK_RE.finditer(text):
            error = _validate_link(
                source_file,
                match.group('target'),
                check_external=args.external,
                external_timeout=args.external_timeout,
            )
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
