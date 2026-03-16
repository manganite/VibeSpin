"""Regenerate the committed Sphinx API reference pages."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / 'docs' / 'source' / 'api'

PACKAGE_TITLES = {
    'models': 'vibespin.models package',
    'scripts': 'vibespin.scripts package',
    'scripts.clock': 'vibespin.scripts.clock package',
    'scripts.ising': 'vibespin.scripts.ising package',
    'scripts.xy': 'vibespin.scripts.xy package',
    'utils': 'vibespin.utils package',
}

TOCTREE_CHILDREN = {
    'vibespin': ['vibespin.models', 'vibespin.scripts', 'vibespin.utils'],
    'scripts': ['vibespin.scripts.clock', 'vibespin.scripts.ising', 'vibespin.scripts.xy'],
}

def _title(text: str) -> str:
    return f'{text}\n{"=" * len(text)}\n\n'


def _section(title: str, underline: str = '-') -> str:
    return f'{title}\n{underline * len(title)}\n\n'


def _escape_module_name(name: str) -> str:
    return name.replace('_', '\\_')


def _iter_modules(package: str) -> list[str]:
    package_dir = ROOT / package.replace('.', '/')
    return sorted(
        path.stem
        for path in package_dir.glob('*.py')
        if path.name != '__init__.py'
    )


def _module_section(module_path: str) -> str:
    heading = f'{_escape_module_name(module_path)} module'
    return (
        _section(heading)
        + f'.. automodule:: {module_path}\n'
        + '   :members:\n'
        + '   :show-inheritance:\n'
        + '   :undoc-members:\n\n'
    )


def _render_modules_index() -> str:
    return (
        _title('vibespin')
        + '.. toctree::\n'
        + '   :maxdepth: 4\n\n'
        + '   vibespin\n'
    )


def _render_root_package() -> str:
    content = _title('vibespin package')
    content += _section('Subpackages')
    content += '.. toctree::\n'
    content += '   :maxdepth: 4\n\n'
    for child in TOCTREE_CHILDREN['vibespin']:
        content += f'   {child}\n'
    content += '\n'
    return content


def _render_package(package: str) -> str:
    content = _title(PACKAGE_TITLES[package])
    children = TOCTREE_CHILDREN.get(package)
    if children:
        content += _section('Subpackages')
        content += '.. toctree::\n'
        content += '   :maxdepth: 4\n\n'
        for child in children:
            content += f'   {child}\n'
        content += '\n'

    modules = _iter_modules(package)
    if modules:
        content += _section('Submodules')
        for module_name in modules:
            content += _module_section(f'{package}.{module_name}')

    content += _section('Module contents')
    content += f'.. automodule:: {package}\n'
    content += '   :members:\n'
    content += '   :show-inheritance:\n'
    content += '   :undoc-members:\n'
    return content


def main() -> None:
    API_DIR.mkdir(parents=True, exist_ok=True)

    rendered = {
        'modules.rst': _render_modules_index(),
        'vibespin.rst': _render_root_package(),
        'vibespin.models.rst': _render_package('models'),
        'vibespin.scripts.rst': _render_package('scripts'),
        'vibespin.scripts.clock.rst': _render_package('scripts.clock'),
        'vibespin.scripts.ising.rst': _render_package('scripts.ising'),
        'vibespin.scripts.xy.rst': _render_package('scripts.xy'),
        'vibespin.utils.rst': _render_package('utils'),
    }

    existing_files = {path.name for path in API_DIR.glob('*.rst')}
    expected_files = set(rendered)
    for stale_file in sorted(existing_files - expected_files):
        (API_DIR / stale_file).unlink()

    for file_name, content in rendered.items():
        (API_DIR / file_name).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
