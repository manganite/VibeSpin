# Configuration file for the Sphinx documentation builder.
import os
import shutil
import sys

# Path to the root of the project
sys.path.insert(0, os.path.abspath('../..'))


def _ensure_pandoc_on_path() -> None:
    """Add a bundled pandoc binary to PATH when no system pandoc is available."""
    if shutil.which('pandoc') is not None:
        return

    try:
        import pypandoc
    except ImportError:
        return

    try:
        pandoc_path = pypandoc.get_pandoc_path()
    except OSError:
        return

    pandoc_dir = os.path.dirname(pandoc_path)
    os.environ['PATH'] = pandoc_dir + os.pathsep + os.environ.get('PATH', '')


_ensure_pandoc_on_path()

# -- Project information -----------------------------------------------------
project = 'VibeSpin'
copyright = '2026, Thomas Lottermoser'
author = 'Thomas Lottermoser'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- MyST settings -----------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3
# Note: 'myst.xref_missing' is deliberately NOT suppressed; with -W this
# makes dead internal anchors (e.g. broken citation links) fail the build.
suppress_warnings = [
    # CI docs jobs may omit ipywidgets; keep notebook docs buildable with -W.
    'nbsphinx.ipywidgets',
]

# -- nbsphinx settings -------------------------------------------------------

nbsphinx_execute = 'never'  # Don't run notebooks during build
