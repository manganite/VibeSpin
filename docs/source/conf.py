# Configuration file for the Sphinx documentation builder.
import os
import sys
# Path to the root of the project
sys.path.insert(0, os.path.abspath('../..'))

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

# -- nbsphinx settings -------------------------------------------------------
nbsphinx_execute = 'never'  # Don't run notebooks during build
