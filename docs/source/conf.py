# -- Project information -----------------------------------------------------
project = 'scivianna'
copyright = '2026, T. Moulignier'
author = 'T. Moulignier'
release = '0.5.0'

# -- General configuration ---------------------------------------------------
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_nb",
]

# nbsphinx configuration
nbsphinx_execute = "never"
nb_execution_mode = "off"

# myst_nb configuration
myst_nb_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

autoapi_type = "python"
import os, sys
sys.path.insert(0, os.path.abspath("../../src"))
autoapi_dirs = ["../../src"]