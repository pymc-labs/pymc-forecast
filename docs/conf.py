"""Sphinx configuration for the pymc_forecast documentation."""

from pymc_forecast import __version__

project = "pymc_forecast"
copyright = "2026, PyMC Labs"
author = "PyMC Labs"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute", "**.ipynb_checkpoints"]

# -- Autodoc / napoleon ------------------------------------------------------

autodoc_typehints = "none"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
}

# -- MyST / notebooks --------------------------------------------------------

# Notebooks are committed fully executed; render their stored outputs.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist", "substitution"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "pymc_forecast"
html_theme_options = {
    "github_url": "https://github.com/pymc-labs/pymc_forecast",
    "use_edit_page_button": False,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
}
html_context = {
    "github_user": "pymc-labs",
    "github_repo": "pymc_forecast",
    "github_version": "main",
    "doc_path": "docs",
}
