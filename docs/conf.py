"""Sphinx configuration for scModelForge documentation."""

project = "scModelForge"
copyright = "2026, scModelForge Contributors"  # noqa: A001
author = "scModelForge Contributors"

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

# MyST settings
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Autodoc2 settings
autodoc2_packages = ["../src/scmodelforge"]
autodoc2_render_plugin = "myst"
autodoc2_hidden_objects = ["dunder", "private"]
autodoc2_module_all_regexes = []

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

# Theme
html_theme = "sphinx_book_theme"
html_title = "scModelForge"
html_theme_options = {
    "repository_url": "https://github.com/EhsanRS/scModelForge",
    "use_repository_button": True,
    "use_issues_button": True,
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
