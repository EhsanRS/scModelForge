"""Sphinx configuration for scModelForge documentation."""

project = "scModelForge"
copyright = "2026, scModelForge Contributors"  # noqa: A001
author = "scModelForge Contributors"

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx_copybutton",
]

# MyST settings
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Autodoc2 settings
autodoc2_packages = ["../src/scmodelforge"]

# Theme
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/EhsanRS/scModelForge",
    "use_repository_button": True,
    "use_issues_button": True,
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
