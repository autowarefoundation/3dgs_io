"""Sphinx configuration for 3dgs_io documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "3dgs_io"
author = "Masaya Kataoka"
copyright = "2025, Masaya Kataoka"  # noqa: A001

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autodoc_member_order = "bysource"

napoleon_google_docstrings = True
napoleon_numpy_docstrings = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
