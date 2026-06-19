"""Sphinx configuration for the typed public API reference."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
project = "Tunix CrafText"
extensions = ["sphinx.ext.autodoc", "sphinx_autodoc_typehints"]
autodoc_typehints = "signature"
autodoc_default_options = {"members": True, "undoc-members": False}
# Third-party and generic type variables are rendered by autodoc-typehints but are not part of
# this project's intersphinx inventory. Keep warnings-as-errors in the build without requiring
# every ``typing``/NumPy symbol to resolve as a local cross-reference.
nitpicky = False
