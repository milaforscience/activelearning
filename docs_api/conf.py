"""Sphinx configuration for the standalone API reference."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "Multi-Fidelity Active Learning API Reference"
author = "milaforscience"
copyright = "2026, milaforscience"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = project
html_baseurl = "https://milaforscience.github.io/activelearning/reference/"
html_static_path: list[str] = []
html_theme_options = {
    "navigation_depth": 5,
    "collapse_navigation": False,
}
maximum_signature_line_length = 88
python_use_unqualified_type_names = True

autoapi_type = "python"
autoapi_dirs = [str(ROOT / "src")]
autoapi_root = "."
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autoapi_template_dir = str(ROOT / "docs_api" / "_templates")
autoapi_member_order = "bysource"
autoapi_python_class_content = "class"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

autodoc_inherit_docstrings = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True


def _strip_pydantic_boilerplate(
    _app: object,
    what: str,
    name: str,
    _obj: object,
    _options: object,
    lines: list[str],
) -> None:
    """Remove inherited Pydantic boilerplate from config model pages."""

    if what != "class" or not name.endswith("Config"):
        return

    joined = "\n".join(lines)
    if "A base class for creating Pydantic models." not in joined:
        return

    lines[:] = []


def setup(app: object) -> None:
    """Register Sphinx hooks for API rendering cleanup."""

    app.connect("autodoc-process-docstring", _strip_pydantic_boilerplate)
