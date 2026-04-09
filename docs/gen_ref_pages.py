"""Generate the API reference pages automatically from the source tree.

Implements the mkdocstrings "Automatic code reference pages" recipe:
https://mkdocstrings.github.io/recipes/

At MkDocs build time (via the ``gen-files`` plugin) this script:

1. Recursively walks ``src/activelearning`` and creates one virtual Markdown
   page per public Python module under the ``reference/`` virtual directory.
2. Writes a ``reference/summary.md`` that lists every documented module
   grouped by top-level sub-package, auto-updating whenever modules are added.
3. Writes a ``reference/SUMMARY.md`` that ``mkdocs-literate-nav`` uses to
   build the full navigation tree automatically — no ``nav:`` entries need
   manual updates when new modules are added.

Routing rules
-------------
* ``__main__.py``  → skipped entirely (entry-point, no public API).
* ``__init__.py``  → maps to an ``index.md`` section-index page
                     (``mkdocs-section-index`` makes it clickable in the nav).
* All other files  → map to a same-name ``.md`` page.
"""

from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SRC = Path("src")  # root of the Python source tree
REFERENCE = Path("reference")  # virtual output directory managed by gen-files

# Front matter added to every generated page to suppress search indexing.
# API reference pages flood search results with symbol names, making it harder
# to find conceptual documentation via the search bar.
_FRONT_MATTER = "---\nsearch:\n  exclude: true\n---\n\n"

# ---------------------------------------------------------------------------
# Build virtual pages and accumulate nav + summary data
# ---------------------------------------------------------------------------

nav = mkdocs_gen_files.Nav()

# Collect (module_identifier, relative_doc_path) grouped by top-level
# sub-package (e.g. "acquisition", "budget") for the summary page.
# Modules that live directly under the root package use the key "".
summary_sections: dict[str, list[tuple[str, str]]] = defaultdict(list)

for py_path in sorted(SRC.rglob("*.py")):
    module_path = py_path.relative_to(SRC).with_suffix("")
    parts = tuple(module_path.parts)

    # ------------------------------------------------------------------
    # Routing rules
    # ------------------------------------------------------------------

    if parts[-1] == "__main__":
        # Entry-point files expose no public API worth documenting.
        continue

    if parts[-1] == "__init__":
        # Package directory → section-index page so the folder heading
        # in the nav is itself a clickable link to the package docstring.
        parts = parts[:-1]
        if not parts:
            # Skip a bare src-root __init__ with no meaningful public API.
            continue
        doc_path = REFERENCE / Path(*parts) / "index.md"
    else:
        doc_path = REFERENCE / Path(*parts).with_suffix(".md")

    # ------------------------------------------------------------------
    # Virtual page: a single mkdocstrings injection directive.
    # Global options (show_source, heading_level, …) are set in mkdocs.yml
    # so no per-page overrides are needed here.
    # ------------------------------------------------------------------

    module_identifier = ".".join(parts)
    rel_doc_path = doc_path.relative_to(REFERENCE).as_posix()

    nav[parts] = rel_doc_path

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(_FRONT_MATTER + f"::: {module_identifier}\n")

    # "Edit on GitHub" links point back to the real Python source file.
    mkdocs_gen_files.set_edit_path(doc_path, py_path)

    # Group by top-level sub-package for the summary page.
    # parts[0] is always "activelearning"; parts[1] (if present) is the sub-package.
    section_key = parts[1] if len(parts) > 1 else ""
    summary_sections[section_key].append((module_identifier, rel_doc_path))

# ---------------------------------------------------------------------------
# reference/summary.md — auto-generated module listing
# ---------------------------------------------------------------------------

with mkdocs_gen_files.open(REFERENCE / "summary.md", "w") as summary_page:
    summary_page.write(_FRONT_MATTER)
    summary_page.write("# Module Summary\n\n")
    summary_page.write(
        "Complete listing of every documented module in the `activelearning` package, "
        "auto-generated from the source tree.\n\n"
    )

    # Top-level (root package) modules first, then sub-packages alphabetically.
    ordered_keys = ([""] if "" in summary_sections else []) + sorted(
        k for k in summary_sections if k != ""
    )

    for key in ordered_keys:
        # Section heading: root modules → "activelearning", others → "activelearning.X"
        heading = "activelearning" if key == "" else f"activelearning.{key}"
        summary_page.write(f"## `{heading}`\n\n")
        for module_id, rel_path in sorted(summary_sections[key]):
            summary_page.write(f"- [`{module_id}`]({rel_path})\n")
        summary_page.write("\n")

# ---------------------------------------------------------------------------
# reference/SUMMARY.md — drives mkdocs-literate-nav
# ---------------------------------------------------------------------------

with mkdocs_gen_files.open(REFERENCE / "SUMMARY.md", "w") as nav_file:
    # Static pages that live outside the generated tree come first.
    nav_file.write("* [Overview](../api/index.md)\n")
    nav_file.write("* [Module Summary](summary.md)\n")
    # All auto-generated module pages follow, properly nested.
    nav_file.writelines(nav.build_literate_nav())
