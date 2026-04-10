"""Generate checked-in API reference pages from the source tree.

The Zensical migration keeps the ``mkdocstrings`` directives, but it can no
longer rely on MkDocs build-time plugins to create virtual files and navigation.
Instead, this script writes the generated Markdown pages directly under
``docs/reference/`` so they can be committed and built by Zensical.

Routing rules
-------------
* ``__main__.py``  → skipped entirely (entry-point, no public API).
* ``__init__.py``  → maps to an ``overview.md`` package overview page.
* All other files  → map to a same-name ``.md`` page.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import shutil

SRC = Path("src")
REFERENCE = Path("docs/reference")
_FRONT_MATTER = "---\nsearch:\n  exclude: true\n---\n\n"


def _defined_public_members(py_path: Path) -> list[str]:
    """Return public top-level members defined directly in a Python source file."""
    module_ast = ast.parse(py_path.read_text(encoding="utf-8"))
    members: list[str] = []

    for node in module_ast.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                members.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    members.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                members.append(target.id)

    return members


def _mkdocstrings_block(module_identifier: str, members: list[str]) -> str:
    """Build a mkdocstrings directive restricted to file-local public members."""
    block = _FRONT_MATTER + f"::: {module_identifier}\n"
    if members:
        block += "    options:\n"
        block += "      members:\n"
        for member in members:
            block += f"        - {member}\n"
    return block


def _iter_reference_modules() -> tuple[
    list[tuple[str, Path, list[str]]], set[tuple[str, ...]]
]:
    """Return module pages to generate plus discovered package paths."""
    reference_modules: list[tuple[str, Path, list[str]]] = []
    package_paths: set[tuple[str, ...]] = set()

    for py_path in sorted(SRC.rglob("*.py")):
        module_path = py_path.relative_to(SRC).with_suffix("")
        parts = tuple(module_path.parts)

        if parts[-1] == "__main__":
            continue

        if parts[-1] == "__init__":
            package_parts = parts[:-1]
            if package_parts:
                package_paths.add(package_parts)
            continue

        doc_path = REFERENCE / Path(*parts).with_suffix(".md")
        reference_modules.append(
            (".".join(parts), doc_path, _defined_public_members(py_path))
        )

    return reference_modules, package_paths


def _write_module_pages(
    reference_modules: list[tuple[str, Path, list[str]]],
) -> tuple[
    dict[str, list[tuple[str, str]]], dict[tuple[str, ...], list[tuple[str, str]]]
]:
    """Write module pages and collect summary/package child metadata."""
    summary_sections: dict[str, list[tuple[str, str]]] = defaultdict(list)
    package_children: dict[tuple[str, ...], list[tuple[str, str]]] = defaultdict(list)

    for module_identifier, doc_path, members in reference_modules:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(
            _mkdocstrings_block(module_identifier, members), encoding="utf-8"
        )

        parts = tuple(module_identifier.split("."))
        section_key = parts[1] if len(parts) > 1 else ""
        rel_doc_path = doc_path.relative_to(REFERENCE).as_posix()
        summary_sections[section_key].append((module_identifier, rel_doc_path))

        for depth in range(1, len(parts)):
            package_children[parts[:depth]].append((module_identifier, rel_doc_path))

    return summary_sections, package_children


def _write_package_pages(
    package_paths: set[tuple[str, ...]],
    package_children: dict[tuple[str, ...], list[tuple[str, str]]],
) -> None:
    """Write package overview pages listing descendant modules."""
    for package_parts in sorted(package_paths):
        package_identifier = ".".join(package_parts)
        package_path = REFERENCE / Path(*package_parts) / "overview.md"
        package_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            _FRONT_MATTER,
            f"# `{package_identifier}`\n\n",
            f"Reference pages available under `{package_identifier}`.\n\n",
        ]

        for module_identifier, rel_doc_path in sorted(
            package_children.get(package_parts, [])
        ):
            relative_target = (
                Path(rel_doc_path).relative_to(Path(*package_parts)).as_posix()
            )
            lines.append(f"- [`{module_identifier}`]({relative_target})\n")

        package_path.write_text("".join(lines), encoding="utf-8")


def _write_summary_page(summary_sections: dict[str, list[tuple[str, str]]]) -> None:
    """Write the generated module summary page."""
    summary_path = REFERENCE / "summary.md"
    with summary_path.open("w", encoding="utf-8") as summary_page:
        summary_page.write(_FRONT_MATTER)
        summary_page.write("# Module Summary\n\n")
        summary_page.write(
            "Complete listing of every documented module in the `activelearning` package, "
            "auto-generated from the source tree.\n\n"
        )

        ordered_keys = ([""] if "" in summary_sections else []) + sorted(
            key for key in summary_sections if key != ""
        )
        for key in ordered_keys:
            heading = "activelearning" if key == "" else f"activelearning.{key}"
            summary_page.write(f"## `{heading}`\n\n")
            for module_identifier, rel_doc_path in sorted(summary_sections[key]):
                summary_page.write(f"- [`{module_identifier}`]({rel_doc_path})\n")
            summary_page.write("\n")


def main() -> None:
    """Regenerate the reference tree under ``docs/reference``."""
    for generated_path in (REFERENCE / "activelearning", REFERENCE / "summary.md"):
        if generated_path.is_dir():
            shutil.rmtree(generated_path)
        elif generated_path.exists():
            generated_path.unlink()
    REFERENCE.mkdir(parents=True, exist_ok=True)

    reference_modules, package_paths = _iter_reference_modules()
    summary_sections, package_children = _write_module_pages(reference_modules)
    _write_package_pages(package_paths, package_children)
    _write_summary_page(summary_sections)


if __name__ == "__main__":
    main()
