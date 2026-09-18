"""Command-line entry point for molecular active-learning experiments."""

from collections.abc import Sequence

from activelearning.main import run

from activelearning_molecules.config_catalogs import CONFIG_CATALOGS


def main(argv: Sequence[str] | None = None) -> None:
    """Run an active-learning experiment with molecular components."""
    run(
        argv,
        catalogs={"activelearning-molecules": CONFIG_CATALOGS},
        program_name="activelearning-molecules",
    )
