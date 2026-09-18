"""Validation tests for the bundled xTB IPEA benchmark data."""

from collections import Counter
import csv
import math
from pathlib import Path

import pytest
import selfies
from rdkit import Chem

from activelearning.dataset.config import CSVInitialDataConfig


DATA_DIR = (
    Path(__file__).resolve().parents[1] / "config" / "xtb_ipea_benchmark" / "data"
)

SINGLE_FIDELITY_SCHEMAS = (
    ("ea_sf", 135, ["selfies", "y", "source_split", "smiles"]),
    ("ip_sf", 135, ["selfies", "y", "source_split", "smiles"]),
)

MULTI_FIDELITY_SCHEMAS = (
    (
        "ea_mf",
        699,
        ["selfies", "y", "fidelity", "source_split", "smiles"],
        {1: 624, 2: 61, 3: 14},
    ),
    (
        "ip_mf",
        705,
        ["selfies", "y", "fidelity", "source_split", "smiles"],
        {1: 630, 2: 61, 3: 14},
    ),
)


def _read_benchmark_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    """Read one bundled benchmark CSV and return its schema and rows."""
    path = DATA_DIR / f"{name}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"{path} is missing a CSV header")
        return reader.fieldnames, list(reader)


@pytest.mark.parametrize(
    ("name", "expected_row_count", "expected_columns"),
    [*SINGLE_FIDELITY_SCHEMAS, *[entry[:3] for entry in MULTI_FIDELITY_SCHEMAS]],
)
def test_benchmark_csv_schema_and_row_count(
    name: str,
    expected_row_count: int,
    expected_columns: list[str],
) -> None:
    """Benchmark assets preserve source columns and expected train sizes."""
    columns, rows = _read_benchmark_csv(name)

    assert columns == expected_columns
    assert len(rows) == expected_row_count
    assert {row["source_split"] for row in rows} == {"train"}


@pytest.mark.parametrize(
    "name",
    ["ea_sf", "ea_mf", "ip_sf", "ip_mf"],
)
def test_benchmark_molecules_are_finite_valid_and_canonical(name: str) -> None:
    """Each row has finite targets and a valid connected canonical SMILES."""
    _, rows = _read_benchmark_csv(name)

    for row in rows:
        assert math.isfinite(float(row["y"]))

        decoded_smiles = selfies.decoder(row["selfies"])
        decoded_molecule = Chem.MolFromSmiles(decoded_smiles)
        assert decoded_molecule is not None
        assert len(Chem.GetMolFrags(decoded_molecule, asMols=False)) == 1

        canonical_smiles = Chem.MolToSmiles(decoded_molecule, canonical=True)
        assert row["smiles"] == canonical_smiles

        stored_molecule = Chem.MolFromSmiles(row["smiles"])
        assert stored_molecule is not None
        assert len(Chem.GetMolFrags(stored_molecule, asMols=False)) == 1


@pytest.mark.parametrize(
    ("name", "fidelity_column"),
    [
        ("ea_sf", None),
        ("ea_mf", "fidelity"),
        ("ip_sf", None),
        ("ip_mf", "fidelity"),
    ],
)
def test_benchmark_csv_loads_smiles_without_runtime_conversion(
    name: str,
    fidelity_column: str | None,
) -> None:
    """The current CSV loader can use the stored canonical SMILES directly."""
    observations = CSVInitialDataConfig(
        path=DATA_DIR / f"{name}.csv",
        x_columns="smiles",
        fidelity_column=fidelity_column,
    ).load_observations()

    _, rows = _read_benchmark_csv(name)

    assert len(observations) == len(rows)
    assert [observation.x for observation in observations] == [
        row["smiles"] for row in rows
    ]
    assert all(observation.metadata is not None for observation in observations)


@pytest.mark.parametrize(
    ("name", "expected_fidelities"),
    [(entry[0], entry[3]) for entry in MULTI_FIDELITY_SCHEMAS],
)
def test_multi_fidelity_benchmark_has_expected_fidelities(
    name: str,
    expected_fidelities: dict[int, int],
) -> None:
    """MF assets contain the expected counts at xTB fidelities one through three."""
    _, rows = _read_benchmark_csv(name)

    fidelities = Counter(int(row["fidelity"]) for row in rows)

    assert fidelities == expected_fidelities
