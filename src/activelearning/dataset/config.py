import csv
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from activelearning.dataset.dataset import Dataset
from activelearning.dataset.list_dataset import ListDataset
from activelearning.utils.types import Observation

MetadataColumns = Literal["remaining"] | list[str] | None


class CSVInitialDataConfig(BaseModel):
    """CSV source for preloading observations into a ListDataset."""

    path: Path
    x_columns: str | list[str]
    y_column: str = "y"
    fidelity_column: str | None = None
    metadata_columns: MetadataColumns = "remaining"

    def load_observations(
        self,
        negate_targets: bool = False,
    ) -> list[Observation]:
        """Load observations from the configured CSV file."""

        if not self.path.is_file():
            raise FileNotFoundError(f"Initial data CSV not found: {self.path!s}")

        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV {self.path!s} must contain a header row.")
            self._validate_columns(reader.fieldnames)

            observations = []
            for row_number, raw_row in enumerate(reader, start=2):
                observation = self._parse_observation(
                    raw_row,
                    row_number,
                    negate_targets=negate_targets,
                )
                if observation is not None:
                    observations.append(observation)

        if not observations:
            raise ValueError(f"CSV {self.path!s} must contain at least one data row.")
        return observations

    @property
    def _x_column_names(self) -> list[str]:
        if isinstance(self.x_columns, str):
            return [self.x_columns]
        return list(self.x_columns)

    @property
    def _consumed_columns(self) -> set[str]:
        columns = {*self._x_column_names, self.y_column}
        if self.fidelity_column is not None:
            columns.add(self.fidelity_column)
        return columns

    def _validate_columns(self, fieldnames: list[str]) -> None:
        required_columns = [*self._x_column_names, self.y_column]
        if self.fidelity_column is not None:
            required_columns.append(self.fidelity_column)
        if isinstance(self.metadata_columns, list):
            required_columns.extend(self.metadata_columns)

        missing = sorted(set(required_columns) - set(fieldnames))
        if missing:
            raise ValueError(
                f"CSV {self.path!s} is missing required columns: {', '.join(missing)}."
            )

    def _parse_observation(
        self,
        raw_row: dict[str, str | None],
        row_number: int,
        *,
        negate_targets: bool,
    ) -> Observation | None:
        if not any((value or "").strip() for value in raw_row.values()):
            return None

        row = {
            key: value if value is not None else "" for key, value in raw_row.items()
        }
        x_value: str | list[float]
        if isinstance(self.x_columns, str):
            x_value = self._read_string(row, self.x_columns, row_number)
        else:
            x_value = [
                self._read_float(row, column, row_number)
                for column in self._x_column_names
            ]

        return Observation(
            x=x_value,
            y=_maybe_negate_target(
                self._read_float(row, self.y_column, row_number),
                negate_targets=negate_targets,
            ),
            fidelity=self._read_fidelity(row, row_number),
            metadata=self._read_metadata(row),
        )

    def _read_fidelity(self, row: dict[str, str], row_number: int) -> int | None:
        if self.fidelity_column is None:
            return None

        raw_value = row[self.fidelity_column].strip()
        if raw_value == "":
            return None
        try:
            return int(raw_value)
        except ValueError as error:
            raise ValueError(
                f"CSV {self.path!s} row {row_number} column {self.fidelity_column!r} "
                f"must be an integer fidelity; got {raw_value!r}."
            ) from error

    def _read_metadata(self, row: dict[str, str]) -> dict[str, str] | None:
        if self.metadata_columns is None:
            return None
        if self.metadata_columns == "remaining":
            metadata = {
                key: value
                for key, value in row.items()
                if key not in self._consumed_columns and value.strip() != ""
            }
        else:
            metadata = {
                key: row[key] for key in self.metadata_columns if row[key].strip() != ""
            }
        return metadata or None

    def _read_float(
        self,
        row: dict[str, str],
        column: str,
        row_number: int,
    ) -> float:
        raw_value = row[column].strip()
        try:
            return float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"CSV {self.path!s} row {row_number} column {column!r} "
                f"must be a float; got {raw_value!r}."
            ) from error

    def _read_string(
        self,
        row: dict[str, str],
        column: str,
        row_number: int,
    ) -> str:
        value = row[column].strip()
        if value == "":
            raise ValueError(
                f"CSV {self.path!s} row {row_number} column {column!r} must be non-empty."
            )
        return value


class ListDatasetConfig(BaseModel):
    type: Literal["ListDataset"] = "ListDataset"
    initial_data: CSVInitialDataConfig | None = None
    negate_initial_targets: bool = False

    def build(self) -> Dataset:
        observations = (
            self.initial_data.load_observations(
                negate_targets=self.negate_initial_targets
            )
            if self.initial_data is not None
            else None
        )
        return ListDataset(initial_observations=observations)


def _maybe_negate_target(value: float, *, negate_targets: bool) -> float:
    """Optionally negate seeded target values before they enter the dataset."""

    return -value if negate_targets else value


DatasetConfig = Annotated[
    Union[ListDatasetConfig],
    Field(discriminator="type"),
]
