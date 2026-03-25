from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from activelearning.dataset.dataset import Dataset
from activelearning.dataset.list_dataset import ListDataset


class ListDatasetConfig(BaseModel):
    type: Literal["ListDataset"] = "ListDataset"

    def build(self) -> Dataset:
        return ListDataset()


DatasetConfig = Annotated[
    Union[ListDatasetConfig],
    Field(discriminator="type"),
]
