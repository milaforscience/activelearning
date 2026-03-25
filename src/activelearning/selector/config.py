from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from activelearning.selector.cost_aware_selector import CostAwareSelector
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.selector.selector import Selector


class TopKAcquisitionSelectorConfig(BaseModel):
    type: Literal["TopKAcquisitionSelector"] = "TopKAcquisitionSelector"
    num_samples: int

    def build(self) -> Selector:
        return TopKAcquisitionSelector(num_samples=self.num_samples)


class CostAwareSelectorConfig(BaseModel):
    type: Literal["CostAwareSelector"] = "CostAwareSelector"

    def build(self) -> Selector:
        return CostAwareSelector()


SelectorConfig = Annotated[
    Union[TopKAcquisitionSelectorConfig, CostAwareSelectorConfig],
    Field(discriminator="type"),
]
