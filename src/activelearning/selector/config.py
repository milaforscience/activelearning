"""Pydantic models of selectors."""

from typing import Literal

from pydantic import Field

from activelearning.config_registry import BuildableConfig, registered_config
from activelearning.selector.cost_aware_selector import CostAwareSelector
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.selector.selector import Selector


class TopKAcquisitionSelectorConfig(BuildableConfig):
    type: Literal["TopKAcquisitionSelector"] = "TopKAcquisitionSelector"
    num_samples: int = Field(gt=0)

    def build(self) -> Selector:
        return TopKAcquisitionSelector(num_samples=self.num_samples)


class CostAwareSelectorConfig(BuildableConfig):
    type: Literal["CostAwareSelector"] = "CostAwareSelector"

    def build(self) -> Selector:
        return CostAwareSelector()


SELECTOR_CONFIGS = (TopKAcquisitionSelectorConfig, CostAwareSelectorConfig)
SelectorConfig = registered_config("selector")
