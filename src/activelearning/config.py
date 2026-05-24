"""Pydantic model for the configuration of an active learning run.

The definition of configurations is based on pydantic. The class ActiveLearningConfig
is a pydantic models which defines the inputs of an active learning run (dataset,
surrofate, acquisition, etc.), which are defined within the corresponding modules.

Changes in the active learning interface should be reflected in this configuration to
ensure consistency.

See the `Pydantic Docs <https://pydantic.dev/docs/validation/latest/get-started/>`_ for
further reference.
"""

from pydantic import BaseModel, Field

from activelearning.acquisition.config import AcquisitionConfig
from activelearning.budget.config import BudgetConfig
from activelearning.dataset.config import DatasetConfig
from activelearning.logger.config import LoggerConfig
from activelearning.oracle.config import OracleConfig
from activelearning.sampler.config import SamplerConfig
from activelearning.selector.config import SelectorConfig
from activelearning.surrogate.config import SurrogateConfig
from activelearning.runtime import RuntimeConfig


class ActiveLearningConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    dataset: DatasetConfig
    surrogate: SurrogateConfig
    acquisition: AcquisitionConfig
    sampler: SamplerConfig
    selector: SelectorConfig
    oracle: OracleConfig
    budget: BudgetConfig
    logger: LoggerConfig | None = None
