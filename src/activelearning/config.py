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
