"""Configuration models provided by the core distribution."""

from activelearning.acquisition.config import ACQUISITION_CONFIGS
from activelearning.dataset.config import DATASET_CONFIGS
from activelearning.logger.config import LOGGER_CONFIGS
from activelearning.oracle.config import ORACLE_CONFIGS
from activelearning.sampler.config import SAMPLER_CONFIGS
from activelearning.selector.config import SELECTOR_CONFIGS
from activelearning.surrogate.config import SURROGATE_CONFIGS
from activelearning.surrogate.encoder_config import (
    ENCODER_CONFIGS,
    FIXED_ENCODER_CONFIGS,
)


CONFIG_CATALOGS = {
    "dataset": DATASET_CONFIGS,
    "surrogate": SURROGATE_CONFIGS,
    "acquisition": ACQUISITION_CONFIGS,
    "sampler": SAMPLER_CONFIGS,
    "selector": SELECTOR_CONFIGS,
    "oracle": ORACLE_CONFIGS,
    "logger": LOGGER_CONFIGS,
    "encoder": ENCODER_CONFIGS,
    "fixed_encoder": FIXED_ENCODER_CONFIGS,
}
