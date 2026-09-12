"""Configuration models provided by the molecular distribution."""

from activelearning_molecules.encoders.config import (
    MOLECULE_ENCODER_CONFIGS,
    MOLECULE_FIXED_ENCODER_CONFIGS,
)
from activelearning_molecules.oracles.config import MOLECULE_ORACLE_CONFIGS
from activelearning_molecules.samplers.config import MOLECULE_SAMPLER_CONFIGS


CONFIG_CATALOGS = {
    "sampler": MOLECULE_SAMPLER_CONFIGS,
    "oracle": MOLECULE_ORACLE_CONFIGS,
    "encoder": MOLECULE_ENCODER_CONFIGS,
    "fixed_encoder": MOLECULE_FIXED_ENCODER_CONFIGS,
}
