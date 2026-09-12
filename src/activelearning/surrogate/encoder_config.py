"""Configuration contracts for extensible encoder schemas."""

from activelearning.config_registry import registered_config


ENCODER_CONFIGS = ()
FIXED_ENCODER_CONFIGS = ()
EncoderConfig = registered_config("encoder")
FixedEncoderConfig = registered_config("fixed_encoder")

__all__ = [
    "ENCODER_CONFIGS",
    "FIXED_ENCODER_CONFIGS",
    "EncoderConfig",
    "FixedEncoderConfig",
]
