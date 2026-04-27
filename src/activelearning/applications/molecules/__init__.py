"""Molecule-specific AL components: tokenization, encoding, DKL surrogate, XTB oracle.

To use this module install the molecules extras::

    uv sync --extra molecules
"""

try:
    import selfies  # noqa: F401
    import rdkit  # noqa: F401
except ImportError as _err:
    raise ImportError(
        "The molecules application requires optional dependencies that are not installed.\n"
        "Install them with:  uv sync --extra molecules\n"
        "or:                 pip install activelearning[molecules]"
    ) from _err

from activelearning.applications.molecules.selfies_tokenizer import (
    SELFIES_VOCAB_SMALL,
    SelfiesTokenizer,
)
from activelearning.applications.molecules.selfies_transformer_encoder import (
    SelfiesTransformerEncoder,
)

__all__ = [
    "SELFIES_VOCAB_SMALL",
    "SelfiesTransformerEncoder",
    "SelfiesTokenizer",
]
