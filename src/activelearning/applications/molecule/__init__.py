"""Molecule-specific AL components: tokenization, encoding, DKL surrogate, XTB oracle.

To use this module install the molecule extras::

    uv sync --extra molecule
"""

try:
    import selfies  # noqa: F401
    import rdkit  # noqa: F401
except ImportError as _err:
    raise ImportError(
        "The molecule application requires optional dependencies that are not installed.\n"
        "Install them with:  uv sync --extra molecule\n"
        "or:                 pip install activelearning[molecule]"
    ) from _err

from activelearning.applications.molecule.selfies_tokenizer import (
    SELFIES_VOCAB_SMALL,
    SelfiesTokenizer,
)
from activelearning.applications.molecule.selfies_transformer_encoder import (
    SelfiesTransformerEncoder,
)

__all__ = [
    "SELFIES_VOCAB_SMALL",
    "SelfiesTransformerEncoder",
    "SelfiesTokenizer",
]
