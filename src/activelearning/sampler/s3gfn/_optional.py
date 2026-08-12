"""Lazy optional-dependency boundaries for the S3-GFN core."""

from __future__ import annotations

from types import ModuleType
from typing import Any


class S3GFNOptionalDependencyError(ImportError):
    """Raised when an optional molecular-generation dependency is unavailable."""


def require_transformers() -> tuple[Any, Any]:
    """Return Hugging Face auto classes or raise an actionable import error.

    Returns
    -------
    tuple[type, type]
        ``(AutoModelForCausalLM, AutoTokenizer)`` from Transformers.

    Raises
    ------
    S3GFNOptionalDependencyError
        If the molecule dependencies are not installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise S3GFNOptionalDependencyError(
            "S3-GFN model loading requires the molecule dependencies. "
            "Install them with: uv sync --extra molecules."
        ) from error
    return AutoModelForCausalLM, AutoTokenizer


def require_rdkit() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Return RDKit modules used by replay and SA-score evaluation.

    Returns
    -------
    tuple[ModuleType, ModuleType, ModuleType]
        ``(Chem, DataStructs, AllChem)``.

    Raises
    ------
    S3GFNOptionalDependencyError
        If RDKit is not installed.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as error:
        raise S3GFNOptionalDependencyError(
            "Molecular validation and Tanimoto replay require RDKit. "
            "Install it with: uv sync --extra molecules."
        ) from error
    return Chem, DataStructs, AllChem


def require_sa_scorer() -> Any:
    """Return RDKit's contributed synthetic-accessibility scorer.

    Returns
    -------
    Any
        RDKit's ``sascorer`` module.

    Raises
    ------
    S3GFNOptionalDependencyError
        If RDKit or its contributed SA-score module is unavailable.
    """
    require_rdkit()
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError as error:
        raise S3GFNOptionalDependencyError(
            "The installed RDKit build does not include Contrib.SA_Score."
        ) from error
    return sascorer
