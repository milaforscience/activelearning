"""Tests for optional Hugging Face sequence dependencies."""

import builtins
from typing import Any

import pytest

from activelearning.surrogate.sequence.huggingface_tokenizer import (
    HuggingFaceTokenizer,
)


def test_missing_transformers_error_has_install_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Transformers install should produce actionable guidance."""
    original_import = builtins.__import__

    def block_transformers_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "transformers":
            raise ModuleNotFoundError("blocked optional dependency: transformers")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_transformers_import)

    with pytest.raises(
        ImportError,
        match=r"pip install activelearning\[transformers\]",
    ):
        HuggingFaceTokenizer("unused")
