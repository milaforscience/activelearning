"""SA-score based synthesizability predicates with lazy RDKit imports."""

from __future__ import annotations

# Adapted from hyeonahkimm/s3gfn; see THIRD_PARTY_NOTICES.md.

import math
from collections import OrderedDict
from collections.abc import Sequence

from activelearning.sampler.s3gfn._optional import require_rdkit, require_sa_scorer


def passes_sa_threshold(sa_score: float, threshold: float = 3.0) -> bool:
    """Return whether an SA score satisfies the strict threshold.

    Parameters
    ----------
    sa_score : float
        Synthetic-accessibility score. Non-finite scores are rejected.
    threshold : float, optional
        Strict upper bound. A score passes only when
        ``sa_score < threshold``.

    Returns
    -------
    bool
        ``True`` when the score is finite and below ``threshold``.

    Raises
    ------
    ValueError
        If ``threshold`` is not finite.
    """
    if not math.isfinite(sa_score):
        return False
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return sa_score < threshold


class SAScoreSynthesizability:
    """Classify molecules using RDKit's synthetic-accessibility score.

    The callable returns ``True`` exactly when a valid molecule has
    ``SA_score < threshold``. Empty and invalid SMILES receive
    ``invalid_score`` and are always classified as negative.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        invalid_score: float = 10.0,
        cache_size: int = 50_000,
    ) -> None:
        """Initialize the SA-score predicate and its bounded cache.

        Parameters
        ----------
        threshold : float, optional
            Strict maximum SA score accepted as synthesizable.
        invalid_score : float, optional
            Score assigned to empty, invalid, or non-finite-scoring SMILES.
            It must not pass the configured threshold.
        cache_size : int, optional
            Maximum number of canonical SMILES scores cached. Zero disables
            caching.

        Raises
        ------
        ValueError
            If a score is non-finite, ``invalid_score`` could pass the
            threshold, or ``cache_size`` is negative.
        """
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        if not math.isfinite(invalid_score):
            raise ValueError("invalid_score must be finite.")
        if invalid_score < threshold:
            raise ValueError("invalid_score must not pass the SA threshold.")
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative.")
        self.threshold = threshold
        self.invalid_score = invalid_score
        self.cache_size = cache_size
        self._scores: OrderedDict[str, float] = OrderedDict()

    def _canonicalize(self, smiles: str) -> tuple[str, object] | None:
        """Return canonical SMILES and its molecule, or reject invalid input."""
        if not isinstance(smiles, str):
            raise TypeError("smiles must be a string.")
        if not smiles.strip():
            return None
        Chem, _, _ = require_rdkit()
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False)
        return canonical_smiles, molecule

    def _cache(self, canonical_smiles: str, score: float) -> None:
        """Store one score in least-recently-used order."""
        if self.cache_size == 0:
            return
        self._scores[canonical_smiles] = score
        self._scores.move_to_end(canonical_smiles)
        if len(self._scores) > self.cache_size:
            self._scores.popitem(last=False)

    def score(self, smiles: str) -> float:
        """Return a molecule's SA score or ``invalid_score`` when invalid.

        Parameters
        ----------
        smiles : str
            SMILES string to parse and score. Equivalent valid strings share
            one cached score after canonicalization.

        Returns
        -------
        float
            The finite RDKit SA score, or ``invalid_score`` when the input is
            empty, invalid, or produces a non-finite score.
        """
        molecule_data = self._canonicalize(smiles)
        if molecule_data is None:
            return self.invalid_score
        canonical_smiles, molecule = molecule_data
        cached = self._scores.get(canonical_smiles)
        if cached is not None:
            self._scores.move_to_end(canonical_smiles)
            return cached

        scorer = require_sa_scorer()
        score = float(scorer.calculateScore(molecule))
        if not math.isfinite(score):
            return self.invalid_score
        self._cache(canonical_smiles, score)
        return score

    def __call__(self, smiles: str) -> bool:
        """Return whether one molecule passes the configured SA threshold.

        Parameters
        ----------
        smiles : str
            SMILES string to validate and score.

        Returns
        -------
        bool
            ``True`` when the molecule's finite SA score is strictly below
            the configured threshold.
        """
        return passes_sa_threshold(self.score(smiles), self.threshold)

    def score_batch(self, smiles: Sequence[str]) -> list[float]:
        """Return SA scores aligned with a SMILES sequence.

        Parameters
        ----------
        smiles : Sequence[str]
            SMILES strings to score in order.

        Returns
        -------
        list[float]
            One SA score per input string.
        """
        return [self.score(molecule_smiles) for molecule_smiles in smiles]

    def classify_batch(self, smiles: Sequence[str]) -> list[bool]:
        """Return synthesizability labels aligned with a SMILES sequence.

        Parameters
        ----------
        smiles : Sequence[str]
            SMILES strings to classify in order.

        Returns
        -------
        list[bool]
            One threshold result per input string.
        """
        return [
            passes_sa_threshold(score, self.threshold)
            for score in self.score_batch(smiles)
        ]
