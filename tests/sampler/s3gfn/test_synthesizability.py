from __future__ import annotations

import activelearning.sampler.s3gfn.synthesizability as synth_module


class _FakeChem:
    @staticmethod
    def MolFromSmiles(smiles: str):
        return None if smiles == "invalid" else smiles

    @staticmethod
    def MolToSmiles(molecule, isomericSmiles=False):
        return molecule.replace(" ", "")


class _FakeScorer:
    calls = 0

    @classmethod
    def calculateScore(cls, molecule):
        cls.calls += 1
        return {"CC": 2.0, "CCC": 4.0}[molecule]


def test_sa_scores_are_cached_and_thresholded(monkeypatch):
    monkeypatch.setattr(
        synth_module,
        "require_rdkit",
        lambda: (_FakeChem, object(), object()),
    )
    monkeypatch.setattr(
        synth_module,
        "require_sa_scorer",
        lambda: _FakeScorer,
    )
    _FakeScorer.calls = 0
    predicate = synth_module.SAScoreSynthesizability(threshold=4.0)

    assert predicate("CC")
    assert predicate("CC")
    assert not predicate("CCC")
    assert not predicate("invalid")
    assert _FakeScorer.calls == 2
