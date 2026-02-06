from activelearning.oracle.oracle import Oracle


class DummyOracle(Oracle):
    """Deterministic oracle with constant per-sample cost."""

    def __init__(self, cost_per_sample=1.0, score_fn=None):
        self._cost_per_sample = float(cost_per_sample)
        self._score_fn = score_fn or (lambda s: float(s))

    def get_cost(self, candidates):
        """Return cost proportional to number of candidates."""
        return self._cost_per_sample * len(candidates)

    def query(self, candidates):
        """Score candidates via the provided scoring function."""
        scores = []
        for sample in candidates:
            value = sample.x if hasattr(sample, "x") else sample
            scores.append(self._score_fn(value))
        return scores
