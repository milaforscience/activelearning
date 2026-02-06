from activelearning.acquisition.acquisition import Acquisition


class DummyAcquisition(Acquisition):
    """UCB-style acquisition over "mean" and optional "std" predictions."""

    def __init__(self, beta=1.0):
        super().__init__()
        self._beta = float(beta)

    def __call__(self, candidates):
        """Score candidates using mean + beta * std when available."""
        if self.surrogate is None:
            return [0.0 for _ in candidates]
        pred = self.surrogate.predict(candidates)
        means = pred.get("mean")
        if means is None:
            raise ValueError("DummyAcquisition expects prediction payload key 'mean'.")
        stds = pred.get("std")
        if stds is None:
            return list(means)
        return [mean + self._beta * std for mean, std in zip(means, stds)]
