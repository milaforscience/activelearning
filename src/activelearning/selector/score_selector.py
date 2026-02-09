from activelearning.selector.selector import Selector


class ScoreSelector(Selector):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, candidates, acquisition=None, **kwargs):
        if acquisition is None:
            raise ValueError("Acquisition function is required for ScoreSelector.")

        values = acquisition(candidates)
        ranked = sorted(zip(candidates, values), key=lambda cv: cv[1], reverse=True)
        return [candidate for candidate, _ in ranked[: self.num_samples]]
