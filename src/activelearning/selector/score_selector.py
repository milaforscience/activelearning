from activelearning.selector.selector import Selector


class ScoreSelector(Selector):
    def __init__(self, score_fn, num_samples):
        self.score_fn = score_fn
        self.num_samples = num_samples

    def __call__(self, candidates):
        values = self.score_fn(candidates)
        ranked = sorted(zip(candidates, values), key=lambda cv: cv[1], reverse=True)
        return [candidate for candidate, _ in ranked[: self.num_samples]]
