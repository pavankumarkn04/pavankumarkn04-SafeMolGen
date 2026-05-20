from models.integrated.pipeline import _apply_improvement_pacing


class _Pred:
    def __init__(self, value: float):
        self.overall_prob = value


def test_improvement_pacing_first_iteration():
    pool = [("s1", _Pred(0.40)), ("s2", _Pred(0.08)), ("s3", _Pred(0.05)), ("s4", _Pred(0.02))]
    out = _apply_improvement_pacing(pool, 0.0, 0.5, 0, 6, 0.05)
    assert out is not None
    assert out[1].overall_prob == 0.05


def test_improvement_pacing_later_bounded():
    pool = [("a", _Pred(0.12)), ("b", _Pred(0.25)), ("c", _Pred(0.14))]
    out = _apply_improvement_pacing(pool, 0.10, 0.5, 2, 6, 0.05)
    assert out is not None
    assert out[1].overall_prob == 0.14


def test_improvement_pacing_later_smallest_fallback():
    pool = [("a", _Pred(0.15)), ("b", _Pred(0.30))]
    out = _apply_improvement_pacing(pool, 0.06, 0.5, 1, 6, 0.05)
    assert out is not None
    assert out[1].overall_prob == 0.15

