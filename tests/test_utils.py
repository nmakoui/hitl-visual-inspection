from inspection.utils import normalise_confidence


def test_normalise_confidence_within_range():
    assert normalise_confidence(0.75) == 0.75


def test_normalise_confidence_clamps_above_one():
    assert normalise_confidence(1.4) == 1.0


def test_normalise_confidence_clamps_below_zero():
    assert normalise_confidence(-0.2) == 0.0