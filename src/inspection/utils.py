"""Small shared helper functions for the inspection package."""


def normalise_confidence(score: float) -> float:
    """Clamp a confidence score into the valid [0.0, 1.0] range.

    Model outputs occasionally fall slightly outside this range due to
    floating-point rounding. Routing decisions later in the pipeline
    assume a valid probability, so every score is clamped here first.
    """
    return max(0.0, min(1.0, score))