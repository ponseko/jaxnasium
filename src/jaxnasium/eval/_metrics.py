"""Performance metrics in the style of `rliable`.

https://github.com/google-research/rliable/

https://openreview.net/pdf?id=uqv8-U4lKBe
"""

import numpy as np
import scipy.stats


def iqm(scores) -> float:
    """Interquartile mean: the mean of the middle 50% of `scores`.

    Args:
        scores: Scores of every run (any shape; flattened). The IQM is taken over
        the full set of given scores (axis=None)

    Returns:
        The 25% trimmed mean.
    """
    return float(scipy.stats.trim_mean(scores, 0.25, axis=None))  # type: ignore


def optimality_gap(scores, baseline: float) -> float:
    """How far `scores` fall short of `baseline`, on average.

    Args:
        scores: Scores of every run (any shape; flattened).
        baseline: The threshold considered "solved".

    Returns:
        `baseline - mean(min(scores, baseline))`, which is 0 exactly when every
        run reaches the baseline and is never negative.
    """
    return float(baseline - np.mean(np.minimum(scores, baseline)))


def probability_of_improvement(scores_x, scores_y) -> float:
    """Probability that a random run of `X` beats a random run of `Y`.

    Note that this is the probability of improvement for a single task (the scores
    are ravalled to a flat array). Typically, when considering multiple tasks (e.g. environments)
    one may want to take the POI for each individually task and average those.

    Args:
        scores_x: Scores of every run of algorithm `X` (any shape; flattened).
        scores_y: Scores of every run of algorithm `Y` (any shape; flattened).

    Returns:
        A probability in [0, 1]; 0.5 means the two are indistinguishable.
    """
    statistic = scipy.stats.mannwhitneyu(
        scores_x,
        scores_y,
        alternative="greater",
        axis=None,  # type: ignore
    ).statistic  # type: ignore
    return float(statistic / (np.size(scores_x) * np.size(scores_y)))
