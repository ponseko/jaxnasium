import numpy as np
import pytest

from jaxnasium.eval import iqm, optimality_gap, probability_of_improvement


class TestIQM:
    def test_drops_the_outer_quartiles(self):
        # n=8 -> two values trimmed from each end, leaving the mean of 3..6.
        assert iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)

    def test_is_robust_to_extreme_runs(self):
        scores = [1, 2, 3, 4, 5, 6, 7, 8]
        outliers = [-1000, 2, 3, 4, 5, 6, 7, 1000]
        assert iqm(outliers) == pytest.approx(iqm(scores))

    def test_flattens_any_shape(self):
        scores = np.arange(8)
        assert iqm(scores.reshape(2, 4)) == pytest.approx(iqm(scores))

    def test_constant_scores(self):
        assert iqm(np.full((3, 4, 5), 2.5)) == pytest.approx(2.5)


class TestOptimalityGap:
    def test_zero_when_every_run_reaches_the_baseline(self):
        assert optimality_gap([1.0, 1.0, 5.0], baseline=1.0) == pytest.approx(0.0)

    def test_counts_only_the_shortfall(self):
        # Scores above the baseline are clipped, so only the 0.4 gap remains.
        assert optimality_gap([0.6, 1.0, 3.0], baseline=1.0) == pytest.approx(0.4 / 3)

    def test_never_negative(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=(5, 4))
        assert optimality_gap(scores, baseline=0.0) >= 0.0

    def test_flattens_any_shape(self):
        scores = np.linspace(0, 1, 12)
        flat = optimality_gap(scores, baseline=0.5)
        assert optimality_gap(scores.reshape(3, 4), baseline=0.5) == pytest.approx(flat)


class TestProbabilityOfImprovement:
    def test_one_when_x_always_wins(self):
        assert probability_of_improvement([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)

    def test_zero_when_x_always_loses(self):
        assert probability_of_improvement([1, 2, 3], [10, 11, 12]) == pytest.approx(0.0)

    def test_half_for_identical_distributions(self):
        scores = [1, 2, 3, 4]
        assert probability_of_improvement(scores, scores) == pytest.approx(0.5)

    def test_counts_pairwise_wins(self):
        # 3 of the 9 (x, y) pairs have x > y (3>2, 5>2, 5>4), and none are tied.
        assert probability_of_improvement([1, 3, 5], [2, 4, 6]) == pytest.approx(3 / 9)

    def test_is_complementary(self):
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=8), rng.normal(size=6)
        forward = probability_of_improvement(x, y)
        backward = probability_of_improvement(y, x)
        assert forward + backward == pytest.approx(1.0)

    def test_flattens_any_shape(self):
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=12), rng.normal(size=12)
        flat = probability_of_improvement(x, y)
        reshaped = probability_of_improvement(x.reshape(3, 4), y.reshape(4, 3))
        assert reshaped == pytest.approx(flat)
