import distrax
import jax
import jax.numpy as jnp

from jaxnasium.algorithms.core._distributions import (
    DistraxContainer,
    EpsilonGreedy,
    TanhNormal,
    TanhNormalFactory,
)
from jaxnasium.tree._tree import _transpose_tree_of_tuples

SEED = jax.random.PRNGKey(0)


def _make_normal(loc=0.0, scale=1.0):
    return distrax.Normal(loc=jnp.array(loc), scale=jnp.array(scale))


# ---------------------------------------------------------------------------
# _transpose_tree_of_tuples unit tests
# ---------------------------------------------------------------------------


class TestTransposeTreeOfTuples:
    def test_flat_dict(self):
        """Dict of tuples -> tuple of dicts."""
        tree = {"a": (1, 10), "b": (2, 20)}
        outer_td = jax.tree.structure({"a": 0, "b": 0})
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == {"a": 1, "b": 2}
        assert result[1] == {"a": 10, "b": 20}

    def test_nested_dict(self):
        """Nested dict of tuples -> tuple of nested dicts."""
        tree = {"a": {"x": (1, 10), "y": (2, 20)}, "b": (3, 30)}
        outer_td = jax.tree.structure({"a": {"x": 0, "y": 0}, "b": 0})
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == {"a": {"x": 1, "y": 2}, "b": 3}
        assert result[1] == {"a": {"x": 10, "y": 20}, "b": 30}

    def test_tuple_structure(self):
        """Tuple-of-tuples where outer is structural, inner is result."""
        tree = ((1, 10), (2, 20))
        outer_td = jax.tree.structure((0, 0))
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == (1, 2)
        assert result[1] == (10, 20)

    def test_list_structure(self):
        """List of tuples -> tuple of lists."""
        tree = [(1, 10), (2, 20)]
        outer_td = jax.tree.structure([0, 0])
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == [1, 2]
        assert result[1] == [10, 20]

    def test_passthrough_when_no_tuples(self):
        """Non-tuple leaves are returned unchanged."""
        tree = {"a": 1, "b": 2}
        outer_td = jax.tree.structure({"a": 0, "b": 0})
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert result == tree


# ---------------------------------------------------------------------------
# DistraxContainer tests
# ---------------------------------------------------------------------------


class TestDistraxContainerSingle:
    """Single (non-nested) distribution."""

    def test_sample(self):
        c = DistraxContainer(distribution=_make_normal())
        s = c.sample(seed=SEED)
        assert s.shape == ()

    def test_sample_and_log_prob(self):
        c = DistraxContainer(distribution=_make_normal())
        s, lp = c.sample_and_log_prob(seed=SEED)
        assert s.shape == ()
        assert lp.shape == ()

    def test_log_prob(self):
        c = DistraxContainer(distribution=_make_normal())
        lp = c.log_prob(jnp.array(0.5))
        assert lp.shape == ()


class TestDistraxContainerDict:
    """Dict of distributions (flat)."""

    def setup_method(self):
        self.container = DistraxContainer(
            distribution={"a": _make_normal(0.0), "b": _make_normal(1.0)}
        )

    def test_sample(self):
        s = self.container.sample(seed=SEED)
        assert isinstance(s, dict) and set(s.keys()) == {"a", "b"}

    def test_sample_and_log_prob(self):
        s, lp = self.container.sample_and_log_prob(seed=SEED)
        assert isinstance(s, dict) and set(s.keys()) == {"a", "b"}
        assert isinstance(lp, dict) and set(lp.keys()) == {"a", "b"}

    def test_log_prob(self):
        lp = self.container.log_prob({"a": jnp.array(0.0), "b": jnp.array(1.0)})
        assert isinstance(lp, dict) and set(lp.keys()) == {"a", "b"}

    def test_getattr_sample_and_log_prob(self):
        """sample_and_log_prob called via __getattr__ returns (dict, dict)."""
        s, lp = self.container.sample_and_log_prob(seed=SEED)
        assert isinstance(s, dict)
        assert isinstance(lp, dict)


class TestDistraxContainerNestedDict:
    """Nested dict of distributions — the case the old code failed on."""

    def setup_method(self):
        self.container = DistraxContainer(
            distribution={
                "group1": {"x": _make_normal(0.0), "y": _make_normal(1.0)},
                "group2": _make_normal(2.0),
            }
        )

    def test_sample(self):
        s = self.container.sample(seed=SEED)
        assert isinstance(s["group1"], dict)
        assert set(s["group1"].keys()) == {"x", "y"}
        assert s["group2"].shape == ()

    def test_sample_and_log_prob(self):
        s, lp = self.container.sample_and_log_prob(seed=SEED)
        # samples
        assert isinstance(s, dict)
        assert isinstance(s["group1"], dict)
        assert set(s["group1"].keys()) == {"x", "y"}
        assert s["group2"].shape == ()
        # log_probs
        assert isinstance(lp, dict)
        assert isinstance(lp["group1"], dict)
        assert set(lp["group1"].keys()) == {"x", "y"}
        assert lp["group2"].shape == ()

    def test_log_prob(self):
        value = {
            "group1": {"x": jnp.array(0.0), "y": jnp.array(1.0)},
            "group2": jnp.array(2.0),
        }
        lp = self.container.log_prob(value)
        assert isinstance(lp["group1"], dict)  # type: ignore


class TestDistraxContainerTupleStructure:
    """Tuple of distributions — the tricky case with tuple-as-structure."""

    def setup_method(self):
        self.container = DistraxContainer(
            distribution=(_make_normal(0.0), _make_normal(1.0))
        )

    def test_sample(self):
        s = self.container.sample(seed=SEED)
        assert isinstance(s, tuple) and len(s) == 2

    def test_sample_and_log_prob(self):
        s, lp = self.container.sample_and_log_prob(seed=SEED)
        assert isinstance(s, tuple) and len(s) == 2
        assert isinstance(lp, tuple) and len(lp) == 2
        for val in (*s, *lp):
            assert val.shape == ()

    def test_log_prob(self):
        lp = self.container.log_prob((jnp.array(0.0), jnp.array(1.0)))
        assert isinstance(lp, tuple) and len(lp) == 2


def test_samples_within_default_bounds():
    dist = TanhNormal(mean=jnp.zeros((4,)), std=jnp.ones((4,)))
    samples = dist.sample(seed=SEED, sample_shape=(1000,))
    assert samples.shape == (1000, 4)
    # Default shift=0, scale=1 -> support is (-1, 1).
    assert jnp.all(samples > -1.0)
    assert jnp.all(samples < 1.0)


def test_mode_matches_shifted_scaled_tanh():
    mean = jnp.array([0.5, -1.0, 0.0])
    dist = TanhNormal(mean=mean, std=jnp.ones((3,)), shift=1.0, scale=2.0)
    assert jnp.allclose(dist.mode(), 1.0 + 2.0 * jnp.tanh(mean))


def test_batch_and_event_shape():
    dist = TanhNormal(mean=jnp.zeros((4,)), std=jnp.ones((4,)))
    assert dist.batch_shape == (4,)
    assert dist.event_shape == ()


def test_factory_sets_shift_and_scale_from_bounds():
    low, high = -3.0, 5.0
    factory = TanhNormalFactory(low, high)
    dist = factory(mean=jnp.zeros((2,)), std=jnp.ones((2,)))
    assert isinstance(dist, TanhNormal)
    # scale = (high - low) / 2, shift = (high + low) / 2
    assert jnp.allclose(dist._scale, 4.0)
    assert jnp.allclose(dist._shift, 1.0)


def test_factory_samples_within_bounds():
    low, high = -3.0, 5.0
    factory = TanhNormalFactory(low, high)
    dist = factory(mean=jnp.zeros((3,)), std=jnp.ones((3,)))
    samples = dist.sample(seed=SEED, sample_shape=(1000,))
    assert jnp.all(samples > low)
    assert jnp.all(samples < high)


_PREFS = jnp.array([1.0, 5.0, 2.0, 3.0])


def test_epsilon_greedy_masked_actions_get_zero_probability():
    mask = jnp.array([False, True, True, False])
    for epsilon in (0.0, 0.5, 1.0):
        probs = EpsilonGreedy(
            _PREFS, epsilon=epsilon, action_mask=mask
        ).distributions.probs
        assert jnp.allclose(probs[0], 0.0), epsilon
        assert jnp.allclose(probs[3], 0.0), epsilon
        assert jnp.allclose(probs.sum(), 1.0), epsilon


def test_epsilon_greedy_explores_uniformly_over_valid_actions():
    mask = jnp.array([False, True, True, False])
    probs = EpsilonGreedy(_PREFS, epsilon=1.0, action_mask=mask).distributions.probs
    assert jnp.allclose(probs, jnp.array([0.0, 0.5, 0.5, 0.0]))


def test_epsilon_greedy_is_greedy_over_valid_actions_only():
    """The argmax must avoid a masked action even if it has the highest preference."""
    mask = jnp.array([True, False, True, True])  # index 1 is the unmasked argmax
    probs = EpsilonGreedy(_PREFS, epsilon=0.0, action_mask=mask).distributions.probs
    assert jnp.allclose(probs, jnp.array([0.0, 0.0, 0.0, 1.0]))  # next best is index 3


def test_epsilon_greedy_all_masked_falls_back_to_uniform():
    mask = jnp.zeros((4,), dtype=bool)
    for epsilon in (0.0, 0.5, 1.0):
        probs = EpsilonGreedy(
            _PREFS, epsilon=epsilon, action_mask=mask
        ).distributions.probs
        assert jnp.all(jnp.isfinite(probs)), epsilon
        assert jnp.allclose(probs, 0.25), epsilon


def test_epsilon_greedy_mask_supports_pytree_action_spaces():
    prefs = {"a": jnp.array([1.0, 5.0]), "b": jnp.array([2.0, 0.0, 1.0])}
    masks = {"a": jnp.array([True, False]), "b": jnp.array([False, True, True])}
    dist = EpsilonGreedy(prefs, epsilon=1.0, action_mask=masks)
    probs = jax.tree.map(
        lambda d: d.probs,
        dist.distributions,
        is_leaf=lambda x: isinstance(x, distrax.Distribution),
    )
    assert jnp.allclose(probs["a"], jnp.array([1.0, 0.0]))
    assert jnp.allclose(probs["b"], jnp.array([0.0, 0.5, 0.5]))
