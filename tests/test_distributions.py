"""Tests for DistraxContainer and _transpose_tree_of_tuples."""

import distrax
import jax
import jax.numpy as jnp

from jaxnasium.algorithms.utils._distributions import (
    DistraxContainer,
    _transpose_tree_of_tuples,
)

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
        assert isinstance(lp["group1"], dict)  # pyright: ignore[reportIndexIssue]


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
