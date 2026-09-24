"""Tests for the Schedule helper (constant + linear, single- and multi-agent)."""

import pytest

from jaxnasium.algorithms.core import Schedule


def test_multi_agent_per_agent_start_and_end():
    """Both start and end are per-agent pytrees."""
    schedule = Schedule(
        start={"agent_0": 1.0, "agent_1": 0.0},
        end={"agent_0": 0.0, "agent_1": 4.0},
        transition_steps=10,
    )
    out = schedule(5)
    assert isinstance(out, dict)
    assert out["agent_0"] == pytest.approx(0.5)
    assert out["agent_1"] == pytest.approx(2.0)


def test_multi_agent_constant_returns_start_tree():
    schedule = Schedule(
        start={"agent_0": 0.3, "agent_1": 0.7}, end=None, transition_steps=10
    )
    out = schedule(5)
    assert out == {"agent_0": 0.3, "agent_1": 0.7}


# test with end values that are None for one agent
def test_multi_agent_constant_returns_start_tree_with_none_end():
    schedule = Schedule(
        start={"agent_0": 1.0, "agent_1": 1.0},
        end={"agent_0": None, "agent_1": 0.0},
        transition_steps=10,
    )
    out = schedule(5)
    assert out == {"agent_0": 1.0, "agent_1": 0.5}
