"""Tests for the tabular Q-learning agent."""

import numpy as np
import pytest

from scalerl.agent import QLearningAgent
from scalerl.config import AgentConfig


def test_qtable_shape_matches_bins():
    cfg = AgentConfig()
    agent = QLearningAgent(cfg)
    expected = (cfg.latency_bins + 1, cfg.memory_bins + 1, cfg.request_bins + 1, cfg.action_size)
    assert agent.q_table.shape == expected


def test_discretize_within_index_bounds():
    agent = QLearningAgent()
    # Extreme values must still map to valid table indices.
    for state in [
        np.array([0.0, 0.0, 0.0]),
        np.array([1000.0, 100.0, 10000.0]),
        np.array([5000.0, 200.0, 99999.0]),  # beyond bounds
    ]:
        idx = agent.discretize(state)
        for i, dim in enumerate(agent.q_table.shape[:3]):
            assert 0 <= idx[i] < dim


def test_high_latency_states_are_representable():
    """Regression: original bins capped latency at 500 while it reaches 1000."""
    agent = QLearningAgent()
    low = agent.discretize(np.array([100.0, 50.0, 5000.0]))
    high = agent.discretize(np.array([900.0, 50.0, 5000.0]))
    assert high[0] > low[0]  # distinct latency bins at the high end


def test_learn_moves_qvalue_toward_reward():
    agent = QLearningAgent(AgentConfig(alpha=0.5, gamma=0.0))
    state = np.array([50.0, 40.0, 1000.0])
    next_state = np.array([55.0, 40.0, 1000.0])
    idx = agent.discretize(state)
    before = agent.q_table[idx][1]
    agent.learn(state, action=1, reward=10.0, next_state=next_state, done=True)
    after = agent.q_table[idx][1]
    assert after > before
    assert after == pytest.approx(0.5 * 10.0)  # alpha * (target - 0)


def test_epsilon_decays_and_floors():
    cfg = AgentConfig(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    agent = QLearningAgent(cfg)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)
    for _ in range(20):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)


def test_greedy_action_is_argmax():
    agent = QLearningAgent()
    state = np.array([50.0, 40.0, 1000.0])
    idx = agent.discretize(state)
    agent.q_table[idx] = np.array([1.0, 5.0, 2.0, 0.0])
    assert agent.get_action(state, explore=False) == 1


def test_save_and_load_roundtrip(tmp_path):
    agent = QLearningAgent()
    state = np.array([50.0, 40.0, 1000.0])
    agent.learn(state, 2, 7.0, state, True)
    path = tmp_path / "q.npy"
    agent.save(path)

    loaded = QLearningAgent.from_file(path)
    assert np.array_equal(loaded.q_table, agent.q_table)
    assert loaded.epsilon == loaded.cfg.epsilon_min


def test_load_shape_mismatch_raises(tmp_path):
    agent = QLearningAgent()
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError):
        agent.load(path)
