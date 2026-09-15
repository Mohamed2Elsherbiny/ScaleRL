"""Tests for the DQN agent, replay buffer, and observation normalizer."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scalerl.config import DQNConfig, EnvConfig
from scalerl.dqn import DQNAgent, QNetwork, ReplayBuffer
from scalerl.normalize import ObservationNormalizer


def _tiny_cfg(**overrides):
    base = dict(
        hidden_sizes=(16,),
        batch_size=8,
        warmup_steps=8,
        buffer_size=1000,
        train_every=1,
        target_sync_every=10,
    )
    base.update(overrides)
    return DQNConfig(**base)


def test_qnetwork_output_shape():
    net = QNetwork(obs_dim=3, action_size=4, hidden_sizes=(16, 16))
    out = net(torch.zeros(5, 3))
    assert out.shape == (5, 4)


def test_normalizer_scales_to_unit_range():
    norm = ObservationNormalizer(EnvConfig())
    low = norm(np.array([5.0, 10.0, 100.0]))     # all mins
    high = norm(np.array([1000.0, 100.0, 10000.0]))  # all maxes
    assert np.allclose(low, 0.0, atol=1e-5)
    assert np.allclose(high, 1.0, atol=1e-5)
    # Out-of-range values are clipped, not extrapolated.
    assert np.all(norm(np.array([99999.0, 999.0, 99999.0])) <= 1.0)


def test_replay_buffer_push_and_sample():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=100, rng=rng)
    for i in range(20):
        buf.push(np.zeros(3), i % 4, float(i), np.ones(3), i % 2 == 0)
    assert len(buf) == 20
    s, a, r, ns, d = buf.sample(8)
    assert s.shape == (8, 3)
    assert a.shape == (8,)
    assert r.shape == (8,)
    assert ns.shape == (8, 3)
    assert d.shape == (8,)


def test_replay_buffer_respects_capacity():
    buf = ReplayBuffer(capacity=5)
    for _ in range(10):
        buf.push(np.zeros(3), 0, 0.0, np.zeros(3), False)
    assert len(buf) == 5


def test_get_action_greedy_is_deterministic():
    agent = DQNAgent(_tiny_cfg())
    state = np.array([50.0, 40.0, 1000.0])
    a1 = agent.get_action(state, explore=False)
    a2 = agent.get_action(state, explore=False)
    assert a1 == a2
    assert 0 <= a1 < 4


def test_learn_triggers_optimization_after_warmup():
    agent = DQNAgent(_tiny_cfg(warmup_steps=8, batch_size=8, train_every=1))
    state = np.array([50.0, 40.0, 1000.0])
    for _ in range(20):
        agent.learn(state, 1, -1.0, state, False)
    assert agent._learn_steps > 0  # gradient steps happened


def test_save_and_load_roundtrip(tmp_path):
    cfg = _tiny_cfg()
    agent = DQNAgent(cfg)
    state = np.array([50.0, 40.0, 1000.0])
    for _ in range(30):
        agent.learn(state, 2, 1.0, state, False)

    path = tmp_path / "dqn.pt"
    agent.save(path)

    loaded = DQNAgent.from_file(path, config=cfg)
    # Same network weights -> same greedy decision on a fixed input.
    assert loaded.get_action(state, explore=False) == agent.get_action(
        state, explore=False
    )


def test_num_parameters_positive():
    agent = DQNAgent(_tiny_cfg())
    assert agent.num_parameters > 0
