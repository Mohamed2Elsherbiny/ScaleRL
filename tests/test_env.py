"""Tests for the ScaleRLCacheEnv Gymnasium environment."""

import numpy as np
import pytest

from scalerl.config import EnvConfig
from scalerl.env import ScaleRLCacheEnv


def test_reset_returns_obs_and_info():
    env = ScaleRLCacheEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    assert set(info) >= {"latency", "memory", "requests", "step"}
    assert info["step"] == 0


def test_step_returns_five_tuple():
    env = ScaleRLCacheEnv()
    env.reset(seed=0)
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (3,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_state_stays_within_bounds():
    cfg = EnvConfig()
    env = ScaleRLCacheEnv(cfg)
    env.reset(seed=1)
    for _ in range(cfg.max_steps):
        action = np.random.randint(4)
        obs, _, terminated, truncated, _ = env.step(action)
        assert cfg.latency_min <= obs[0] <= cfg.latency_max
        assert cfg.memory_min <= obs[1] <= cfg.memory_max
        assert cfg.request_min <= obs[2] <= cfg.request_max
        if terminated or truncated:
            break


def test_truncation_at_max_steps():
    cfg = EnvConfig(max_steps=10, outage_latency=10_000)  # avoid early termination
    env = ScaleRLCacheEnv(cfg)
    env.reset(seed=2)
    truncated = False
    for _ in range(cfg.max_steps):
        _, _, terminated, truncated, _ = env.step(1)  # soft evict keeps latency sane
        if terminated or truncated:
            break
    assert truncated is True


def test_invalid_action_raises():
    env = ScaleRLCacheEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_reset_is_deterministic_with_seed():
    env = ScaleRLCacheEnv()
    env.reset(seed=123)
    seq_a = [env.step(0)[1] for _ in range(20)]
    env.reset(seed=123)
    seq_b = [env.step(0)[1] for _ in range(20)]
    assert seq_a == seq_b


def test_aggressive_eviction_reduces_memory():
    env = ScaleRLCacheEnv()
    env.reset(seed=5)
    # Drive memory up by idling under load first.
    for _ in range(10):
        env.step(0)
    mem_before = env.memory_util
    env.step(2)  # aggressive evict
    assert env.memory_util < mem_before


def test_memory_overflow_incurs_penalty():
    cfg = EnvConfig()
    env = ScaleRLCacheEnv(cfg)
    env.reset(seed=0)
    env.memory_util = 95.0  # above critical threshold
    penalized = env._reward(action_cost=0.0)
    env.memory_util = 50.0  # safe
    safe = env._reward(action_cost=0.0)
    assert penalized < safe
