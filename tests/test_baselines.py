"""Tests for baseline policies and the LRU cache simulator."""

import numpy as np

from scalerl.baselines import IdlePolicy, LRUCacheSimulator, ThresholdHeuristicPolicy


def test_idle_policy_always_zero():
    p = IdlePolicy()
    assert p.get_action(np.array([500.0, 95.0, 9000.0])) == 0


def test_threshold_heuristic_actions():
    p = ThresholdHeuristicPolicy(
        soft_evict_memory=75.0, aggressive_evict_memory=88.0, reroute_requests=6000.0
    )
    assert p.get_action(np.array([20.0, 30.0, 1000.0])) == 0   # idle
    assert p.get_action(np.array([20.0, 80.0, 1000.0])) == 1   # soft evict
    assert p.get_action(np.array([20.0, 90.0, 1000.0])) == 2   # aggressive
    assert p.get_action(np.array([20.0, 40.0, 7000.0])) == 3   # reroute


def test_lru_hit_and_eviction():
    cache = LRUCacheSimulator(capacity=2)
    assert cache.access(1) is False  # miss
    assert cache.access(2) is False  # miss
    assert cache.access(1) is True   # hit, 1 now most-recent
    cache.access(3)                  # cache now {1, 3}; evicts 2 (LRU), keeps 1
    assert cache.access(3) is True   # 3 survived
    assert cache.access(2) is False  # 2 was evicted -> miss


def test_lru_hit_rate():
    cache = LRUCacheSimulator(capacity=10)
    for _ in range(2):
        for item in range(5):
            cache.access(item)
    # 5 misses first pass, 5 hits second pass -> 50%.
    assert cache.hit_rate == 50.0


def test_lru_reset():
    cache = LRUCacheSimulator(capacity=3)
    cache.access(1)
    cache.reset()
    assert cache.hits == 0 and cache.misses == 0
    assert len(cache.cache) == 0
