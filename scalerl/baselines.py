"""Baseline policies used to benchmark the RL agent.

Two kinds of baseline are provided:

* ``ThresholdHeuristicPolicy`` - a hand-tuned rule-based controller that
  acts on the *same* environment observations as the RL agent, giving a
  fair like-for-like comparison of decision quality.
* ``LRUCacheSimulator`` - a classic Least-Recently-Used cache, included to
  represent the traditional "static policy" the README contrasts against.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class ThresholdHeuristicPolicy:
    """Rule-based controller over (latency, memory, request_volume).

    Encodes the "obvious" operational playbook so we can show the RL agent
    learns timing/tradeoffs a static ruleset misses.
    """

    def __init__(
        self,
        soft_evict_memory: float = 75.0,
        aggressive_evict_memory: float = 88.0,
        reroute_requests: float = 6000.0,
    ):
        self.soft_evict_memory = soft_evict_memory
        self.aggressive_evict_memory = aggressive_evict_memory
        self.reroute_requests = reroute_requests

    def get_action(self, state: np.ndarray, explore: bool = False) -> int:
        _, memory, requests = float(state[0]), float(state[1]), float(state[2])
        if memory >= self.aggressive_evict_memory:
            return 2  # AGGRESSIVE_EVICTION
        if requests >= self.reroute_requests:
            return 3  # DYNAMIC_TRAFFIC_REROUTE
        if memory >= self.soft_evict_memory:
            return 1  # SOFT_EVICTION
        return 0  # IDLE_STANDBY


class IdlePolicy:
    """Do-nothing control (never intervenes) - a lower bound reference."""

    def get_action(self, state: np.ndarray, explore: bool = False) -> int:
        return 0


class LRUCacheSimulator:
    """Standard LRU cache used to report a hit-rate baseline."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: OrderedDict[int, None] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, item: int) -> bool:
        if item in self.cache:
            self.cache.move_to_end(item)
            self.hits += 1
            return True
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # evict least-recently-used
        self.cache[item] = None
        self.misses += 1
        return False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 100.0

    def reset(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0
