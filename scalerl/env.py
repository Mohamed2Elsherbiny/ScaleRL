"""Gymnasium environment simulating a distributed cache/traffic node.

State (continuous, Box):
    [0] System latency (ms)
    [1] Memory utilization (%)
    [2] Request volume (RPS)

Actions (discrete):
    0: IDLE_STANDBY            - let default cache rules run
    1: SOFT_EVICTION           - drop cold tier, small latency cost
    2: AGGRESSIVE_EVICTION     - drop warm+cold tiers, larger latency cost
    3: DYNAMIC_TRAFFIC_REROUTE - offload 25% of traffic to a backup node

This is a real ``gymnasium.Env`` (reset -> (obs, info), step -> 5-tuple),
so it plugs into standard RL tooling and the Gymnasium API checker.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # Prefer real Gymnasium; degrade gracefully if unavailable.
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without gymnasium
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _GYM_AVAILABLE = False

from scalerl.config import EnvConfig

_Base = gym.Env if _GYM_AVAILABLE else object


class ScaleRLCacheEnv(_Base):  # type: ignore[misc]
    """Distributed server node modeled as an MDP."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()

        if _GYM_AVAILABLE:
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(
                low=np.array(
                    [self.cfg.latency_min, self.cfg.memory_min, self.cfg.request_min],
                    dtype=np.float32,
                ),
                high=np.array(
                    [self.cfg.latency_max, self.cfg.memory_max, self.cfg.request_max],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )

        self._rng = np.random.default_rng()
        self.latency = self.cfg.init_latency
        self.memory_util = self.cfg.init_memory
        self.request_vol = self.cfg.init_request_vol
        self.step_count = 0

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            if _GYM_AVAILABLE:
                super().reset(seed=seed)

        self.latency = self.cfg.init_latency
        self.memory_util = self.cfg.init_memory
        self.request_vol = self.cfg.init_request_vol
        self.step_count = 0
        return self._get_state(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.cfg
        self.step_count += 1

        # Traffic fluctuates each step.
        traffic_growth = self._rng.normal(
            cfg.traffic_growth_mean, cfg.traffic_growth_std
        )
        self.request_vol = float(
            np.clip(self.request_vol + traffic_growth, cfg.request_min, cfg.request_max)
        )

        # Latency grows super-linearly with request volume.
        base_latency = (self.request_vol / 200.0) ** 1.5 + 15.0

        # Memory drifts up with load unless an eviction action fires.
        memory_delta = (self.request_vol / 2500.0) - 1.5
        self.memory_util = float(
            np.clip(
                self.memory_util + memory_delta, cfg.memory_min, cfg.memory_max
            )
        )

        action_cost = 0.0
        if action == 0:  # IDLE_STANDBY
            if self.memory_util > 85.0:
                base_latency += (self.memory_util - 85.0) * 15.0
        elif action == 1:  # SOFT_EVICTION
            self.memory_util = float(
                np.clip(self.memory_util - 15.0, cfg.memory_min, cfg.memory_max)
            )
            base_latency += 5.0
            action_cost = -1.0
        elif action == 2:  # AGGRESSIVE_EVICTION
            self.memory_util = float(
                np.clip(self.memory_util - 35.0, cfg.memory_min, cfg.memory_max)
            )
            base_latency += 18.0
            action_cost = -3.0
        elif action == 3:  # DYNAMIC_TRAFFIC_REROUTE
            self.request_vol = float(
                np.clip(self.request_vol * 0.75, cfg.request_min, cfg.request_max)
            )
            base_latency += 8.0
            action_cost = -5.0
        else:
            raise ValueError(f"Invalid action {action}; expected 0-3.")

        # Natural jitter.
        self.latency = float(
            np.clip(
                base_latency + self._rng.normal(0, cfg.latency_jitter_std),
                cfg.latency_min,
                cfg.latency_max,
            )
        )

        reward = self._reward(action_cost)

        terminated = bool(self.latency > cfg.outage_latency)
        truncated = bool(self.step_count >= cfg.max_steps)

        return self._get_state(), reward, terminated, truncated, self._info()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _reward(self, action_cost: float) -> float:
        cfg = self.cfg
        latency_penalty = -((self.latency / cfg.latency_penalty_divisor) ** cfg.latency_penalty_exponent)
        memory_penalty = (
            -cfg.memory_critical_penalty
            if self.memory_util > cfg.memory_critical_threshold
            else 0.0
        )
        throughput_reward = (
            self.request_vol / cfg.throughput_reward_divisor
        ) * cfg.throughput_reward_scale
        return float(latency_penalty + memory_penalty + throughput_reward + action_cost)

    def _get_state(self) -> np.ndarray:
        return np.array(
            [self.latency, self.memory_util, self.request_vol], dtype=np.float32
        )

    def _info(self) -> dict[str, Any]:
        return {
            "latency": self.latency,
            "memory": self.memory_util,
            "requests": self.request_vol,
            "step": self.step_count,
        }
