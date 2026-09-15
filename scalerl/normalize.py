"""Observation normalization for neural-network agents.

The tabular agent works fine on raw values (it just bins them), but a neural
network trains far more stably when inputs are on a comparable scale. This
module maps the raw [latency, memory, requests] observation into ~[0, 1] using
the environment's declared bounds.
"""

from __future__ import annotations

import numpy as np

from scalerl.config import EnvConfig, observation_bounds


class ObservationNormalizer:
    def __init__(self, env_cfg: EnvConfig | None = None):
        bounds = observation_bounds(env_cfg or EnvConfig())
        self.low = np.array([b[0] for b in bounds], dtype=np.float32)
        self.high = np.array([b[1] for b in bounds], dtype=np.float32)
        self._range = np.maximum(self.high - self.low, 1e-8)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        scaled = (state - self.low) / self._range
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)
