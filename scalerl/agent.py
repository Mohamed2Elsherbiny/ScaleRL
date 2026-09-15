"""Tabular Q-learning agent with state discretization.

Chosen deliberately over a deep network: the state space is small (3 dims),
inference is a single array lookup (sub-millisecond, no GPU), and the learned
policy is fully inspectable - all useful properties for an always-on control
loop embedded in infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scalerl.config import AgentConfig


class QLearningAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.cfg = config or AgentConfig()
        self._rng = rng or np.random.default_rng()
        self.epsilon = self.cfg.epsilon_start

        # Bin edges. digitize returns indices in [0, len(edges)], so the
        # table needs len(edges)+1 slots per dimension.
        self.latency_edges = np.linspace(0, self.cfg.latency_bin_max, self.cfg.latency_bins)
        self.memory_edges = np.linspace(0, self.cfg.memory_bin_max, self.cfg.memory_bins)
        self.request_edges = np.linspace(0, self.cfg.request_bin_max, self.cfg.request_bins)

        self.q_table = np.zeros(
            (
                len(self.latency_edges) + 1,
                len(self.memory_edges) + 1,
                len(self.request_edges) + 1,
                self.cfg.action_size,
            ),
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    def discretize(self, state: np.ndarray) -> tuple[int, int, int]:
        lat, mem, req = float(state[0]), float(state[1]), float(state[2])
        return (
            int(np.digitize(lat, self.latency_edges)),
            int(np.digitize(mem, self.memory_edges)),
            int(np.digitize(req, self.request_edges)),
        )

    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() <= self.epsilon:
            return int(self._rng.integers(self.cfg.action_size))
        idx = self.discretize(state)
        return int(np.argmax(self.q_table[idx]))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        s = self.discretize(state)
        ns = self.discretize(next_state)

        best_next = np.max(self.q_table[ns])
        td_target = reward if done else reward + self.cfg.gamma * best_next
        td_error = td_target - self.q_table[s][action]
        self.q_table[s][action] += self.cfg.alpha * td_error

    def decay_epsilon(self) -> None:
        if self.epsilon > self.cfg.epsilon_min:
            self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    @property
    def q_table_density(self) -> float:
        """Fraction of state-action cells that have been updated."""
        return float(np.count_nonzero(self.q_table) / self.q_table.size)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        loaded = np.load(path)
        if loaded.shape != self.q_table.shape:
            raise ValueError(
                f"Q-table shape mismatch: file {loaded.shape} vs agent {self.q_table.shape}. "
                "Config and saved model are incompatible."
            )
        self.q_table = loaded
        self.epsilon = self.cfg.epsilon_min  # trained model exploits

    @classmethod
    def from_file(
        cls, path: str | Path, config: AgentConfig | None = None
    ) -> QLearningAgent:
        agent = cls(config=config)
        agent.load(path)
        return agent
