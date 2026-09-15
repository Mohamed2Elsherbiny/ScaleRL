"""Deep Q-Network agent.

Provides the same interface as ``QLearningAgent`` (``get_action``, ``learn``,
``decay_epsilon``, ``save``/``load``/``from_file``) so the training loop,
evaluation harness, and server can treat the two interchangeably. This is the
point of Tier 2: a real head-to-head between tabular Q-learning and DQN on the
identical environment.

Standard DQN components:
    * MLP Q-network Q(s, .) -> action values
    * experience replay (decorrelates samples)
    * a target network synced periodically (stabilizes the TD target)
    * Huber loss + gradient clipping
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scalerl.config import DQNConfig, EnvConfig
from scalerl.normalize import ObservationNormalizer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_size: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator | None = None):
        self.buffer: deque = deque(maxlen=capacity)
        self._rng = rng or np.random.default_rng()

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        config: DQNConfig | None = None,
        env_cfg: EnvConfig | None = None,
        rng: np.random.Generator | None = None,
        device: str | None = None,
    ):
        self.cfg = config or DQNConfig()
        self._rng = rng or np.random.default_rng()
        self.device = torch.device(device or "cpu")
        self.epsilon = self.cfg.epsilon_start

        self.normalizer = ObservationNormalizer(env_cfg)

        self.q_net = QNetwork(
            self.cfg.obs_dim, self.cfg.action_size, self.cfg.hidden_sizes
        ).to(self.device)
        self.target_net = QNetwork(
            self.cfg.obs_dim, self.cfg.action_size, self.cfg.hidden_sizes
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.cfg.learning_rate
        )
        self.replay = ReplayBuffer(self.cfg.buffer_size, rng=self._rng)

        self._learn_steps = 0
        self._env_steps = 0

    # ------------------------------------------------------------------ #
    # Interface shared with QLearningAgent
    # ------------------------------------------------------------------ #
    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() <= self.epsilon:
            return int(self._rng.integers(self.cfg.action_size))
        obs = self.normalizer(state)
        with torch.no_grad():
            t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            q = self.q_net(t)
        return int(torch.argmax(q, dim=1).item())

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store the transition and take a gradient step on the configured
        cadence. Signature matches the tabular agent so the training loop is
        identical for both."""
        self.replay.push(
            self.normalizer(state), action, reward, self.normalizer(next_state), done
        )
        self._env_steps += 1

        if len(self.replay) < max(self.cfg.warmup_steps, self.cfg.batch_size):
            return
        if self._env_steps % self.cfg.train_every == 0:
            self._optimize()
        if self._env_steps % self.cfg.target_sync_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        if self.epsilon > self.cfg.epsilon_min:
            self.epsilon = max(
                self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay
            )

    # ------------------------------------------------------------------ #
    def _optimize(self) -> None:
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.cfg.batch_size
        )
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.cfg.gamma * next_q

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self._learn_steps += 1

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "config": self.cfg.__dict__,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.epsilon = self.cfg.epsilon_min  # trained model exploits
        self.q_net.eval()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        config: DQNConfig | None = None,
        env_cfg: EnvConfig | None = None,
    ) -> DQNAgent:
        agent = cls(config=config, env_cfg=env_cfg)
        agent.load(path)
        return agent

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.q_net.parameters())
