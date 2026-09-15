"""Central configuration for the ScaleRL environment, agent, and training.

Keeping every tunable in one dataclass module means experiments are
reproducible: a run is fully described by these values plus a seed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Parameters describing the simulated distributed cache node."""

    cache_capacity: int = 1000
    max_steps: int = 500

    # Initial (healthy) state
    init_latency: float = 20.0        # ms
    init_memory: float = 30.0         # %
    init_request_vol: float = 500.0   # RPS

    # State bounds
    latency_min: float = 5.0
    latency_max: float = 1000.0
    memory_min: float = 10.0
    memory_max: float = 100.0
    request_min: float = 100.0
    request_max: float = 10000.0

    # Traffic dynamics
    traffic_growth_mean: float = 50.0
    traffic_growth_std: float = 150.0
    latency_jitter_std: float = 5.0

    # Reward shaping
    memory_critical_threshold: float = 90.0
    memory_critical_penalty: float = 50.0
    latency_penalty_divisor: float = 40.0
    latency_penalty_exponent: float = 1.8
    throughput_reward_divisor: float = 1000.0
    throughput_reward_scale: float = 2.0

    # Termination
    outage_latency: float = 900.0


@dataclass(frozen=True)
class AgentConfig:
    """Tabular Q-learning agent hyperparameters and discretization grid."""

    action_size: int = 4
    gamma: float = 0.95
    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    # Discretization bins. Upper bounds match the env state bounds so the
    # whole reachable state space is covered (the original code capped
    # latency at 500 while latency could reach 1000, leaving high-latency
    # states unlearnable).
    latency_bins: int = 12
    memory_bins: int = 10
    request_bins: int = 10
    latency_bin_max: float = 1000.0
    memory_bin_max: float = 100.0
    request_bin_max: float = 10000.0


@dataclass(frozen=True)
class DQNConfig:
    """Deep Q-Network agent hyperparameters.

    The state space is tiny (3 dims), so a small MLP is plenty. Standard DQN
    machinery is included (experience replay, a target network synced
    periodically, Huber loss) because that is what makes DQN stable and is
    the point of comparing it against the tabular agent.
    """

    action_size: int = 4
    obs_dim: int = 3
    hidden_sizes: tuple[int, ...] = (64, 64)
    gamma: float = 0.95
    learning_rate: float = 1e-3

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995  # applied per-episode, matching the tabular agent

    buffer_size: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 1_000        # fill replay before learning
    train_every: int = 4             # gradient step frequency (env steps)
    target_sync_every: int = 500     # hard target-network sync (env steps)
    grad_clip_norm: float = 10.0


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop configuration."""

    algo: str = "qlearning"          # "qlearning" | "dqn"
    total_steps: int = 400_000
    seed: int = 42
    eval_every_steps: int = 10_000
    model_dir: str = "artifacts"
    model_name: str = "q_table.npy"
    dqn_model_name: str = "dqn.pt"
    metadata_name: str = "model_meta.json"
    curve_name: str = "learning_curve.png"


def observation_bounds(env_cfg: EnvConfig) -> tuple[tuple[float, float], ...]:
    """Per-dimension (min, max) bounds used to normalize observations."""
    return (
        (env_cfg.latency_min, env_cfg.latency_max),
        (env_cfg.memory_min, env_cfg.memory_max),
        (env_cfg.request_min, env_cfg.request_max),
    )


ACTION_NAMES: list[str] = [
    "IDLE_STANDBY",
    "SOFT_EVICTION",
    "AGGRESSIVE_EVICTION",
    "DYNAMIC_TRAFFIC_REROUTE",
]

ACTION_RECOMMENDATIONS: list[str] = [
    "System within nominal boundaries. Maintain standard cache policies.",
    "Cold data evictions triggered. Clearing Tier 3 cache pipelines.",
    "High load warning. Triggering immediate core memory cleanup blocks.",
    "Traffic limits breached. Directing 25% overflow capacity to target backends.",
]
