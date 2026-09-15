"""Config-driven experiments: build dataclass configs from YAML.

A run is then fully described by a single YAML file plus a seed, which makes
experiments declarative and reproducible. Example:

    algo: dqn
    seed: 7
    total_steps: 300000
    env:
      max_steps: 500
    dqn:
      hidden_sizes: [128, 128]
      learning_rate: 0.0005
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from scalerl.config import AgentConfig, DQNConfig, EnvConfig, TrainConfig


def _build(cls, overrides: dict[str, Any] | None):
    """Instantiate a frozen dataclass, applying only known keys."""
    if not overrides:
        return cls()
    valid = {f.name for f in fields(cls)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = dict(overrides)
    # Normalize list -> tuple for tuple-typed fields (YAML has no tuples).
    for f in fields(cls):
        if f.name in kwargs and isinstance(kwargs[f.name], list):
            kwargs[f.name] = tuple(kwargs[f.name])
    return cls(**kwargs)


@dataclasses.dataclass
class ExperimentConfig:
    algo: str
    seed: int
    train: TrainConfig
    env: EnvConfig
    agent: AgentConfig
    dqn: DQNConfig
    name: str = "experiment"


def load_experiment(path: str | Path) -> ExperimentConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    algo = raw.get("algo", "qlearning")
    seed = int(raw.get("seed", TrainConfig.seed))
    name = raw.get("name", Path(path).stem)

    env = _build(EnvConfig, raw.get("env"))
    agent = _build(AgentConfig, raw.get("agent"))
    dqn = _build(DQNConfig, raw.get("dqn"))

    # Top-level training keys (algo/seed/total_steps/...) feed TrainConfig.
    train_overrides = dict(raw.get("train", {}))
    for key in ("total_steps", "eval_every_steps", "model_dir"):
        if key in raw:
            train_overrides[key] = raw[key]
    train_overrides["algo"] = algo
    train_overrides["seed"] = seed
    train = _build(TrainConfig, train_overrides)

    return ExperimentConfig(
        algo=algo, seed=seed, train=train, env=env, agent=agent, dqn=dqn, name=name
    )


def config_to_dict(cfg) -> dict[str, Any]:
    """Serialize a (possibly nested) dataclass config to a JSON-friendly dict."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, tuple):
        return list(cfg)
    return cfg
