"""Train a ScaleRL agent (tabular Q-learning or DQN) and persist the policy.

Usage:
    python train.py                                  # tabular Q-learning, defaults
    python train.py --algo dqn --steps 200000
    python train.py --config configs/dqn.yaml        # config-driven experiment
    python train.py --no-plot --track                # skip plot, enable tracking

Outputs (under --model-dir, default ./artifacts):
    q_table.npy | dqn.pt   learned policy (algo-specific)
    model_meta.json        hyperparameters, seed, final metrics (reproducibility)
    <algo>_learning_curve.png   reward-per-episode curve (unless --no-plot)

If --track is set, per-run params/metrics are also written under ./runs/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scalerl.agent import QLearningAgent
from scalerl.config import AgentConfig, DQNConfig, EnvConfig, TrainConfig
from scalerl.dqn import DQNAgent
from scalerl.env import ScaleRLCacheEnv
from scalerl.experiment import ExperimentConfig, config_to_dict, load_experiment
from scalerl.tracking import ExperimentTracker


def _build_agent(algo: str, agent_cfg, dqn_cfg, env_cfg, rng):
    if algo == "qlearning":
        return QLearningAgent(agent_cfg, rng=rng)
    if algo == "dqn":
        return DQNAgent(dqn_cfg, env_cfg=env_cfg, rng=rng)
    raise ValueError(f"Unknown algo '{algo}'. Expected 'qlearning' or 'dqn'.")


def _model_path(train_cfg: TrainConfig) -> Path:
    model_dir = Path(train_cfg.model_dir)
    name = train_cfg.model_name if train_cfg.algo == "qlearning" else train_cfg.dqn_model_name
    return model_dir / name


def _agent_diag(agent, algo: str) -> dict:
    if algo == "qlearning":
        return {"q_table_density": round(agent.q_table_density, 4), "q_table_shape": list(agent.q_table.shape)}
    return {"num_parameters": agent.num_parameters}


def train(
    train_cfg: TrainConfig,
    env_cfg: EnvConfig,
    agent_cfg: AgentConfig,
    dqn_cfg: DQNConfig,
    make_plot: bool = True,
    tracker: ExperimentTracker | None = None,
) -> dict:
    rng = np.random.default_rng(train_cfg.seed)
    env = ScaleRLCacheEnv(env_cfg)
    agent = _build_agent(train_cfg.algo, agent_cfg, dqn_cfg, env_cfg, rng)

    state, _ = env.reset(seed=train_cfg.seed)

    episode_rewards: list[float] = []
    current_episode_reward = 0.0
    steps_done = 0
    start = time.time()

    while steps_done < train_cfg.total_steps:
        action = agent.get_action(state, explore=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.learn(state, action, reward, next_state, done)
        current_episode_reward += reward
        steps_done += 1

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            agent.decay_epsilon()
            state, _ = env.reset()
        else:
            state = next_state

        if steps_done % train_cfg.eval_every_steps == 0:
            recent = episode_rewards[-20:] if episode_rewards else [0.0]
            avg = float(np.mean(recent))
            print(
                f"[{train_cfg.algo:>9}][{steps_done:>7}/{train_cfg.total_steps}] "
                f"episodes={len(episode_rewards):>4} "
                f"epsilon={agent.epsilon:.3f} "
                f"avg_reward(last20)={avg:+.1f}"
            )
            if tracker is not None:
                tracker.log(steps_done, avg_reward_last20=avg, epsilon=agent.epsilon)

    elapsed = time.time() - start

    model_path = _model_path(train_cfg)
    agent.save(model_path)

    final_avg = float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0
    meta = {
        "version": 3,
        "algo": train_cfg.algo,
        "seed": train_cfg.seed,
        "total_steps": train_cfg.total_steps,
        "episodes": len(episode_rewards),
        "final_epsilon": round(agent.epsilon, 4),
        "final_avg_reward_last50": round(final_avg, 2),
        "train_seconds": round(elapsed, 1),
        "model_path": str(model_path),
        **_agent_diag(agent, train_cfg.algo),
        "agent_config": (agent_cfg.__dict__ if train_cfg.algo == "qlearning" else dqn_cfg.__dict__),
        "env_config": env_cfg.__dict__,
    }
    # Write both a stable name (last run) and an algo-specific name so the
    # tabular and DQN metadata coexist for the benchmark write-up.
    meta_json = json.dumps(meta, indent=2, default=str)
    (Path(train_cfg.model_dir) / train_cfg.metadata_name).write_text(
        meta_json, encoding="utf-8"
    )
    (Path(train_cfg.model_dir) / f"{train_cfg.algo}_meta.json").write_text(
        meta_json, encoding="utf-8"
    )

    print(f"\nSaved policy -> {model_path}")
    print(f"Trained {len(episode_rewards)} episodes in {elapsed:.1f}s "
          f"(final avg reward {final_avg:+.1f})")

    if make_plot:
        curve_path = Path(train_cfg.model_dir) / f"{train_cfg.algo}_learning_curve.png"
        _plot_learning_curve(episode_rewards, curve_path, train_cfg.algo)
        if tracker is not None:
            tracker.log_artifact(curve_path)

    if tracker is not None:
        tracker.finish(summary=meta)

    return meta


def _plot_learning_curve(episode_rewards: list[float], out_path: Path, algo: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping learning curve plot.")
        return

    if not episode_rewards:
        return

    rewards = np.array(episode_rewards)
    window = max(1, len(rewards) // 50)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(9, 5))
    plt.plot(rewards, color="#94a3b8", alpha=0.35, label="Episode reward")
    plt.plot(
        np.arange(window - 1, len(rewards)),
        smoothed,
        color="#10b981",
        linewidth=2.2,
        label=f"Moving avg (w={window})",
    )
    plt.title(f"ScaleRL Training Curve ({algo})")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved learning curve -> {out_path}")


def _configs_from_args(args: argparse.Namespace):
    if args.config:
        exp: ExperimentConfig = load_experiment(args.config)
        # CLI overrides on top of the config file.
        train_cfg = exp.train
        if args.steps is not None:
            train_cfg = TrainConfig(**{**train_cfg.__dict__, "total_steps": args.steps})
        if args.model_dir is not None:
            train_cfg = TrainConfig(**{**train_cfg.__dict__, "model_dir": args.model_dir})
        return train_cfg, exp.env, exp.agent, exp.dqn, exp.name

    train_cfg = TrainConfig(
        algo=args.algo,
        total_steps=args.steps if args.steps is not None else TrainConfig.total_steps,
        seed=args.seed,
        model_dir=args.model_dir or TrainConfig.model_dir,
    )
    return train_cfg, EnvConfig(), AgentConfig(), DQNConfig(), f"{args.algo}-cli"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a ScaleRL agent.")
    p.add_argument("--algo", choices=["qlearning", "dqn"], default="qlearning")
    p.add_argument("--config", type=str, default=None, help="YAML experiment config.")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--track", action="store_true", help="Log params/metrics under ./runs.")
    p.add_argument("--mlflow", action="store_true", help="Mirror tracking to MLflow if installed.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_cfg, env_cfg, agent_cfg, dqn_cfg, run_name = _configs_from_args(args)

    tracker = None
    if args.track or args.mlflow:
        params = {
            "algo": train_cfg.algo,
            "seed": train_cfg.seed,
            "total_steps": train_cfg.total_steps,
            "env": config_to_dict(env_cfg),
            "agent": config_to_dict(agent_cfg if train_cfg.algo == "qlearning" else dqn_cfg),
        }
        tracker = ExperimentTracker(run_name, params, use_mlflow=args.mlflow)

    train(
        train_cfg,
        env_cfg,
        agent_cfg,
        dqn_cfg,
        make_plot=not args.no_plot,
        tracker=tracker,
    )


if __name__ == "__main__":
    main()
