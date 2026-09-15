"""Benchmark the trained ScaleRL agent against baseline control policies.

Every policy is run on the SAME sequence of environment seeds, so the
comparison is controlled: differences come from decisions, not luck. All
reported numbers (mean latency, p99 latency, memory-overflow rate, reward)
are computed here from the simulation - none are hardcoded.

Usage:
    python evaluate.py                        # 200 episodes, default seed
    python evaluate.py --episodes 500 --seed 123
    python evaluate.py --no-plot

Outputs (under --out-dir, default ./artifacts):
    benchmark_results.json    machine-readable metrics per policy
    benchmark_results.csv     same, flat CSV
    benchmark_comparison.png  grouped bar chart (unless --no-plot)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scalerl.agent import QLearningAgent
from scalerl.baselines import IdlePolicy, ThresholdHeuristicPolicy
from scalerl.config import AgentConfig, DQNConfig, EnvConfig, TrainConfig
from scalerl.env import ScaleRLCacheEnv


def run_policy(
    policy,
    env_cfg: EnvConfig,
    episode_seeds: list[int],
) -> dict[str, float]:
    """Run one policy over a fixed set of episode seeds and collect metrics."""
    env = ScaleRLCacheEnv(env_cfg)
    latencies: list[float] = []
    memories: list[float] = []
    rewards: list[float] = []
    overflow_steps = 0
    total_steps = 0
    outages = 0

    for seed in episode_seeds:
        state, _ = env.reset(seed=seed)
        ep_reward = 0.0
        while True:
            action = policy.get_action(state, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            latencies.append(info["latency"])
            memories.append(info["memory"])
            if info["memory"] > env_cfg.memory_critical_threshold:
                overflow_steps += 1
            total_steps += 1
            ep_reward += reward
            if terminated:
                outages += 1
            if terminated or truncated:
                break
        rewards.append(ep_reward)

    lat = np.array(latencies)
    return {
        "mean_latency_ms": float(np.mean(lat)),
        "p50_latency_ms": float(np.percentile(lat, 50)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "mean_memory_pct": float(np.mean(memories)),
        "memory_overflow_rate_pct": float(100.0 * overflow_steps / total_steps),
        "outage_episodes": int(outages),
        "mean_episode_reward": float(np.mean(rewards)),
        "episodes": len(episode_seeds),
        "steps": total_steps,
    }


def evaluate(
    model_dir: Path,
    episodes: int,
    seed: int,
    env_cfg: EnvConfig,
    agent_cfg: AgentConfig,
    dqn_cfg: DQNConfig,
) -> dict[str, dict[str, float]]:
    seed_rng = np.random.default_rng(seed)
    episode_seeds = [int(x) for x in seed_rng.integers(0, 2**31 - 1, size=episodes)]

    learned: dict[str, object] = {}

    qtable_path = model_dir / TrainConfig.model_name
    if qtable_path.exists():
        learned["ScaleRL-QLearning"] = QLearningAgent.from_file(
            qtable_path, config=agent_cfg
        )
    else:
        print(f"NOTE: no tabular model at {qtable_path}; skipping ScaleRL-QLearning.")

    dqn_path = model_dir / TrainConfig.dqn_model_name
    if dqn_path.exists():
        from scalerl.dqn import DQNAgent

        learned["ScaleRL-DQN"] = DQNAgent.from_file(
            dqn_path, config=dqn_cfg, env_cfg=env_cfg
        )
    else:
        print(f"NOTE: no DQN model at {dqn_path}; skipping ScaleRL-DQN.")

    if not learned:
        print("WARNING: no trained models found. Run `python train.py` first.")

    policies: dict[str, object] = {
        **learned,
        "ThresholdHeuristic": ThresholdHeuristicPolicy(),
        "IdleNoOp": IdlePolicy(),
    }

    results: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        print(f"Evaluating {name} over {episodes} episodes ...")
        results[name] = run_policy(policy, env_cfg, episode_seeds)

    _print_table(results)
    _add_relative_improvement(results)
    return results


def _add_relative_improvement(results: dict[str, dict[str, float]]) -> None:
    """Add each learned policy's latency reduction vs the heuristic baseline."""
    base = results.get("ThresholdHeuristic")
    if base is None:
        return

    def pct_drop(policy: dict[str, float], metric: str) -> float:
        b = base[metric]
        return round(float(100.0 * (b - policy[metric]) / b), 2) if b else 0.0

    for name in ("ScaleRL-QLearning", "ScaleRL-DQN"):
        if name in results:
            results[name]["latency_reduction_vs_heuristic_pct"] = pct_drop(
                results[name], "mean_latency_ms"
            )
            results[name]["p99_reduction_vs_heuristic_pct"] = pct_drop(
                results[name], "p99_latency_ms"
            )


def _print_table(results: dict[str, dict[str, float]]) -> None:
    metrics = [
        "mean_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "mean_memory_pct",
        "memory_overflow_rate_pct",
        "mean_episode_reward",
    ]
    names = list(results.keys())
    print("\n" + "=" * 78)
    header = f"{'metric':<28}" + "".join(f"{n:>16}" for n in names)
    print(header)
    print("-" * 78)
    for m in metrics:
        row = f"{m:<28}" + "".join(f"{results[n][m]:>16.2f}" for n in names)
        print(row)
    print("=" * 78 + "\n")


def _write_outputs(
    results: dict[str, dict[str, float]], out_dir: Path, make_plot: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Flat CSV.
    all_keys = sorted({k for r in results.values() for k in r})
    with (out_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy"] + all_keys)
        for name, r in results.items():
            writer.writerow([name] + [r.get(k, "") for k in all_keys])

    print(f"Saved -> {out_dir / 'benchmark_results.json'}")
    print(f"Saved -> {out_dir / 'benchmark_results.csv'}")

    if make_plot:
        _plot(results, out_dir / "benchmark_comparison.png")


def _plot(results: dict[str, dict[str, float]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping comparison plot.")
        return

    names = list(results.keys())
    metrics = ["mean_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(names))

    plt.figure(figsize=(9, 5))
    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        plt.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)])
    plt.xticks(x + width * (len(names) - 1) / 2, ["mean", "p95", "p99"])
    plt.ylabel("Latency (ms)")
    plt.title("Latency: ScaleRL vs Baselines (lower is better)")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved -> {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ScaleRL vs baselines.")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--model-dir", type=str, default=TrainConfig.model_dir)
    p.add_argument("--out-dir", type=str, default=TrainConfig.model_dir)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results = evaluate(
        model_dir=Path(args.model_dir),
        episodes=args.episodes,
        seed=args.seed,
        env_cfg=EnvConfig(),
        agent_cfg=AgentConfig(),
        dqn_cfg=DQNConfig(),
    )
    _write_outputs(results, Path(args.out_dir), make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
