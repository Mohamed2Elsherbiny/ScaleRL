"""Lightweight experiment tracking.

Writes hyperparameters and scalar metrics to a per-run directory so training
runs are auditable and comparable without any external service:

    runs/<timestamp>-<name>/
        params.json      full config + seed
        metrics.jsonl    one JSON object per logged step

If ``mlflow`` is installed and ``use_mlflow=True``, metrics/params are also
mirrored to MLflow. The JSONL log always works, so CI and offline runs need no
extra infrastructure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentTracker:
    def __init__(
        self,
        run_name: str,
        params: dict[str, Any],
        base_dir: str | Path = "runs",
        use_mlflow: bool = False,
    ):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(base_dir) / f"{ts}-{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.run_dir / "metrics.jsonl"

        (self.run_dir / "params.json").write_text(
            json.dumps(params, indent=2, default=str), encoding="utf-8"
        )

        self._mlflow = None
        if use_mlflow:
            try:
                import mlflow

                self._mlflow = mlflow
                mlflow.start_run(run_name=run_name)
                mlflow.log_params(_flatten(params))
            except ImportError:
                print("mlflow not installed; falling back to JSONL logging only.")

    def log(self, step: int, **metrics: float) -> None:
        record = {"step": step, **metrics}
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self._mlflow is not None:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items()}, step=step
            )

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if summary:
            (self.run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        if self._mlflow is not None:
            self._mlflow.end_run()


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out
