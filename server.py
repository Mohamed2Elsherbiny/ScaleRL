"""FastAPI inference gateway for ScaleRL.

Loads a pre-trained Q-table from disk at startup (produced by ``train.py``)
instead of training in-process. This makes the service:

* fast to start and stateless-ish (the policy is a versioned artifact),
* reproducible (same artifact -> same decisions),
* honest about training vs serving (training is an offline concern).

Run:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from scalerl.agent import QLearningAgent
from scalerl.config import ACTION_NAMES, ACTION_RECOMMENDATIONS, AgentConfig, TrainConfig

MODEL_PATH = Path(TrainConfig.model_dir) / TrainConfig.model_name
_DASHBOARD_PATH = Path(__file__).with_name("scalerl_dashboard.html")

app = FastAPI(
    title="ScaleRL Distributed Infrastructure Gateway",
    description="Real-time RL inference for cache eviction and traffic routing.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated at startup.
_agent: QLearningAgent | None = None
_model_loaded = False


class ServerStatePayload(BaseModel):
    latency: float = Field(..., ge=0, description="System latency in ms")
    memory_utilization: float = Field(..., ge=0, le=100, description="Memory %")
    request_volume: float = Field(..., ge=0, description="Requests per second")


class ActionResponse(BaseModel):
    action_id: int
    action_name: str
    recommendation: str
    inference_ms: float


@app.on_event("startup")
def load_policy() -> None:
    """Load the trained Q-table. The service starts even if it is missing,
    but inference will report the model as unavailable until trained."""
    global _agent, _model_loaded
    _agent = QLearningAgent(AgentConfig())
    if MODEL_PATH.exists():
        try:
            _agent.load(MODEL_PATH)
            _model_loaded = True
            print(f"Loaded trained policy from {MODEL_PATH}")
        except Exception as exc:  # noqa: BLE001
            _model_loaded = False
            print(f"Failed to load policy ({exc}); serving untrained agent.")
    else:
        _model_loaded = False
        print(
            f"No trained policy at {MODEL_PATH}. Run `python train.py` first. "
            "Serving an untrained agent (decisions will be poor)."
        )


@app.post("/api/v1/inference", response_model=ActionResponse)
def get_eviction_decision(state: ServerStatePayload) -> ActionResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    state_vector = np.array(
        [state.latency, state.memory_utilization, state.request_volume],
        dtype=np.float32,
    )
    start = time.perf_counter()
    action = _agent.get_action(state_vector, explore=False)
    inference_ms = (time.perf_counter() - start) * 1000.0

    return ActionResponse(
        action_id=action,
        action_name=ACTION_NAMES[action],
        recommendation=ACTION_RECOMMENDATIONS[action],
        inference_ms=round(inference_ms, 4),
    )


@app.get("/api/v1/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "model_loaded": _model_loaded,
        "model_path": str(MODEL_PATH),
        "q_table_density": (
            float(_agent.q_table_density) if _agent is not None else 0.0
        ),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard_root() -> HTMLResponse:
    if not _DASHBOARD_PATH.exists():
        return HTMLResponse(
            content="<h3>scalerl_dashboard.html not found</h3>", status_code=404
        )
    return HTMLResponse(content=_DASHBOARD_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
