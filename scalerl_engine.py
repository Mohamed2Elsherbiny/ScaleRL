"""
ScaleRL: Distributed RL-Based Cache Eviction and Traffic Engine
Contains:
1. Gymnasium Environment (ScaleRLCacheEnv)
2. Q-Learning Agent (Discretized State-Space Policy)
3. LRU Baseline Cache Simulator
4. FastAPI Inference and Monitoring Server
5. Automated Simulation Script
"""

import os
import random
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import uvicorn
from pydantic import BaseModel

# Gymnasium fallback to standard Gym structure for simplicity and compatibility
class ScaleRLCacheEnv:
    """
    Gymnasium-style environment representing a distributed server node.
    State Space (Continuous):
        1. System Latency (ms): [0.0 to 1000.0]
        2. Memory Utilization (%): [0.0 to 100.0]
        3. Request Volume (RPS): [0.0 to 10000.0]
    
    Action Space (Discrete):
        0: Idle (Do nothing, let default cache rules run)
        1: Soft Evict (Remove cold tier assets, lower latency impact)
        2: Aggressive Evict (Clear warm & cold tiers, high immediate memory recovery)
        3: Dynamic Route (Offload 25% traffic to adjacent backup server nodes)
    """
    def __init__(self, cache_capacity: int = 1000):
        self.capacity = cache_capacity
        self.reset()

    def reset(self) -> np.ndarray:
        # Initialize to a healthy starting state
        self.latency = 20.0        # ms
        self.memory_util = 30.0    # %
        self.request_vol = 500.0   # RPS
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array([self.latency, self.memory_util, self.request_vol], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # State transitions based on traffic fluctuation and action taken
        traffic_growth = np.random.normal(50, 150)
        self.request_vol = np.clip(self.request_vol + traffic_growth, 100, 10000)
        
        # Base latency increases with request volume
        base_latency = (self.request_vol / 200.0) ** 1.5 + 15.0
        
        # Memory consumption moves up with request volume unless evicted
        memory_delta = (self.request_vol / 2500.0) - 1.5
        self.memory_util = np.clip(self.memory_util + memory_delta, 10.0, 100.0)

        # Apply Agent Actions
        action_cost = 0.0
        if action == 0:  # Do nothing
            # Latency spikes heavily if memory limit is breached
            if self.memory_util > 85.0:
                base_latency += (self.memory_util - 85.0) * 15.0
        
        elif action == 1:  # Soft Evict
            # Lowers memory utilization slightly, small temporary latency spike
            self.memory_util = np.clip(self.memory_util - 15.0, 10.0, 100.0)
            base_latency += 5.0
            action_cost = -1.0
            
        elif action == 2:  # Aggressive Evict
            # Drops memory significantly, larger temporary block latency spike
            self.memory_util = np.clip(self.memory_util - 35.0, 10.0, 100.0)
            base_latency += 18.0
            action_cost = -3.0
            
        elif action == 3:  # Dynamic Traffic Route
            # Immediately drops incoming load, but introduces remote network overhead cost
            self.request_vol = np.clip(self.request_vol * 0.75, 100, 10000)
            base_latency += 8.0
            action_cost = -5.0

        # Incorporate natural jitter
        self.latency = np.clip(base_latency + np.random.normal(0, 5), 5.0, 1000.0)

        # Calculate Reward (Minimize latency spikes, stay under memory cap, penalize action costs)
        # Latency Penalty (exponential scaling above 100ms)
        latency_penalty = - (self.latency / 40.0) ** 1.8
        
        # Memory Penalty (critical penalty above 90% utilization)
        memory_penalty = -50.0 if self.memory_util > 90.0 else 0.0
        
        # Throughput Reward
        throughput_reward = (self.request_vol / 1000.0) * 2.0

        reward = latency_penalty + memory_penalty + throughput_reward + action_cost
        
        # Stop condition (extreme system outage)
        done = bool(self.latency > 900.0 or self.step_count >= 500)
        
        info = {
            "latency": self.latency,
            "memory": self.memory_util,
            "requests": self.request_vol
        }
        
        return self._get_state(), float(reward), done, info


class ScaleRLAgent:
    """
    Q-learning agent supporting continuous space discretization for fast,
    efficient server deployment without high GPU overhead.
    """
    def __init__(self, action_size: int = 4):
        self.action_size = action_size
        self.gamma = 0.95        # Discount factor
        self.epsilon = 1.0       # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.alpha = 0.1         # Learning rate

        # Discretize state space bins
        self.latency_bins = np.linspace(0, 500, 10)
        self.memory_bins = np.linspace(0, 100, 8)
        self.req_bins = np.linspace(0, 8000, 8)

        # Initialize Q-Table with zeros
        # Shape: (10, 8, 8, 4) -> 2,560 discrete state-action values
        self.q_table = np.zeros((len(self.latency_bins) + 1, 
                                 len(self.memory_bins) + 1, 
                                 len(self.req_bins) + 1, 
                                 action_size))

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, int, int]:
        lat, mem, req = state
        lat_idx = int(np.digitize(lat, self.latency_bins))
        mem_idx = int(np.digitize(mem, self.memory_bins))
        req_idx = int(np.digitize(req, self.req_bins))
        return lat_idx, mem_idx, req_idx

    def get_action(self, state: np.ndarray, train: bool = True) -> int:
        if train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        lat_idx, mem_idx, req_idx = self._discretize_state(state)
        return int(np.argmax(self.q_table[lat_idx, mem_idx, req_idx]))

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        lat_idx, mem_idx, req_idx = self._discretize_state(state)
        next_lat, next_mem, next_req = self._discretize_state(next_state)

        # Target value calculation
        best_next_action = np.argmax(self.q_table[next_lat, next_mem, next_req])
        td_target = reward if done else reward + self.gamma * self.q_table[next_lat, next_mem, next_req, best_next_action]
        
        # Update Q-Value
        self.q_table[lat_idx, mem_idx, req_idx, action] += self.alpha * (td_target - self.q_table[lat_idx, mem_idx, req_idx, action])

        # Decay exploration
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class LRU_CacheSimulator:
    """
    Standard LRU Baseline to run comparative evaluation against ScaleRL.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: List[int] = []
        self.hits = 0
        self.misses = 0

    def access(self, item: int):
        if item in self.cache:
            self.cache.remove(item)
            self.cache.append(item)
            self.hits += 1
            return True
        else:
            if len(self.cache) >= self.capacity:
                self.cache.pop(0)  # Evict oldest element (index 0)
            self.cache.append(item)
            self.misses += 1
            return False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 100.0


# ------------------ FASTAPI WEB SERVICE ------------------

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ScaleRL Distributed Infrastructure Gateway",
    description="Production backend running real-time Reinforcement Learning state analysis and eviction control.",
    version="1.0.0"
)

# Enable CORS for frontend visualization dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables holding real-time training engines
env = ScaleRLCacheEnv()
agent = ScaleRLAgent()
metrics_history: List[Dict[str, Any]] = []

class ServerStatePayload(BaseModel):
    latency: float
    memory_utilization: float
    request_volume: float

class ActionResponse(BaseModel):
    action_id: int
    action_name: str
    recommendation: str

ACTION_MAPPING = {
    0: "IDLE_STANDBY",
    1: "SOFT_EVICTION",
    2: "AGGRESSIVE_EVICTION",
    3: "DYNAMIC_TRAFFIC_REROUTE"
}

ACTION_RECOMMENDATIONS = {
    0: "System within nominal boundaries. Maintain standard cache policies.",
    1: "Cold data evictions triggered. Clearing Tier 3 cache pipelines.",
    2: "High load warning. Triggering immediate core memory cleanup blocks.",
    3: "Traffic limits breached. Directing 25% overflow capacity to target backends."
}

@app.on_event("startup")
def pretrain_agent():
    """Run an initial background pretraining session on system launch."""
    print("Pre-training ScaleRL Agent...")
    state = env.reset()
    for _ in range(2000):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = env.reset() if done else next_state
    print("ScaleRL Pre-training Phase Complete. Epsilon state:", agent.epsilon)

@app.post("/api/v1/inference", response_model=ActionResponse)
def get_eviction_decision(state: ServerStatePayload):
    """
    Accept physical host metrics and output a dynamic RL routing & eviction directive.
    """
    state_vector = np.array([state.latency, state.memory_utilization, state.request_volume])
    action = agent.get_action(state_vector, train=False)
    
    return ActionResponse(
        action_id=action,
        action_name=ACTION_MAPPING[action],
        recommendation=ACTION_RECOMMENDATIONS[action]
    )

@app.post("/api/v1/train-step")
def register_train_step(state: ServerStatePayload, action: int, reward: float, next_state: ServerStatePayload, done: bool):
    """
    Online Learning Endpoint: Collect experience tuples directly from deployed clusters.
    """
    state_vec = np.array([state.latency, state.memory_utilization, state.request_volume])
    next_state_vec = np.array([next_state.latency, next_state.memory_utilization, next_state.request_volume])
    
    agent.learn(state_vec, action, reward, next_state_vec, done)
    return {"status": "success", "updated_epsilon": float(agent.epsilon)}

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "rl_engine_state": "online",
        "epsilon_exploration": float(agent.epsilon),
        "q_table_density": float(np.count_nonzero(agent.q_table) / agent.q_table.size)
    }

# --- Optional: serve the dashboard HTML ---
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

_DASHBOARD_PATH = Path(__file__).with_name("scalerl_dashboard.html")

@app.get("/", response_class=HTMLResponse)
def dashboard_root():
    if not _DASHBOARD_PATH.exists():
        return HTMLResponse(content="<h3>scalerl_dashboard.html not found in container</h3>", status_code=404)
    return _DASHBOARD_PATH.read_text(encoding="utf-8")

if __name__ == "__main__":
    # If file run directly, start the production API server
    uvicorn.run("scalerl_engine.py:app", host="0.0.0.0", port=8000, reload=True)
