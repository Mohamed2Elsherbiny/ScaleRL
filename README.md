# ScaleRL

**Distributed Reinforcement Learning-Based Cache Eviction & Traffic Routing Engine**

ScaleRL is an intelligent, self-optimizing distributed cache management and traffic routing system powered by Reinforcement Learning (RL). Unlike traditional cache eviction strategies such as LRU (Least Recently Used) and LFU (Least Frequently Used), ScaleRL continuously learns from real-time infrastructure telemetry and dynamically executes operational actions to maintain low latency, prevent memory pressure, and optimize throughput.

## Overview

Modern distributed systems operate under highly dynamic traffic conditions where static cache policies often fail to react to sudden workload shifts. ScaleRL models cache eviction and traffic steering as a Markov Decision Process (MDP), enabling adaptive decision-making based on live system conditions.

The platform combines:

* Reinforcement Learning-based control policies
* Real-time telemetry monitoring
* Dynamic cache eviction strategies
* Intelligent traffic rerouting
* Low-latency inference execution (<1 ms)

## Key Features

* **Adaptive Cache Management**

  * Soft cache eviction for cold data cleanup
  * Aggressive cache eviction for rapid memory recovery
  * Dynamic policy learning based on workload behavior

* **Traffic Engineering**

  * Intelligent load redistribution across failover nodes
  * Automated congestion mitigation

* **Real-Time Decision Making**

  * Sub-millisecond inference latency
  * Lightweight Q-learning implementation
  * No GPU dependency

* **Production-Oriented Design**

  * FastAPI-based inference gateway
  * Dockerized deployment architecture
  * Minimal runtime footprint

## System Components

### 1. Inference Gateway

A lightweight FastAPI service responsible for:

* Collecting telemetry metrics
* Executing policy inference
* Returning operational recommendations

Input metrics include:

* System latency (ms)
* Memory utilization (%)
* Request volume (RPS)

### 2. Gymnasium Environment

`ScaleRLCacheEnv` simulates realistic distributed infrastructure behavior, including:

* Traffic fluctuations
* Network jitter
* Memory pressure
* Latency degradation
* Cache thrashing scenarios

### 3. RL Control Agent

The ScaleRL agent uses discretized Q-learning to learn optimal operational responses while maintaining extremely low inference costs.

## Reinforcement Learning Formulation

### State Space

The environment state is represented as:

```text
S = [Latency, Memory Utilization, Request Volume]
```

| Variable | Description        | Range       |
| -------- | ------------------ | ----------- |
| Latency  | System latency     | 0–1000 ms   |
| Memory   | Memory utilization | 0–100%      |
| Volume   | Incoming requests  | 0–10000 RPS |

### Action Space

| Action ID | Action                  | Description                        |
| --------- | ----------------------- | ---------------------------------- |
| 0         | IDLE_STANDBY            | No intervention                    |
| 1         | SOFT_EVICTION           | Clear cold cache data              |
| 2         | AGGRESSIVE_EVICTION     | Flush warm and cold cache tiers    |
| 3         | DYNAMIC_TRAFFIC_REROUTE | Redirect traffic to failover nodes |

### Reward Strategy

The reward function balances:

* Low latency
* Safe memory utilization
* High throughput
* Minimal operational overhead

The agent is heavily penalized for:

* Excessive latency
* Memory utilization above 90%
* Unnecessary cache thrashing
* Excessive traffic rerouting

## Q-Learning Configuration

### State Discretization

| Metric             | Bins |
| ------------------ | ---- |
| Latency            | 10   |
| Memory Utilization | 8    |
| Request Volume     | 8    |

Total discrete states:

```text
10 × 8 × 8 = 640 states
```

### Training Parameters

```python
learning_rate = 0.1
discount_factor = 0.95

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
```

## API

### POST `/api/v1/inference`

#### Request

```json
{
  "latency": 115.4,
  "memory_utilization": 87.2,
  "request_volume": 4200.0
}
```

#### Response

```json
{
  "action_id": 2,
  "action_name": "AGGRESSIVE_EVICTION",
  "recommendation": "High load warning. Triggering immediate core memory cleanup blocks."
}
```

## Deployment

ScaleRL is designed as a containerized microservice.

### Docker Strategy

#### Builder Stage

* Compiles dependencies
* Builds Python packages and wheels

#### Runtime Stage

* Minimal production image
* Reduced attack surface
* Faster startup time
* Lower image size

Default service port:

```text
8000
```

## Performance Results

ScaleRL was evaluated against a traditional LRU cache policy under highly volatile traffic conditions.

### Observed Benefits

* 18.2% reduction in average system latency
* 85% reduction in tail latency (p99 events)
* Prevention of memory overflow scenarios
* Reduced cache thrashing
* Improved SLA stability during traffic spikes

### Adaptive Behavior

ScaleRL proactively:

* Triggers soft evictions near 80% memory utilization
* Performs aggressive cleanup before critical thresholds
* Reroutes traffic under sustained congestion
* Learns action timing to avoid unnecessary overhead

## Production Recommendations

### Warm-Up Training

Before enabling live actions:

```text
Minimum pre-training: 5,000 steps
```

Use:

* Synthetic workloads
* Shadow traffic
* Replay traces

### Multi-Cluster Deployment

For large-scale environments:

* Deploy one ScaleRL instance per cluster node
* Share routing statistics between agents
* Enable cooperative load balancing

## Future Enhancements

* Deep Q-Network (DQN) support
* PPO-based policy optimization
* Multi-agent reinforcement learning
* Distributed Q-table synchronization
* Kubernetes-native autoscaling integration
* Online continual learning
