"""
run_your_agent.py — Test YOUR submitted agents
==============================================
This script evaluates your agents from submission_template/agent.py
against published benchmarks.

Usage:
    python run_your_agent.py

What it does:
    1. Loads all your agents from submission_template/agent.py
    2. Runs each for 5,000 rounds
    3. Shows your CTR and clicks compared to the required benchmarks
"""

import sys
import os
import time
import importlib.util
import numpy as np
import pandas as pd
from environment import AdBanditEnv
from agent_template import BaseAdAgent

BENCHMARKS = {
    "EpsilonGreedyAgent":    {"target": 20.0, "desc": "Beat Random (~10%)"},
    "UCBAgent":              {"target": 21.0, "desc": "Learn effectively"},
    "ThompsonSamplingAgent": {"target": 30.0, "desc": "Reach non-contextual ceiling"},
    "CustomAgent":           {"target": 38.0, "desc": "Use user_type context"},
}

NUM_ROUNDS = 5_000
SEED       = 8     # development seed


def load_agents(num_ads: int):
    """Load agents from submission_template/agent.py."""
    agent_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "submission_template", "agent.py"
    )
    if not os.path.exists(agent_path):
        print("ERROR: submission_template/agent.py not found.")
        sys.exit(1)

    spec   = importlib.util.spec_from_file_location("my_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.join(os.path.dirname(agent_path)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    spec.loader.exec_module(module)

    agents = {}
    for name in BENCHMARKS.keys():
        if hasattr(module, name):
            agent_class = getattr(module, name)
            agents[name] = agent_class(num_ads=num_ads)
        else:
            print(f"ERROR: Missing class {name} in agent.py")
    
    return agents


def run_agent(agent: BaseAdAgent, env: AdBanditEnv) -> dict:
    """Run the agent for NUM_ROUNDS rounds and collect metrics."""
    agent.reset()
    env.reset(seed=SEED)

    total_reward     = 0.0
    unique_ads_shown = set()
    latencies_ms     = []

    for _ in range(NUM_ROUNDS):
        context = env.observe()
        t0      = time.perf_counter()
        ad_id   = agent.select_ad(context, env.catalog)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        reward = env.pull(ad_id)
        agent.update(ad_id, reward)

        total_reward += reward
        unique_ads_shown.add(ad_id)

    return {
        "ctr":          round(total_reward / NUM_ROUNDS * 100, 2),
        "total_clicks": int(total_reward),
        "unique_ads":   len(unique_ads_shown),
        "diversity_pct":round(len(unique_ads_shown) / env.NUM_ADS * 100, 1),
        "avg_latency":  round(float(np.mean(latencies_ms)), 4),
    }


if __name__ == "__main__":
    env    = AdBanditEnv(catalog_path="data/ad_catalog.csv")
    agents = load_agents(num_ads=env.NUM_ADS)

    print()
    print("=" * 70)
    print("  Testing YOUR Agents")
    print("=" * 70)
    print(f"  Rounds : {NUM_ROUNDS:,}  |  Ads : {env.NUM_ADS}")
    print()

    for name, agent in agents.items():
        print(f"── {name} ───────────────────────────────────────")
        try:
            metrics = run_agent(agent, env)
            ctr = metrics["ctr"]
            target = BENCHMARKS[name]["target"]
            
            if ctr >= target * 0.95:  # 5% leniency on the dev seed
                status = "PASS"
            else:
                status = "FAIL"
                
            print(f"  CTR      : {ctr}%  (Target: ~{target}%)")
            print(f"  Clicks   : {metrics['total_clicks']}")
            print(f"  Status   : {status}")
            print(f"  Latency  : {metrics['avg_latency']} ms")
            
        except NotImplementedError:
            print(f"  Not Implemented yet")
        except Exception as e:
            print(f"  Error: {e}")
        print()

