#  RL Mid-Eval — Ad Recommender with Multi-Armed Bandits

> **Problem**: Build a Multi-Armed Bandit agent that maximises ad clicks in a simulated web environment  
> **Algorithms**: Epsilon-Greedy, UCB, Thompson Sampling, and a Custom Contextual agent

---

##  What's In This Kit

```
rl-ad-recommender-kit/
│
├── README.md                  ← You are here
├── environment.py             ← The bandit simulator
├── agent_template.py          ← BaseAdAgent class definition
├── run_your_agent.py          ← Script to test your agents
│
├── data/                      ← Pre-built datasets (do not modify)
│   ├── ad_catalog.csv         ← 30 ads with their features
│   └── offline_log.csv        ← 10,000 historical interactions (random policy)
│
└── submission_template/       ← Copy this folder for your submission
    ├── agent.py               ← Fill in your agent here
    ├── artifacts/             ← Place any saved tables/files here
    └── requirements.txt
```

>  **Data is pre-built.** Do not try to regenerate it — you won't have the generator script, and the evaluation uses a fixed hidden version of the environment.

---

##  Quick Start

```bash
# Step 1 — install dependencies
pip install numpy pandas

# Step 2 — test your setup
python run_your_agent.py
```

You should see an output comparing your agents to target benchmarks.
Since your agents are currently blank, they will raise `NotImplementedError`.

---

##  Problem Formulation — Multi-Armed Bandit

This is a **Bernoulli Multi-Armed Bandit** problem.

| Concept | In this problem |
|---------|----------------|
| **Arms** | 30 ads (ad_id 0 – 29) |
| **Pull** | Show one ad to a user |
| **Reward** | 1 = user clicked, 0 = user skipped |
| **Goal** | Maximise total clicks over T rounds |

At each round, your agent sees a **context** and picks one ad:

```python
context = env.observe()                   # {"round": t, "user_type": 0-4}
ad_id   = agent.select_ad(context, catalog)  # your decision
reward  = env.pull(ad_id)                 # 1.0 or 0.0
agent.update(ad_id, reward)              # update your estimates
```

---

##  The Context

Each round provides a small context dict:

```python
{
    "round":     142,   # how many rounds have elapsed
    "user_type": 3      # an integer in [0, 4]
}
```

**user_type** is an opaque category — you are not told what it means.  
A basic agent can ignore it. A smarter agent can learn **per-user-type statistics** — e.g., user type 2 may prefer Tech ads.

---

##  The Ad Catalog (`data/ad_catalog.csv`)

| Column | Description |
|--------|-------------|
| `ad_id` | Integer 0–29 — this is your action |
| `category` | Tech / Fashion / Automotive / Finance / FMCG |
| `format` | Banner / Video / Native |
| `target_demo` | Intended age group (metadata only) |
| `baseline_ctr` | Historical average CTR — **a noisy guide, not the true CTR** |

>  The true CTR in the live environment depends on ad features AND user type. You cannot compute it from the catalog alone — you must **learn it through pulls**.

---

## Offline Data (`data/offline_log.csv`)

10,000 historical interactions logged from a **random policy**. You can use this to:
- Inspect average per-arm rewards before running any algorithm
- Warm-start your agent's initial estimates (e.g., initialise UCB Q-values)
- Understand how reward is distributed across arms and user types

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `user_type` | int | Which user type was shown the ad (0–4) |
| `ad_id` | int | Which ad was shown |
| `reward` | float | 1.0 (clicked) or 0.0 (skipped) |

**Load it:**
```python
import pandas as pd
log = pd.read_csv("data/offline_log.csv")

# Compute per-arm average reward from offline data (warm-start)
arm_means = log.groupby("ad_id")["reward"].mean()
```

---

##  Algorithms to Implement

You must implement **ALL FOUR** of the following agents in your `agent.py`:

### 1. ε-Greedy (`EpsilonGreedyAgent`)
```
With probability ε  → pick a random arm (explore)
With probability 1-ε → pick argmax Q(a) (exploit)

Q(a) updated as: Q(a) ← Q(a) + (r − Q(a)) / N(a)
```

**Tip**: Try a **decaying epsilon**: `ε_t = ε₀ / sqrt(t)` — more exploration early, more exploitation later.

### 2. UCB (`UCBAgent`)
```
UCB(a) = Q(a) + c × sqrt( ln(t) / N(a) )

Pick: a* = argmax UCB(a)
```
- Pull each arm once before using UCB
- `c` controls exploration (start with 0.1–1.0)

### 3. Thompson Sampling (`ThompsonSamplingAgent`)
```
For each arm a, maintain:
    α(a) = clicks + 1       (Beta distribution success count)
    β(a) = no-clicks + 1    (Beta distribution failure count)

Each round:
    θ(a) ~ Beta(α(a), β(a))   for all a
    Pick: a* = argmax θ(a)

Update: α(a*) += reward,  β(a*) += (1 − reward)
```

### 4. Custom Contextual Agent (`CustomAgent`)
Build an agent that uses the `user_type` context (provided in `context["user_type"]`) to track separate estimates for each user type. This breaks the non-contextual performance ceiling.

---



## Submission Instructions

1. Copy `submission_template/` and rename it: `YourTeamName/`
2. Fill in `agent.py` — implement all 4 classes.
3. Add any pre-computed files (e.g., saved Q-tables) to `artifacts/`
4. Update `requirements.txt`
5. **Self-test before submitting:**
   ```bash
   python run_your_agent.py
   ```
6. Zip and submit: `YourTeamName.zip`

---

##  Rules

|  Allowed |  Not Allowed |
|-----------|---------------|
| ε-Greedy, UCB, Thompson Sampling | Hard-coding CTR values from environment.py |
| Any epsilon decay schedule | Training a neural network |
| Warm-starting from `offline_log.csv` | Sharing code between teams |
| Using `user_type` to build per-context estimates | Modifying `environment.py` |
| Saving a Q-table to `artifacts/` | Running updates inside `select_ad()` call |

---

##  Strategy Tips

1. **Don't trust `baseline_ctr`** — the true CTR in the simulator is different. Your algorithm must learn from interactions.
2. **Use offline data to warm-start** — compute per-arm empirical means from `offline_log.csv` and use them to initialise `Q(a)` values or `α/β` parameters. This skips the cold-start penalty entirely.
3. **Tune your hyperparameter** — for ε-Greedy try ε ∈ {0.05, 0.10, 0.15}; for UCB try c ∈ {0.1, 0.5, 1.0}.
4. **Track diversity** — do not always pick the same arm forever.
5. **Thompson Sampling is the strongest** of the three algorithms on Bernoulli bandits — implement it correctly and you will likely see the best CTR.

Good luck! 
