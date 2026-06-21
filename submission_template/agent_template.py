"""
agent_template.py — Agent Interface & Skeleton
================================================
All teams MUST:
  1. Inherit from BaseAdAgent
  2. Implement select_ad(context, ad_catalog)  →  int

You must implement the logic for select_ad() and update().
Your task: implement your OWN version in submission_template/agent.py.
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# BASE CLASS  —  All submitted agents MUST inherit from this
# ══════════════════════════════════════════════════════════════════════

class BaseAdAgent:
    """
    Abstract base class.

    Internal arrays (already set up):
        self.counts[a]  — how many times ad a has been pulled
        self.values[a]  — running average reward (CTR estimate) for ad a

    Add any extra attributes you need in your __init__.
    """

    def __init__(self, num_ads: int = 30):
        self.num_ads = num_ads
        self.counts  = np.zeros(num_ads, dtype=np.float64)   # N(a)
        self.values  = np.zeros(num_ads, dtype=np.float64)   # Q(a)

    def select_ad(self, context: dict, ad_catalog) -> int:
        """
        Choose which ad to show this round.

        Parameters
        ----------
        context : dict
            {"round": int, "user_type": int (0-4)}
        ad_catalog : pd.DataFrame
            Columns: ad_id, category, format, target_demo, baseline_ctr

        Returns
        -------
        int — ad_id in [0, num_ads)
        """
        raise NotImplementedError("Implement select_ad() in your agent.")

    def update(self, ad_id: int, reward: float):
        """
        Update estimates after observing a reward.
        Default: incremental mean  Q(a) ← Q(a) + (r − Q(a)) / N(a)
        Override this if your algorithm needs different logic
        (e.g. Thompson Sampling updates alpha/beta, not just Q(a)).
        """
        self.counts[ad_id] += 1
        n = self.counts[ad_id]
        self.values[ad_id] += (reward - self.values[ad_id]) / n

    def reset(self):
        """Called at the start of every evaluation run."""
        self.counts[:] = 0
        self.values[:] = 0


# ══════════════════════════════════════════════════════════════════════
# BASELINE — Random Agent  (the floor — beat this to pass)
# ══════════════════════════════════════════════════════════════════════

class RandomAgent(BaseAdAgent):
    """
    Picks a random ad every round. No learning.
    This is the absolute floor — any correct MAB implementation beats it.
    """

    def select_ad(self, context: dict, ad_catalog) -> int:
        return int(np.random.randint(0, self.num_ads))
