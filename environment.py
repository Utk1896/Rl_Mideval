"""
environment.py — Ad Recommender Bandit Environment
====================================================
A lightweight, pure-Python environment for the Multi-Armed Bandit
formulation of ad recommendation.

No external RL libraries required — works with plain numpy.

ROUND STRUCTURE
---------------
Each "round" simulates showing one ad to one user.

    context = env.observe()                      # who is this user?
    ad_id   = agent.select_ad(context, catalog)  # agent picks an ad (0–29)
    reward  = env.pull(ad_id)                    # 1 = click, 0 = no click
    agent.update(ad_id, reward)                  # agent learns

USER TYPES (context)
--------------------
There are 5 user types (integers 0–4). The context dict gives you the
user type each round. You are NOT told what each type means or which
ads it prefers — your agent must learn this through interaction.

    A basic agent can ignore user_type and still work.
    A smarter agent won't ignore it.

REWARD
------
Binary: 1.0 (click) or 0.0 (no click).

The true click-through rate depends on BOTH the ad chosen AND the
current user type. This relationship is hidden — learn it from pulls.

NOTE: baseline_ctr in the catalog is a noisy historical average
across all user types. The true per-user-type CTR may differ
significantly. Don't rely on the catalog alone.
"""

import numpy as np
import pandas as pd


class AdBanditEnv:
    """
    Contextual Multi-Armed Bandit environment for Ad Recommendation.

    Parameters
    ----------
    catalog_path : str
        Path to ad_catalog.csv.
    """

    NUM_ADS        = 30
    NUM_USER_TYPES = 5

    def __init__(self, catalog_path: str = "data/ad_catalog.csv"):
        catalog_full = pd.read_csv(catalog_path)
        # Use first NUM_ADS rows, shuffled so arm index is not informative
        self.catalog = (
            catalog_full
            .head(self.NUM_ADS)
            .sample(frac=1, random_state=13)
            .reset_index(drop=True)
        )
        assert len(self.catalog) == self.NUM_ADS, \
            f"Catalog must have exactly {self.NUM_ADS} ads (got {len(self.catalog)})."

        self._rng          = None
        self._round        = 0
        self._current_user = None

        # Pre-compute hidden true CTR table  shape: (NUM_ADS, NUM_USER_TYPES)
        # Students do NOT have direct access to this.
        self._true_ctr_table = self._build_hidden_ctr_table()

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int = 8) -> dict:
        """
        Start a fresh evaluation run.

        Parameters
        ----------
        seed : int   Random seed for full reproducibility.

        Returns
        -------
        context : dict   Context for round 0.
        """
        self._rng          = np.random.default_rng(seed)
        self._round        = 0
        self._current_user = int(self._rng.integers(0, self.NUM_USER_TYPES))
        return self._get_context()

    def observe(self) -> dict:
        """
        Return the context for the current round WITHOUT advancing the clock.

        Returns
        -------
        dict with keys:
            "round"     : int   — rounds elapsed so far
            "user_type" : int   — current user's type (0–4)
        """
        return self._get_context()

    def pull(self, ad_id: int) -> float:
        """
        Show ad `ad_id` to the current user and collect the reward.
        Advances to the next round automatically.

        Parameters
        ----------
        ad_id : int   Must be in [0, NUM_ADS).

        Returns
        -------
        float   1.0 if the user clicked, 0.0 otherwise.
        """
        if self._rng is None:
            raise RuntimeError("Call env.reset(seed) before env.pull().")
        if not (0 <= ad_id < self.NUM_ADS):
            raise ValueError(f"ad_id must be in [0, {self.NUM_ADS}). Got {ad_id}.")

        true_ctr = self._true_ctr_table[ad_id, self._current_user]
        reward   = float(self._rng.random() < true_ctr)

        self._round       += 1
        self._current_user = int(self._rng.integers(0, self.NUM_USER_TYPES))
        return reward

    @property
    def num_ads(self) -> int:
        return self.NUM_ADS

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _get_context(self) -> dict:
        return {"round": self._round, "user_type": self._current_user}

    def _build_hidden_ctr_table(self) -> np.ndarray:
        """
        Build the true CTR matrix.  Shape: (NUM_ADS, NUM_USER_TYPES).

        DESIGN: Universal + Specialist structure
        -----------------------------------------
        Arms are split into three tiers:

          UNIVERSAL GOOD (4 arms)
            CTR is moderately high for ALL user types (0.30 – 0.40).
            A basic non-contextual MAB algorithm will find these and
            plateau here. This is the ceiling for context-blind agents.

          SPECIALIST (5 arms, one per user type)
            CTR is HIGH for one specific user type (~0.55) and LOW for
            all others (~0.05). A contextual agent that tracks per-user
            estimates learns to pick these when the right user arrives,
            pushing CTR well above the universal ceiling.

          BAD / TRAP (21 arms)
            CTR is very low (0.02 – 0.08) for all user types.
            These penalise uniform exploration (ε-Greedy) heavily.

        Expected performance ceilings
        ------------------------------
          Context-blind TS/UCB  →  ~38% CTR   (finds universal best arm)
          Contextual strategy   →  ~53% CTR   (matches specialist to user)

        This table is intentionally opaque — students must discover the
        structure through interaction, not by reading this file.
        """
        rng = np.random.default_rng(161803)   # hidden fixed seed

        NUM_UNIVERSAL  = 4
        NUM_SPECIALIST = self.NUM_USER_TYPES   # one per user type
        NUM_BAD        = self.NUM_ADS - NUM_UNIVERSAL - NUM_SPECIALIST  # 21

        table = np.zeros((self.NUM_ADS, self.NUM_USER_TYPES))

        # ── Universal good arms ───────────────────────────────────────
        universal_ctrs = np.array([0.40, 0.36, 0.32, 0.29])
        for i, ctr in enumerate(universal_ctrs):
            # Small noise across user types so they look natural in the log
            noise = rng.uniform(-0.02, 0.02, self.NUM_USER_TYPES)
            table[i, :] = np.clip(ctr + noise, 0.05, 0.95)

        # ── Specialist arms ───────────────────────────────────────────
        SPECIALIST_PEAK = 0.55   # CTR when user type matches
        SPECIALIST_BASE = 0.05   # CTR for all other user types
        for u in range(NUM_SPECIALIST):
            row = NUM_UNIVERSAL + u
            table[row, :] = SPECIALIST_BASE
            table[row, u] = SPECIALIST_PEAK

        # ── Bad / trap arms ───────────────────────────────────────────
        for i in range(NUM_BAD):
            row = NUM_UNIVERSAL + NUM_SPECIALIST + i
            table[row, :] = rng.uniform(0.02, 0.08, self.NUM_USER_TYPES)

        # Shuffle rows so arm index reveals nothing about tier
        shuffle_idx = rng.permutation(self.NUM_ADS)
        return table[shuffle_idx, :]
