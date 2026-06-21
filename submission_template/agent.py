"""
agent.py — YOUR submission
===========================
Implement the four agents below. You MUST implement select_ad() and update()
for each of them.

Requirements:
  - Inherit from BaseAdAgent
  - Implement the specific algorithm for each class

Test your implementation:
    python run_your_agent.py
"""

import numpy as np
import pandas as pd
from agent_template import BaseAdAgent


class EpsilonGreedyAgent(BaseAdAgent):
    """
    ε-Greedy Agent.
    """
    def __init__(self, num_ads: int = 30):
        super().__init__(num_ads)
        # TODO: Add your state here (e.g., epsilon)
        pass

    def select_ad(self, context: dict, ad_catalog) -> int:
        raise NotImplementedError("Implement select_ad() for EpsilonGreedyAgent!")

    def update(self, ad_id: int, reward: float):
        super().update(ad_id, reward)
        # TODO: Add your update logic here

    def reset(self):
        super().reset()
        # TODO: Reset your extra state


class UCBAgent(BaseAdAgent):
    """
    Upper Confidence Bound (UCB) Agent.
    """
    def __init__(self, num_ads: int = 30):
        super().__init__(num_ads)
        # TODO: Add your state here (e.g., round counter, c parameter)
        pass

    def select_ad(self, context: dict, ad_catalog) -> int:
        raise NotImplementedError("Implement select_ad() for UCBAgent!")

    def update(self, ad_id: int, reward: float):
        super().update(ad_id, reward)
        # TODO: Add your update logic here

    def reset(self):
        super().reset()
        # TODO: Reset your extra state


class ThompsonSamplingAgent(BaseAdAgent):
    """
    Thompson Sampling Agent (context-blind).
    """
    def __init__(self, num_ads: int = 30):
        super().__init__(num_ads)
        # TODO: Add your state here (e.g., alpha, beta)
        pass

    def select_ad(self, context: dict, ad_catalog) -> int:
        raise NotImplementedError("Implement select_ad() for ThompsonSamplingAgent!")

    def update(self, ad_id: int, reward: float):
        super().update(ad_id, reward)
        # TODO: Add your update logic here

    def reset(self):
        super().reset()
        # TODO: Reset your extra state


class CustomAgent(BaseAdAgent):
    """
    Custom Agent (Excellent Tier).
    Use the user_type context to build a contextual bandit algorithm.
    """
    def __init__(self, num_ads: int = 30):
        super().__init__(num_ads)
        # TODO: Add your state here (e.g., contextual variables)
        pass

    def select_ad(self, context: dict, ad_catalog) -> int:
        raise NotImplementedError("Implement select_ad() for CustomAgent!")

    def update(self, ad_id: int, reward: float):
        super().update(ad_id, reward)
        # TODO: Add your update logic here

    def reset(self):
        super().reset()
        # TODO: Reset your extra state
