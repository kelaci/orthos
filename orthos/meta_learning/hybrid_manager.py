"""
Hybrid Meta-Learning Manager.

This module coordinates the global (NES), online (Bandit), and micro (Hebbian)
learning scales within the ORTHOS framework.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from orthos.meta_learning.nes import NaturalEvolutionStrategy
from orthos.meta_learning.bandit import ContextualBanditMetaController


class HybridMetaManager:
    """
    Orchestrator for the Hybrid Meta-Learning strategy.

    Coordinates between:
    - Global NES: For mapping robust base parameters across episodes.
    - Online Bandit: For sub-second adaptation to current environmental context.
    - Safety Clamps: To ensure system stability.
    """

    def __init__(
        self,
        base_parameters: Dict[str, float],
        use_bandit: bool = True,
        use_nes: bool = True
    ):
        """Initialize Hybrid Meta-Manager."""
        self.base_params = base_parameters.copy()
        self.active_params = self.base_params.copy()
        
        self.use_bandit = use_bandit
        self.use_nes = use_nes
        
        # Initialize components
        if use_nes:
            self.nes = NaturalEvolutionStrategy(
                population_size=40,
                sigma=0.1,
                learning_rate=0.01
            )
        
        if use_bandit:
            self.bandit = ContextualBanditMetaController(
                n_features=4,  # error, uncertainty, sparsity, drift
                n_actions=5    # decrease_hard, decrease, stay, increase, increase_hard
            )

    def step(self, context: np.ndarray) -> Dict[str, float]:
        """
        Execute a meta-learning step.

        Args:
            context: Current system context [error, uncertainty, sparsity, drift].

        Returns:
            Modulated meta-parameters to use in the NEXT time step.
        """
        if not self.use_bandit:
            return self.base_params
        
        # Select action based on context
        action = self.bandit.get_action(context)
        
        # Modulate
        self.active_params = self.bandit.apply_modulation(self.base_params, action)
        
        # Apply Safety Clamps
        self._apply_safety_clamps()
        
        return self.active_params

    def update_feedback(self, reward: float) -> None:
        """
        Provide reward feedback to the online controller.

        Args:
            reward: Scalar reward for the previous step.
        """
        if self.use_bandit:
            self.bandit.update(reward)

    def _apply_safety_clamps(self) -> None:
        """Ensure parameters stay within physically meaningful bounds."""
        # Hard clamps for stability
        if 'adaptation_rate' in self.active_params:
            self.active_params['adaptation_rate'] = np.clip(
                self.active_params['adaptation_rate'], 1e-5, 0.5
            )
            
        if 'exploration_noise' in self.active_params:
            self.active_params['exploration_noise'] = np.clip(
                self.active_params['exploration_noise'], 1e-4, 1.0
            )

    def global_update(self, fitness_history: np.ndarray, noise: np.ndarray) -> None:
        """
        Perform a global parameters update using NES.
        Usually called at the end of an episode or a block of tasks.

        Args:
            fitness_history: Fitness scores for the population.
            noise: The noise matrix used to generate the population.
        """
        if self.use_nes:
            self.nes.update(noise, fitness_history)
            # Update base params to the new mean
            new_mean = self.nes.get_mean()
            # Map mean back to dict keys (assuming a fixed order)
            # This requires a mapping between vector and dict, 
            # which will be implemented in a more robust way in v4.3.
            pass

    def get_config(self) -> Dict[str, Any]:
        """Return full configuration."""
        config = {
            "use_bandit": self.use_bandit,
            "use_nes": self.use_nes,
            "base_parameters": self.base_params
        }
        if self.use_bandit:
            config["bandit"] = self.bandit.get_config()
        if self.use_nes:
            config["nes"] = self.nes.get_config()
        return config
