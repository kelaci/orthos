"""
Contextual Bandit Meta-Controller for Real-time Adaptation.

This module provides an online adaptation mechanism that modulates meta-parameters
based on the current system context (e.g., error, uncertainty).
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class ContextualBanditMetaController:
    """
    Contextual Bandit Meta-Controller.

    Adapts parameters like learning rates and sparsity thresholds in real-time
    by observing the 'state' of the system and choosing an 'action' (adjustment).

    Attributes:
        n_features: Number of context features.
        n_actions: Number of discrete action sets.
        learning_rate: Learning rate for the bandit's internal model.
        exploration_prob: Epsilon for epsilon-greedy exploration.
    """

    def __init__(
        self,
        n_features: int = 4,   # [error, uncertainty, sparsity, drift]
        n_actions: int = 5,    # [decrease_all, decrease_small, stay, increase_small, increase_all]
        learning_rate: float = 0.05,
        exploration_prob: float = 0.1
    ):
        """Initialize the Bandit Meta-Controller."""
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = exploration_prob

        # Internal model: A simple linear weight matrix for Thompson Sampling/LinUCB
        # Mapping context -> projected reward for each action
        self.weights = np.zeros((n_actions, n_features))
        self.bias = np.zeros(n_actions)
        
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

    def get_action(self, context: np.ndarray) -> int:
        """
        Select an action based on context using epsilon-greedy exploration.

        Args:
            context: Feature vector of current system state.

        Returns:
            Action index.
        """
        self.last_state = context
        
        if np.random.random() < self.epsilon:
            self.last_action = np.random.randint(self.n_actions)
        else:
            # Predict rewards: Q(s, a) = w_a * s + b_a
            rewards = np.dot(self.weights, context) + self.bias
            self.last_action = int(np.argmax(rewards))
            
        return self.last_action

    def update(self, reward: float) -> None:
        """
        Update the internal model based on the received reward.

        Args:
            reward: Scalar reward (e.g., -loss - uncertainty).
        """
        if self.last_state is None or self.last_action is None:
            return

        # Simple gradient descent on the linear approximator
        # Target is the actual reward
        prediction = np.dot(self.weights[self.last_action], self.last_state) + self.bias[self.last_action]
        error = reward - prediction
        
        # Update weights and bias for the chosen action
        self.weights[self.last_action] += self.lr * error * self.last_state
        self.bias[self.last_action] += self.lr * error

    def apply_modulation(self, base_params: Dict[str, float], action: int) -> Dict[str, float]:
        """
        Apply the selected action to modulate base parameters.

        Args:
            base_params: Original meta-parameters.
            action: Selected action index.

        Returns:
            Modulated parameters.
        """
        # Mapping actions to multipliers
        # 0: Sharp decrease, 1: Slight decrease, 2: Identity, 3: Slight increase, 4: Sharp increase
        multipliers = [0.5, 0.9, 1.0, 1.1, 2.0]
        m = multipliers[action]
        
        modulated = {}
        for k, v in base_params.items():
            # Specifically modulate learning-related params
            if any(term in k for term in ['rate', 'noise', 'weight', 'tau']):
                modulated[k] = v * m
            else:
                modulated[k] = v
                
        return modulated

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            "n_features": self.n_features,
            "n_actions": self.n_actions,
            "epsilon": self.epsilon,
            "lr": self.lr
        }
