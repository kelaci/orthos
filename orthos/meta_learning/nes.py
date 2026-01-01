"""
Natural Evolution Strategies (NES) Implementation.

This module implements the Natural Evolution Strategy optimizer, which uses
natural gradients to optimize meta-parameters.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from orthos.core.types import Tensor


class NaturalEvolutionStrategy:
    """
    Natural Evolution Strategy (NES) optimizer.

    NES represents the search distribution by a multivariate Gaussian and
    updates its parameters (mean and covariance) by following the natural
    gradient of the expected fitness.

    Attributes:
        population_size: Number of individuals per generation.
        sigma: Mutation strength (standard deviation).
        learning_rate: Learning rate for the mean update.
        rank_fitness: Whether to use rank-based fitness normalization.
        mean: Current mean parameter vector.
        history: Fitness history.
    """

    def __init__(
        self,
        population_size: int = 40,
        sigma: float = 0.1,
        learning_rate: float = 0.01,
        rank_fitness: bool = True
    ):
        """Initialize NES optimizer."""
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.rank_fitness = rank_fitness
        
        self.mean: Optional[np.ndarray] = None
        self.history: List[float] = []

    def generate_population(self, initial_mean: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate a population of perturbed parameters.

        Args:
            initial_mean: Base parameter vector.

        Returns:
            Tuple of (list of perturbed parameters, noise matrix).
        """
        self.mean = initial_mean.copy()
        dim = self.mean.shape[0]
        
        # Sample noise: N x dim
        noise = np.random.randn(self.population_size, dim)
        
        # Perturb
        population = [self.mean + self.sigma * noise[i] for i in range(self.population_size)]
        
        return population, noise

    def update(self, noise: np.ndarray, fitness_scores: np.ndarray) -> float:
        """
        Update mean based on fitness scores using natural gradient approximation.

        Args:
            noise: The noise vectors used to generate the population (N x dim).
            fitness_scores: The resulting fitness for each individual (N,).

        Returns:
            The average fitness of this generation.
        """
        if self.mean is None:
            raise ValueError("Mean not initialized. Call generate_population first.")

        # 1. Rank-based fitness transformation (Standard NES practice)
        if self.rank_fitness:
            # Ranks from 0 to N-1
            ranks = np.argsort(np.argsort(fitness_scores))
            # Normalize to [-0.5, 0.5]
            fitness = (ranks / (self.population_size - 1)) - 0.5
        else:
            # Baseline-subtraction normalization
            fitness = (fitness_scores - np.mean(fitness_scores)) / (np.std(fitness_scores) + 1e-8)

        # 2. Natural Gradient Estimate: grad = 1/(N*sigma) * sum(fitness * noise)
        # Note: Since we use rank normalization, we can treat it as a gradient of log-likelihood
        gradient = np.dot(fitness, noise) / (self.population_size * self.sigma)

        # 3. Parameter Update
        self.mean += self.learning_rate * gradient

        # 4. Optional: Sigma adaptation (Basic rule)
        # In a full NES we would also update sigma using the natural gradient over sigma
        
        avg_fitness = float(np.mean(fitness_scores))
        self.history.append(avg_fitness)
        
        return avg_fitness

    def get_mean(self) -> np.ndarray:
        """Return the current optimized mean."""
        if self.mean is None:
            raise ValueError("Mean not initialized.")
        return self.mean.copy()

    def reset(self) -> None:
        """Reset history and mean."""
        self.mean = None
        self.history = []

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            "population_size": self.population_size,
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "rank_fitness": self.rank_fitness
        }

    def dict_to_vector(self, params_dict: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        """Convert a dictionary of parameters to a flat vector."""
        keys = sorted(params_dict.keys())
        vector = np.array([params_dict[k] for k in keys])
        return vector, keys

    def vector_to_dict(self, vector: np.ndarray, keys: List[str]) -> Dict[str, float]:
        """Convert a flat vector back to a dictionary."""
        return {k: float(vector[i]) for i, k in enumerate(keys)}
