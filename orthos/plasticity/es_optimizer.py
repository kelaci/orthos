"""
EvolutionaryStrategy implementation.

Evolutionary Strategy optimizer for plasticity parameter optimization.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from orthos.core.types import Tensor

class EvolutionaryStrategy:
    """
    Evolutionary Strategy optimizer for plasticity parameters.

    This class implements the Evolutionary Strategy algorithm for optimizing
    plasticity parameters through population-based search and elite selection.

    Attributes:
        population_size: Number of individuals in population
        sigma: Mutation strength
        learning_rate: Learning rate for mean update
        elite_fraction: Fraction of elites to select
        mean: Current mean parameter vector
        fitness_history: History of fitness scores
    """

    def __init__(self, population_size: int = 50, sigma: float = 0.1,
                 learning_rate: float = 0.01, elite_fraction: float = 0.2):
        """
        Initialize EvolutionaryStrategy.

        Args:
            population_size: Number of individuals in population
            sigma: Mutation strength (standard deviation)
            learning_rate: Learning rate for mean update
            elite_fraction: Fraction of top performers to select
        """
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.elite_fraction = elite_fraction
        self.mean: Optional[Tensor] = None
        self.fitness_history: List[float] = []

    def generate_population(self, initial_mean: Tensor) -> List[Tensor]:
        """
        Generate population of perturbed parameters.

        Args:
            initial_mean: Initial parameter vector

        Returns:
            List of perturbed parameter vectors

        Raises:
            ValueError: If initial_mean is not 1D array
        """
        if len(initial_mean.shape) != 1:
            raise ValueError("initial_mean must be a 1D array")

        self.mean = initial_mean.copy()
        population: List[Tensor] = []

        for _ in range(self.population_size):
            # Sample from multivariate normal distribution
            perturbation = np.random.randn(*self.mean.shape) * self.sigma
            perturbed_params = self.mean + perturbation
            population.append(perturbed_params)

        return population

    def update_mean(self, current_mean: Tensor,
                   population: List[Tensor],
                   fitness_scores: List[float]) -> None:
        """
        Update mean based on fitness scores.

        Args:
            current_mean: Current mean parameters
            population: List of parameter vectors
            fitness_scores: Corresponding fitness scores

        Raises:
            ValueError: If dimensions don't match or fitness_scores length mismatch
        """
        if len(population) != len(fitness_scores):
            raise ValueError("Population and fitness_scores must have same length")

        if len(population) == 0:
            return

        # Select elites
        elite_indices = self._select_elites(fitness_scores)
        elites = [population[i] for i in elite_indices]

        # Update mean toward elites
        self.mean = current_mean + self.learning_rate * np.mean(
            np.array(elites) - current_mean,
            axis=0
        )
        
        # Clamp parameters to reasonable bounds [0.0001, 1.0]
        self.mean = np.clip(self.mean, 0.0001, 1.0)

        # Store average fitness
        self.fitness_history.append(np.mean(fitness_scores))

    def _select_elites(self, fitness_scores: List[float]) -> List[int]:
        """
        Select elite individuals based on fitness.

        Args:
            fitness_scores: List of fitness scores

        Returns:
            List of indices of elite individuals
        """
        # Get indices of top performers
        num_elites = max(1, int(self.population_size * self.elite_fraction))
        elite_indices = np.argsort(fitness_scores)[-num_elites:]

        return list(elite_indices)

    def get_mean(self) -> Tensor:
        """
        Get current mean parameters.

        Returns:
            Current mean parameter vector

        Raises:
            ValueError: If mean has not been initialized
        """
        if self.mean is None:
            raise ValueError("Mean parameters not initialized")
        return self.mean.copy()

    def adapt_sigma(self, fitness_improvement: float) -> None:
        """
        Adapt mutation strength based on fitness improvement.

        Args:
            fitness_improvement: Improvement in fitness over time
        """
        # Simple adaptation rule
        if fitness_improvement > 0:
            self.sigma *= 1.1  # Increase exploration
        else:
            self.sigma *= 0.9  # Decrease exploration

        # Ensure sigma stays within reasonable bounds
        self.sigma = np.clip(self.sigma, 0.001, 1.0)

    def get_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration.

        Returns:
            Dictionary containing optimizer configuration
        """
        return {
            'population_size': self.population_size,
            'sigma': self.sigma,
            'learning_rate': self.learning_rate,
            'elite_fraction': self.elite_fraction,
            'mean_shape': self.mean.shape if self.mean is not None else None,
            'history_length': len(self.fitness_history)
        }

    def reset_state(self) -> None:
        """Reset optimizer state."""
        self.mean = None
        self.fitness_history = []

    def get_fitness_history(self) -> List[float]:
        """
        Get fitness history.

        Returns:
            List of average fitness scores over time
        """
        return self.fitness_history.copy()

    def __str__(self) -> str:
        """String representation of the optimizer."""
        status = "initialized" if self.mean is not None else "uninitialized"
        return f"EvolutionaryStrategy(pop={self.population_size}, σ={self.sigma:.3f}, {status})"