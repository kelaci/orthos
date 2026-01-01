"""
PlasticityController implementation.

Controls plasticity parameters using Evolutionary Strategy for meta-learning.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Callable, Tuple
from orthos.core.base import PlasticComponent
from orthos.core.types import Tensor, PlasticityParams
from orthos.plasticity.es_optimizer import EvolutionaryStrategy

class PlasticityController:
    """
    Controls plasticity parameters using Evolutionary Strategy.

    This class implements meta-learning of plasticity parameters by using
    evolutionary strategies to optimize parameter configurations across
    multiple target modules.

    Attributes:
        target_modules: List of modules to control
        es_optimizer: Evolutionary Strategy optimizer
        plasticity_params: Current plasticity parameters
        adaptation_rate: Rate of parameter adaptation
        exploration_noise: Noise for exploration
        param_history: History of parameter values
        performance_history: History of performance metrics
    """

    def __init__(self, target_modules: List[PlasticComponent],
                 adaptation_rate: float = 0.01, exploration_noise: float = 0.1):
        """
        Initialize PlasticityController.

        Args:
            target_modules: List of PlasticComponent instances to control
            adaptation_rate: Rate of parameter adaptation
            exploration_noise: Noise for exploration in parameter space
        """
        self.target_modules = target_modules
        self.adaptation_rate = adaptation_rate
        self.exploration_noise = exploration_noise

        # Initialize ES optimizer
        self.es_optimizer = EvolutionaryStrategy()

        # Initialize plasticity parameters
        self.plasticity_params = self._initialize_params()

        # History tracking
        self.param_history: List[Tensor] = []
        self.performance_history: List[float] = []

    def _initialize_params(self) -> Tensor:
        """
        Initialize plasticity parameters.

        Returns:
            Initial parameter vector
        """
        # Get parameter dimensions from target modules
        param_dims = sum(len(module.get_plasticity_params()) for module in self.target_modules)

        # Initialize with reasonable defaults
        initial_params = np.ones(param_dims) * 0.01

        return initial_params

    def adapt_plasticity(self, performance_metric: float) -> None:
        """
        Adapt plasticity parameters based on performance.

        This is the main meta-learning loop that uses evolutionary strategies
        to optimize plasticity parameters.

        Args:
            performance_metric: Current performance metric
        """
        # Store current performance
        self.performance_history.append(performance_metric)

        # Sample perturbed parameters
        perturbed_params = self.es_optimizer.generate_population(self.plasticity_params)

        # Evaluate fitness of perturbed parameters
        fitness_scores = []
        for params in perturbed_params:
            # Apply parameters temporarily
            self._apply_params_temporarily(params)

            # Evaluate performance (simplified for now)
            fitness = self._evaluate_performance()
            fitness_scores.append(fitness)

        # Update parameters based on fitness
        self.es_optimizer.update_mean(self.plasticity_params, perturbed_params, fitness_scores)
        self.plasticity_params = self.es_optimizer.get_mean()

        # Store parameter history
        self.param_history.append(self.plasticity_params.copy())

        # Apply updated parameters
        self._apply_params()

    def _apply_params_temporarily(self, params: Tensor) -> None:
        """
        Temporarily apply parameters for evaluation.

        Args:
            params: Parameter vector to apply temporarily
        """
        self._apply_params_specific(params)

    def _apply_params(self) -> None:
        """
        Apply current parameters to target modules.

        This method maps the parameter vector to individual module parameters.
        """
        self._apply_params_specific(self.plasticity_params)

    def _apply_params_specific(self, params: Tensor) -> None:
        """
        Apply specific parameter vector to target modules.

        Args:
            params: Parameter vector to apply
        """
        param_index = 0
        for module in self.target_modules:
            module_params = module.get_plasticity_params()
            num_params = len(module_params)

            # Update each parameter
            for param_name in module_params.keys():
                module_params[param_name] = params[param_index]
                param_index += 1

            module.set_plasticity_params(module_params)

    def _evaluate_performance(self) -> float:
        """
        Evaluate current performance based on plasticity efficiency.

        Returns:
            Performance score
        """
        from orthos.meta_learning.metrics import measure_plasticity_efficiency
        
        scores = []
        for module in self.target_modules:
            scores.append(measure_plasticity_efficiency(module))
            
        return float(np.mean(scores))

    def get_current_params(self) -> Tensor:
        """
        Get current plasticity parameters.

        Returns:
            Current parameter vector
        """
        return self.plasticity_params.copy()

    def get_param_history(self) -> List[Tensor]:
        """
        Get parameter history.

        Returns:
            List of parameter vectors over time
        """
        return [params.copy() for params in self.param_history]

    def get_performance_history(self) -> List[float]:
        """
        Get performance history.

        Returns:
            List of performance metrics over time
        """
        return self.performance_history.copy()

    def reset_state(self) -> None:
        """Reset controller state."""
        self.param_history = []
        self.performance_history = []
        self.plasticity_params = self._initialize_params()
        self._apply_params()

    def get_config(self) -> Dict[str, Any]:
        """
        Get controller configuration.

        Returns:
            Dictionary containing controller configuration
        """
        return {
            'num_target_modules': len(self.target_modules),
            'adaptation_rate': self.adaptation_rate,
            'exploration_noise': self.exploration_noise,
            'es_config': self.es_optimizer.get_config(),
            'param_dimensions': len(self.plasticity_params),
            'history_length': len(self.param_history)
        }

    def set_adaptation_rate(self, rate: float) -> None:
        """
        Set adaptation rate.

        Args:
            rate: New adaptation rate
        """
        self.adaptation_rate = rate
        self.es_optimizer.learning_rate = rate

    def set_exploration_noise(self, noise: float) -> None:
        """
        Set exploration noise.

        Args:
            noise: New exploration noise level
        """
        self.exploration_noise = noise
        self.es_optimizer.sigma = noise

    def __str__(self) -> str:
        """String representation of the controller."""
        return f"PlasticityController({len(self.target_modules)} modules, {len(self.plasticity_params)} params)"