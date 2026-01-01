"""
MetaOptimizer implementation.

Meta-optimizer for plasticity learning and adaptation.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
from orthos.core.types import Tensor, PerformanceMetrics, LearningHistory
from orthos.plasticity.controller import PlasticityController

class MetaOptimizer:
    """
    Meta-optimizer for plasticity learning.

    This class implements a high-level optimization framework that learns
    optimal plasticity parameters through meta-learning across multiple tasks.

    Attributes:
        inner_loop: Plasticity controller for inner optimization
        outer_optimizer: Outer optimization algorithm
        task_distribution: Distribution of tasks
        learning_history: History of learning performance
        meta_parameters: Meta-parameters being optimized
        adaptation_strategy: Current adaptation strategy
    """

    def __init__(self, plasticity_controller: PlasticityController,
                 outer_optimizer: str = 'adam'):
        """
        Initialize MetaOptimizer.

        Args:
            plasticity_controller: PlasticityController instance for inner loop
            outer_optimizer: Outer optimization algorithm ('adam', 'sgd', 'rmsprop')
        """
        self.inner_loop = plasticity_controller
        self.outer_optimizer = outer_optimizer
        self.task_distribution: Optional[Callable] = None
        self.learning_history: LearningHistory = []
        self.meta_parameters: Dict[str, float] = {}
        self.adaptation_strategy: str = 'uniform'

        self._initialize_meta_parameters()

    def _initialize_meta_parameters(self) -> None:
        """Initialize meta-parameters."""
        # Initialize with default values
        self.meta_parameters = {
            'adaptation_rate': 0.01,
            'exploration_noise': 0.1,
            'task_switch_frequency': 10,
            'performance_threshold': 0.8
        }

    def meta_train(self, num_episodes: int, tasks: List[Callable]) -> None:
        """
        Perform meta-training.

        Args:
            num_episodes: Number of training episodes
            tasks: List of task functions
        """
        for episode in range(num_episodes):
            # Sample a task from distribution
            task = self._sample_task(tasks)

            # Inner loop: adapt to task
            task_performance = self.inner_loop_adaptation(task)

            # Outer loop: update meta-parameters
            self.outer_update(task_performance)

            # Record learning history
            self.learning_history.append(task_performance)

            # Log progress
            if episode % 10 == 0:
                print(f"Meta-episode {episode}: Performance = {task_performance:.4f}")

    def _sample_task(self, tasks: List[Callable]) -> Callable:
        """
        Sample a task from task distribution.

        Supports multiple sampling strategies:
        - 'uniform': Equal probability for all tasks
        - 'curriculum': Progressively harder tasks over time
        - 'performance_weighted': Sample based on recent performance

        Args:
            tasks: List of available tasks

        Returns:
            Selected task function
        """
        strategy = self.adaptation_strategy
        
        if strategy == 'uniform' or len(tasks) == 1:
            return np.random.choice(tasks)
        
        elif strategy == 'curriculum':
            # Curriculum learning: progressively sample harder tasks
            if not hasattr(self, '_curriculum_progress'):
                self._curriculum_progress = 0
            
            # Progress increases with each sample
            self._curriculum_progress += 1
            progress_ratio = min(self._curriculum_progress / 100.0, 1.0)
            
            # Bias sampling toward later (harder) tasks as progress increases
            num_tasks = len(tasks)
            weights = np.zeros(num_tasks)
            for i in range(num_tasks):
                # Early: weight early tasks; Late: weight later tasks
                position_ratio = i / (num_tasks - 1) if num_tasks > 1 else 0.5
                weights[i] = np.exp(-((position_ratio - progress_ratio) ** 2) / 0.3)
            
            weights /= weights.sum()
            return np.random.choice(tasks, p=weights)
        
        elif strategy == 'performance_weighted':
            # Sample tasks where performance is lower (focus on weaknesses)
            if not hasattr(self, '_task_performance_buffer'):
                self._task_performance_buffer = {i: [] for i in range(len(tasks))}
            
            # Calculate inverse performance weights (lower performance = higher weight)
            weights = []
            for i in range(len(tasks)):
                perf_history = self._task_performance_buffer.get(i, [])
                if len(perf_history) > 0:
                    avg_perf = np.mean(perf_history[-10:])  # Recent performance
                    weights.append(1.0 / (avg_perf + 0.1))  # Inverse weighting
                else:
                    weights.append(1.0)  # Default weight for unseen tasks
            
            weights = np.array(weights)
            weights /= weights.sum()
            return np.random.choice(tasks, p=weights)
        
        else:
            # Default to uniform
            return np.random.choice(tasks)

    def inner_loop_adaptation(self, task: Callable) -> float:
        """
        Inner loop adaptation to a specific task.

        Implements gradient-free adaptation with early stopping when
        performance converges (change < 1e-4).

        Args:
            task: Task function that takes a step index and returns task data

        Returns:
            Average task performance over adaptation steps
        """
        # Reset plasticity controller for new task
        self.inner_loop.reset_state()

        # Adapt to task for fixed number of steps
        performance = 0.0
        prev_performance = 0.0
        steps_taken = 0
        
        for step in range(int(self.meta_parameters['task_switch_frequency'])):
            # Get task data and performance
            task_data = task(step)
            current_perf = self._evaluate_task_performance(task_data)
            performance += current_perf
            steps_taken = step + 1

            # Adapt plasticity parameters
            self.inner_loop.adapt_plasticity(task_data)
            
            # Early stopping check: convergence detection
            if step > 0 and abs(current_perf - prev_performance) < 1e-4:
                break
            prev_performance = current_perf

        # Return average performance over steps taken
        return performance / steps_taken

    def _evaluate_task_performance(self, task_data: Any) -> float:
        """
        Evaluate performance on a task.

        Args:
            task_data: Task data/result (expected to be an array or dict)

        Returns:
            Performance score
        """
        if isinstance(task_data, (np.ndarray, list)):
            # Higher energy in representation might indicate better feature detection
            return float(np.mean(np.abs(task_data)))
        elif isinstance(task_data, dict) and 'performance' in task_data:
            return float(task_data['performance'])
        
        return 0.5

    def outer_update(self, performance: float) -> None:
        """
        Outer loop update of meta-parameters using Hill Climbing.

        Args:
            performance: Task performance
        """
        # Hill Climbing:
        # 1. Perturb parameters (already done in Exploration phase, implicitly)
        # 2. If better, keep. If worse, revert (or revert with probability)
        
        # Current implementation: Adaptive heuristic
        # If performance is good, we stabilize (reduce noise/learning rate)
        # If performance is bad, we explore (increase noise/learning rate)
        
        target = self.meta_parameters.get('performance_threshold', 0.8)
        
        if performance >= target:
            # Exploitation: Fine-tune
            self.meta_parameters['adaptation_rate'] *= 0.99
            self.meta_parameters['exploration_noise'] *= 0.95
        else:
            # Exploration: Boost plasticity
            self.meta_parameters['adaptation_rate'] *= 1.05
            self.meta_parameters['exploration_noise'] *= 1.1

        # Clip parameters to reasonable bounds
        self.meta_parameters['adaptation_rate'] = np.clip(
            self.meta_parameters['adaptation_rate'], 0.0001, 0.5
        )
        self.meta_parameters['exploration_noise'] = np.clip(
            self.meta_parameters['exploration_noise'], 0.01, 1.0
        )

        # Apply updated parameters to inner loop
        self.inner_loop.set_adaptation_rate(self.meta_parameters['adaptation_rate'])
        self.inner_loop.set_exploration_noise(self.meta_parameters.get('exploration_noise', 0.1))

    def evaluate_meta_performance(self) -> PerformanceMetrics:
        """
        Evaluate meta-learning performance.

        Returns:
            Dictionary of performance metrics
        """
        if not self.learning_history:
            return {
                'average_performance': 0.0,
                'improvement': 0.0,
                'stability': 0.0,
                'convergence_rate': 0.0
            }

        metrics = {
            'average_performance': float(np.mean(self.learning_history)),
            'improvement': float(self.learning_history[-1] - self.learning_history[0]),
            'stability': float(np.std(self.learning_history[-10:])) if len(self.learning_history) >= 10 else 0.0,
            'convergence_rate': float((self.learning_history[-1] - self.learning_history[0]) / len(self.learning_history))
        }

        return metrics

    def set_task_distribution(self, distribution: Callable) -> None:
        """
        Set task distribution function.

        Args:
            distribution: Task distribution function
        """
        self.task_distribution = distribution

    def set_adaptation_strategy(self, strategy: str) -> None:
        """
        Set adaptation strategy for task sampling.

        Supported strategies:
        - 'uniform': Equal probability for all tasks
        - 'curriculum': Progressive difficulty increase
        - 'performance_weighted': Focus on weaker tasks

        Args:
            strategy: Adaptation strategy name

        Raises:
            ValueError: If unknown strategy is specified
        """
        valid_strategies = ['uniform', 'curriculum', 'performance_weighted']
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Valid: {valid_strategies}")
        self.adaptation_strategy = strategy

    def get_meta_parameters(self) -> Dict[str, float]:
        """
        Get current meta-parameters.

        Returns:
            Dictionary of meta-parameters
        """
        return self.meta_parameters.copy()

    def set_meta_parameters(self, params: Dict[str, float]) -> None:
        """
        Set meta-parameters.

        Args:
            params: Dictionary of new meta-parameters
        """
        self.meta_parameters.update(params)

        # Apply relevant parameters to inner loop
        if 'adaptation_rate' in params:
            self.inner_loop.set_adaptation_rate(params['adaptation_rate'])
        if 'exploration_noise' in params:
            self.inner_loop.set_exploration_noise(params['exploration_noise'])

    def reset_state(self) -> None:
        """Reset meta-optimizer state."""
        self.learning_history = []
        self._initialize_meta_parameters()

    def get_config(self) -> Dict[str, Any]:
        """
        Get meta-optimizer configuration.

        Returns:
            Dictionary containing configuration
        """
        return {
            'outer_optimizer': self.outer_optimizer,
            'adaptation_strategy': self.adaptation_strategy,
            'meta_parameters': self.meta_parameters.copy(),
            'learning_history_length': len(self.learning_history),
            'inner_loop_config': self.inner_loop.get_config()
        }

    def __str__(self) -> str:
        """String representation of the meta-optimizer."""
        return f"MetaOptimizer({self.outer_optimizer}, {len(self.learning_history)} episodes)"