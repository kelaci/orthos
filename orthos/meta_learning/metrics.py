"""
Performance metrics for meta-learning evaluation.

This module provides functions for evaluating the performance of
meta-learning systems and plasticity adaptation.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from orthos.core.base import Module
from orthos.core.types import PerformanceMetrics, LearningHistory

def measure_plasticity_efficiency(module: Module) -> float:
    """
    Measure plasticity efficiency of a module.

    Evaluates how effectively weights adapt to input statistics.
    """
    if hasattr(module, 'activity_history') and len(module.activity_history) > 1:
        # Measure variance in representations - higher variance in response 
        # to diverse inputs indicates efficient plasticity
        representations = [h[1] for h in module.activity_history]
        return float(np.mean(np.std(representations, axis=0)))
    return 0.5

def measure_adaptation_speed(module: Module) -> float:
    """
    Measure adaptation speed of a module.

    Evaluates rate of weight change relative to input intensity.
    """
    if hasattr(module, 'activity_history') and len(module.activity_history) > 2:
        # Measure cumulative change in activity traces
        activities = np.array([h[1] for h in module.activity_history])
        diffs = np.linalg.norm(np.diff(activities, axis=0), axis=1)
        return float(np.mean(diffs))
    return 0.5

def measure_stability_plasticity_tradeoff(module: Module) -> float:
    """
    Measure stability-plasticity tradeoff.

    Evaluates the balance between maintaining stable
    representations and adapting to new information.
    """
    speed = measure_adaptation_speed(module)
    if hasattr(module, 'activity_history') and len(module.activity_history) > 5:
        # High stability = low variance in steady state
        recent_activity = np.array([h[1] for h in module.activity_history[-5:]])
        stability = 1.0 / (np.mean(np.std(recent_activity, axis=0)) + 1e-8)
        # Harmonized mean of speed and stability
        return float(2 * (speed * stability) / (speed + stability + 1e-8))
    return 0.5

def evaluate_meta_learning_curve(learning_history: LearningHistory) -> PerformanceMetrics:
    """
    Evaluate meta-learning curve.

    This function analyzes the learning curve to assess meta-learning
    performance, including convergence properties and stability.

    Args:
        learning_history: History of learning performance

    Returns:
        Dictionary of evaluation metrics

    Raises:
        ValueError: If learning history is too short
    """
    if len(learning_history) < 2:
        raise ValueError("Learning history too short for evaluation")

    # Calculate basic statistics
    final_performance = learning_history[-1]
    initial_performance = learning_history[0]
    improvement = final_performance - initial_performance
    convergence_rate = improvement / len(learning_history)

    # Calculate stability (variance in recent performance)
    stability = np.std(learning_history[-min(10, len(learning_history)):])

    # Calculate consistency (improvement consistency)
    improvements = np.diff(learning_history)
    consistency = np.mean(improvements > 0)  # Fraction of positive improvements

    return {
        'final_performance': float(final_performance),
        'initial_performance': float(initial_performance),
        'improvement': float(improvement),
        'convergence_rate': float(convergence_rate),
        'stability': float(stability),
        'consistency': float(consistency),
        'average_performance': float(np.mean(learning_history))
    }

def evaluate_adaptation_robustness(performance_history: List[float],
                                 task_boundaries: List[int]) -> PerformanceMetrics:
    """
    Evaluate robustness of adaptation across task switches.

    Measures how well the system adapts when switching between tasks,
    including adaptation speed, recovery time, and performance drops.

    Args:
        performance_history: History of performance metrics
        task_boundaries: Indices where task switches occurred

    Returns:
        Dictionary of robustness metrics:
        - adaptation_speed: Average speed of recovery (1/steps)
        - recovery_time: Average steps to recover pre-switch performance
        - performance_drop: Average performance decrease at switch
        - overall_robustness: Combined robustness score
    """
    # Placeholder implementation
    if not task_boundaries or len(performance_history) < 2:
        return {
            'adaptation_speed': 0.0,
            'recovery_time': 0.0,
            'performance_drop': 0.0,
            'overall_robustness': 0.0
        }

    # Calculate metrics across task boundaries
    adaptation_speeds = []
    recovery_times = []
    performance_drops = []

    for boundary in task_boundaries:
        if boundary + 1 >= len(performance_history):
            continue

        # Performance drop at task switch
        drop = performance_history[boundary] - performance_history[boundary + 1]
        performance_drops.append(drop)

        # Recovery time (steps to return to pre-switch performance)
        pre_performance = performance_history[boundary]
        recovery_time = 0
        for i in range(boundary + 1, min(boundary + 10, len(performance_history))):
            if performance_history[i] >= pre_performance:
                recovery_time = i - boundary
                break

        if recovery_time > 0:
            recovery_times.append(recovery_time)
            adaptation_speeds.append(1.0 / recovery_time)

    avg_adaptation_speed = np.mean(adaptation_speeds) if adaptation_speeds else 0.0
    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0
    avg_performance_drop = np.mean(performance_drops) if performance_drops else 0.0

    # Overall robustness score
    robustness = 0.0
    if len(recovery_times) > 0 and len(performance_drops) > 0:
        robustness = (avg_adaptation_speed * (1 - avg_performance_drop)) / (avg_recovery_time + 1)

    return {
        'adaptation_speed': float(avg_adaptation_speed),
        'recovery_time': float(avg_recovery_time),
        'performance_drop': float(avg_performance_drop),
        'overall_robustness': float(robustness)
    }

def evaluate_parameter_convergence(param_history: List[Dict[str, float]]) -> PerformanceMetrics:
    """
    Evaluate parameter convergence properties.

    Analyzes the convergence behavior of meta-learning parameters,
    including speed, stability, and oscillation characteristics.

    Args:
        param_history: History of parameter values (list of dicts)

    Returns:
        Dictionary of convergence metrics:
        - convergence_speed: How quickly parameters approach final values
        - stability: Variance in recent parameter values
        - oscillation: Frequency of direction changes
        - final_values: Final parameter values
    """
    if not param_history or len(param_history) < 2:
        return {
            'convergence_speed': 0.0,
            'stability': 0.0,
            'oscillation': 0.0,
            'final_values': {}
        }

    # Get parameter names
    param_names = list(param_history[0].keys())
    metrics = {'final_values': {}}

    for param_name in param_names:
        # Extract parameter values over time
        values = [step[param_name] for step in param_history]

        # Calculate convergence metrics
        final_value = values[-1]
        initial_value = values[0]
        change = abs(final_value - initial_value)

        # Convergence speed (how quickly it approaches final value)
        convergence_speed = 0.0
        if change > 1e-6:
            for i in range(1, len(values)):
                if abs(values[i] - final_value) < 0.1 * change:
                    convergence_speed = 1.0 / i
                    break

        # Stability (variance in recent values)
        stability = np.std(values[-min(10, len(values)):])

        # Oscillation (number of direction changes)
        direction_changes = 0
        for i in range(1, len(values) - 1):
            if (values[i] - values[i-1]) * (values[i+1] - values[i]) < 0:
                direction_changes += 1
        oscillation = direction_changes / len(values)

        metrics['final_values'][param_name] = float(final_value)
        metrics[f'{param_name}_convergence_speed'] = float(convergence_speed)
        metrics[f'{param_name}_stability'] = float(stability)
        metrics[f'{param_name}_oscillation'] = float(oscillation)

    return metrics