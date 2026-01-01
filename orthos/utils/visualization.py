"""
Visualization utilities for ORTHOS.

This module provides functions for visualizing hierarchical representations,
learning curves, and other aspects of ORTHOS's operation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from orthos.core.types import Tensor

def plot_hierarchy_representations(representations: Dict[int, List[Tensor]],
                                 title: str = "Hierarchical Representations",
                                 figsize: tuple = (12, 8)) -> None:
    """
    Plot representations from different hierarchical levels.

    Args:
        representations: Dictionary of representations by level
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    num_levels = len(representations)
    cols = min(3, num_levels)
    rows = (num_levels + cols - 1) // cols

    for i, (level_id, reps) in enumerate(representations.items()):
        # Average representation over time
        if len(reps) > 0:
            avg_rep = np.mean(reps, axis=0)
        else:
            continue

        plt.subplot(rows, cols, i + 1)

        # Reshape for visualization
        if len(avg_rep.shape) == 1:
            # 1D representation
            plt.plot(avg_rep)
            plt.title(f"Level {level_id} (1D)")
        elif len(avg_rep.shape) == 2:
            # 2D representation
            plt.imshow(avg_rep, aspect='auto', cmap='viridis')
            plt.title(f"Level {level_id}")
            plt.colorbar()
        else:
            # Flatten higher dimensions
            plt.plot(avg_rep.flatten())
            plt.title(f"Level {level_id} (flattened)")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(learning_history: List[float],
                       title: str = "Learning Curve",
                       figsize: tuple = (10, 6)) -> None:
    """
    Plot learning curve.

    Args:
        learning_history: History of learning performance
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(learning_history, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel("Episode/Iteration")
    plt.ylabel("Performance")
    plt.grid(True, alpha=0.3)

    # Add some statistics
    if len(learning_history) > 1:
        plt.axhline(y=np.mean(learning_history), color='r', linestyle='--',
                   label=f'Average: {np.mean(learning_history):.3f}')
        plt.axhline(y=learning_history[-1], color='g', linestyle=':',
                   label=f'Final: {learning_history[-1]:.3f}')
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_weight_matrix(weights: Tensor, title: str = "Weight Matrix",
                      figsize: tuple = (8, 8)) -> None:
    """
    Plot weight matrix.

    Args:
        weights: Weight matrix
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if len(weights.shape) == 2:
        plt.imshow(weights, cmap='viridis', aspect='auto')
        plt.title(title)
        plt.colorbar()
    elif len(weights.shape) == 1:
        plt.plot(weights)
        plt.title(f"{title} (1D)")
    else:
        # Flatten higher dimensions
        plt.plot(weights.flatten())
        plt.title(f"{title} (flattened)")

    plt.tight_layout()
    plt.show()

def plot_plasticity_parameters(params_history: List[Dict[str, float]],
                             title: str = "Plasticity Parameters",
                             figsize: tuple = (12, 6)) -> None:
    """
    Plot plasticity parameter evolution.

    Args:
        params_history: History of plasticity parameters
        title: Plot title
        figsize: Figure size
    """
    if not params_history:
        print("No parameter history to plot")
        return

    param_names = list(params_history[0].keys())
    num_params = len(param_names)

    plt.figure(figsize=figsize)

    for param_name in param_names:
        values = [params[param_name] for params in params_history]
        plt.plot(values, label=param_name)

    plt.title(title)
    plt.xlabel("Update Step")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_parameter_convergence(param_history: List[Dict[str, float]],
                             param_name: str,
                             title: str = "Parameter Convergence",
                             figsize: tuple = (10, 6)) -> None:
    """
    Plot convergence of a specific parameter.

    Args:
        param_history: History of parameter values
        param_name: Name of parameter to plot
        title: Plot title
        figsize: Figure size
    """
    if not param_history:
        print("No parameter history to plot")
        return

    if param_name not in param_history[0]:
        print(f"Parameter {param_name} not found in history")
        return

    values = [params[param_name] for params in param_history]

    plt.figure(figsize=figsize)
    plt.plot(values, 'b-', linewidth=2)
    plt.title(f"{title}: {param_name}")
    plt.xlabel("Update Step")
    plt.ylabel("Parameter Value")

    # Add convergence indicators
    if len(values) > 10:
        final_value = values[-1]
        convergence_threshold = 0.01 * (max(values) - min(values))

        # Find when it converges
        for i in range(len(values)):
            if abs(values[i] - final_value) < convergence_threshold:
                plt.axvline(x=i, color='r', linestyle='--',
                           label=f'Converged at step {i}')
                break

        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_hierarchy_communication(communication_data: Dict[int, List[Dict[str, Any]]],
                               title: str = "Hierarchy Communication",
                               figsize: tuple = (12, 8)) -> None:
    """
    Plot hierarchy communication patterns.

    Args:
        communication_data: Dictionary of communication data by level
        title: Plot title
        figsize: Figure size
    """
    if not communication_data:
        print("No communication data to plot")
        return

    plt.figure(figsize=figsize)

    levels = sorted(communication_data.keys())
    num_levels = len(levels)

    for i, level_id in enumerate(levels):
        comm_events = communication_data[level_id]
        if not comm_events:
            continue

        # Extract communication metrics
        timestamps = [event['timestamp'] for event in comm_events]
        sizes = [event['data_size'] for event in comm_events]

        plt.subplot(num_levels, 1, i + 1)
        plt.scatter(timestamps, sizes, alpha=0.6)
        plt.title(f"Level {level_id} Communication")
        plt.xlabel("Time")
        plt.ylabel("Data Size")
        plt.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(performance_data: Dict[str, List[float]],
                              title: str = "Performance Comparison",
                              figsize: tuple = (10, 6)) -> None:
    """
    Plot comparison of different performance metrics.

    Args:
        performance_data: Dictionary of performance metrics
        title: Plot title
        figsize: Figure size
    """
    if not performance_data:
        print("No performance data to plot")
        return

    plt.figure(figsize=figsize)

    for metric_name, values in performance_data.items():
        plt.plot(values, label=metric_name)

    plt.title(title)
    plt.xlabel("Episode/Iteration")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(correlation_matrix: Tensor,
                           labels: Optional[List[str]] = None,
                           title: str = "Correlation Matrix",
                           figsize: tuple = (10, 8)) -> None:
    """
    Plot correlation matrix.

    Args:
        correlation_matrix: Correlation matrix to plot
        labels: Optional labels for axes
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if len(correlation_matrix.shape) != 2 or correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        print("Input must be a square 2D matrix")
        return

    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.colorbar()

    if labels:
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)

    plt.tight_layout()
    plt.show()