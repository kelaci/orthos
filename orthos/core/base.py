"""
Base classes for ORTHOS modules.

This module defines the abstract base classes that form the foundation of
ORTHOS's hierarchical neural architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class Module(ABC):
    """
    Base class for all ORTHOS modules.

    All components in ORTHOS inherit from this base class, providing
    a consistent interface for forward/backward passes, updates, and state management.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the module.

        Args:
            x: Input tensor

        Returns:
            Output tensor after processing
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient tensor from subsequent layer

        Returns:
            Gradient tensor for preceding layer
        """
        pass

    @abstractmethod
    def update(self, lr: float) -> None:
        """
        Update module parameters.

        Args:
            lr: Learning rate
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset internal state of the module.

        This should clear any accumulated state, activity traces, etc.
        """
        pass

class Layer(Module):
    """
    Base class for all layer implementations.

    Layers are the fundamental processing units in ORTHOS, implementing
    specific transformations and learning rules.
    """

    @abstractmethod
    def activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply activation function to input.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        pass

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """
        Get current weight matrix.

        Returns:
            Weight matrix
        """
        pass

    @abstractmethod
    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set weight matrix.

        Args:
            weights: New weight matrix
        """
        pass

class PlasticComponent(ABC):
    """
    Base class for components with plasticity.

    Plastic components can adapt their behavior through learning rules
    and have configurable plasticity parameters.
    """

    @abstractmethod
    def get_plasticity_params(self) -> Dict[str, float]:
        """
        Get current plasticity parameters.

        Returns:
            Dictionary of plasticity parameters
        """
        pass

    @abstractmethod
    def set_plasticity_params(self, params: Dict[str, float]) -> None:
        """
        Set plasticity parameters.

        Args:
            params: Dictionary of new plasticity parameters
        """
        pass

class HierarchicalLevel(ABC):
    """
    Base class for hierarchical levels.

    Hierarchical levels process information at different temporal
    resolutions and abstraction levels.
    """

    @abstractmethod
    def process_time_step(self, input_data: np.ndarray, t: int) -> np.ndarray:
        """
        Process a single time step.

        Args:
            input_data: Input data for this time step
            t: Global time step index

        Returns:
            Processed output
        """
        pass

    @abstractmethod
    def get_representation(self) -> np.ndarray:
        """
        Get current representation.

        Returns:
            Current representation tensor
        """
        pass