"""
HierarchicalLevel implementation.

Implementation of hierarchical processing levels with temporal abstraction.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union
from orthos.core.base import HierarchicalLevel as BaseHierarchicalLevel
from orthos.core.types import Tensor
from orthos.core.base import Layer

class HierarchicalLevel(BaseHierarchicalLevel):
    """
    Implementation of a hierarchical level.

    This class represents a single level in the hierarchical processing
    architecture, with configurable temporal resolution and processing layers.

    Attributes:
        level_id: Level identifier
        input_size: Size of input features
        output_size: Size of output features
        temporal_resolution: Time compression factor
        parent_level: Reference to parent level
        child_levels: List of child levels
        processing_layers: List of processing layers
        current_representation: Current representation
        time_step: Current time step
        communication_buffer: Buffer for inter-level communication
    """

    def __init__(self, level_id: int, input_size: int, output_size: int,
                 temporal_resolution: int = 1):
        """
        Initialize HierarchicalLevel.

        Args:
            level_id: Level identifier
            input_size: Size of input features
            output_size: Size of output features
            temporal_resolution: Time compression factor (process every N steps)
        """
        self.level_id = level_id
        self.input_size = input_size
        self.output_size = output_size
        self.temporal_resolution = temporal_resolution
        self.parent_level: Optional['HierarchicalLevel'] = None
        self.child_levels: List['HierarchicalLevel'] = []
        self.processing_layers: List[Layer] = []
        self.current_representation: Optional[Tensor] = None
        self.time_step: int = 0
        self.communication_buffer: Dict[str, Any] = {}

    def add_layer(self, layer: Layer) -> None:
        """
        Add a processing layer to this level.

        Args:
            layer: Layer instance to add

        Raises:
            ValueError: If layer input/output size doesn't match level configuration
        """
        # Check layer compatibility
        if len(self.processing_layers) == 0:
            # First layer - check input size
            if layer.input_size != self.input_size:
                raise ValueError(f"First layer input size {layer.input_size} doesn't match level input size {self.input_size}")
        else:
            # Subsequent layers - check compatibility with previous layer
            prev_layer = self.processing_layers[-1]
            if layer.input_size != prev_layer.output_size:
                raise ValueError(f"Layer input size {layer.input_size} doesn't match previous output size {prev_layer.output_size}")

        self.processing_layers.append(layer)

        # Update level output size if this is the last layer
        if layer.output_size != self.output_size:
            self.output_size = layer.output_size

    def process_time_step(self, input_data: Tensor, t: int) -> Tensor:
        """
        Process a single time step.

        Args:
            input_data: Input data for this time step
            t: Global time step index

        Returns:
            Processed output

        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]

        if input_data.shape[1] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {input_data.shape[1]}")

        # Only process every temporal_resolution time steps
        if t % self.temporal_resolution != 0:
            return self.current_representation if self.current_representation is not None else input_data

        # Process through all layers
        output = input_data
        for layer in self.processing_layers:
            output = layer.forward(output)

        # Store current representation
        self.current_representation = output
        self.time_step = t

        return output

    def get_representation(self) -> Optional[Tensor]:
        """
        Get current representation.

        Returns:
            Current representation tensor, or None if not processed yet
        """
        return self.current_representation

    def communicate_with_parent(self) -> Optional[Dict[str, Any]]:
        """
        Communicate with parent level.

        Returns:
            Data dictionary to send to parent, or None
        """
        if self.parent_level is None:
            return None

        # Send current representation and any relevant metrics
        communication_data = {
            'level_id': self.level_id,
            'representation': self.current_representation,
            'time_step': self.time_step,
            'energy': np.sum(np.square(self.current_representation)) if self.current_representation is not None else 0
        }

        return communication_data

    def communicate_with_children(self, data: Dict[str, Any]) -> None:
        """
        Communicate with child levels.

        Args:
            data: Data received from parent
        """
        # Store received data in communication buffer (Top-down modulation)
        self.communication_buffer.update(data)

        # Apply top-down modulation to children if they exist
        if 'representation' in data:
            for child in self.child_levels:
                child.receive_top_down(data)

    def receive_top_down(self, data: Dict[str, Any]) -> None:
        """
        Receive and process top-down modulation data.

        Args:
            data: Top-down data from parent
        """
        self.communication_buffer.update(data)

    def reset_state(self) -> None:
        """Reset internal state."""
        for layer in self.processing_layers:
            layer.reset_state()
        self.current_representation = None
        self.time_step = 0
        self.communication_buffer = {}

    def get_config(self) -> Dict[str, Any]:
        """
        Get level configuration.

        Returns:
            Dictionary containing level configuration
        """
        return {
            'level_id': self.level_id,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'temporal_resolution': self.temporal_resolution,
            'num_layers': len(self.processing_layers),
            'layer_configs': [layer.get_config() for layer in self.processing_layers]
        }

    def get_communication_buffer(self) -> Dict[str, Any]:
        """
        Get communication buffer.

        Returns:
            Communication buffer dictionary
        """
        return self.communication_buffer.copy()

    def __str__(self) -> str:
        """String representation of the level."""
        layer_str = " → ".join(str(layer) for layer in self.processing_layers)
        return f"Level {self.level_id} ({self.input_size}→{self.output_size}, res={self.temporal_resolution}): {layer_str}"