"""
HierarchyManager implementation.

Manager for hierarchical processing with multiple levels and temporal abstraction.
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from orthos.core.types import Tensor
from orthos.hierarchy.level import HierarchicalLevel

class HierarchyManager:
    """
    Manages multiple hierarchical levels.

    This class coordinates processing across multiple hierarchical levels,
    handling temporal abstraction, inter-level communication, and state management.

    Attributes:
        levels: List of hierarchical levels
        communication_schedule: Communication timing configuration
        global_time_step: Current global time step
        performance_metrics: Performance tracking
    """

    def __init__(self):
        """Initialize HierarchyManager."""
        self.levels: List[HierarchicalLevel] = []
        self.communication_schedule: Dict[str, Any] = {}
        self.global_time_step: int = 0
        self.performance_metrics: Dict[str, List[float]] = {
            'processing_time': [],
            'communication_overhead': [],
            'memory_usage': []
        }

    def add_level(self, level: HierarchicalLevel) -> None:
        """
        Add a level to the hierarchy.

        Args:
            level: HierarchicalLevel instance to add

        Raises:
            ValueError: If level_id already exists or levels are not contiguous
        """
        # Check for duplicate level_id
        if any(existing.level_id == level.level_id for existing in self.levels):
            raise ValueError(f"Level with id {level.level_id} already exists")

        self.levels.append(level)

        # Sort levels by level_id
        self.levels.sort(key=lambda x: x.level_id)

        # Update parent/child relationships
        self._update_hierarchy_relationships()

    def _update_hierarchy_relationships(self) -> None:
        """
        Update parent/child relationships between levels.

        This method automatically establishes the hierarchical structure
        based on level_ids.
        """
        for i, level in enumerate(self.levels):
            # Set parent (level above)
            if i > 0 and self.levels[i-1].level_id == level.level_id - 1:
                level.parent_level = self.levels[i-1]

            # Set children (levels below)
            level.child_levels = [
                self.levels[j] for j in range(i+1, len(self.levels))
                if self.levels[j].level_id == level.level_id + 1
            ]

    def process_hierarchy(self, input_data: Tensor, time_steps: int) -> Dict[int, List[Tensor]]:
        """
        Process input through the entire hierarchy.

        Args:
            input_data: Input data sequence (time_steps, input_size)
            time_steps: Number of time steps to process

        Returns:
            Dictionary of representations at each level

        Raises:
            ValueError: If no levels are defined or input shape is invalid
        """
        if len(self.levels) == 0:
            raise ValueError("No levels defined in hierarchy")

        if input_data.shape[0] != time_steps or len(input_data.shape) != 2:
            raise ValueError(f"Input data shape {input_data.shape} doesn't match expected ({time_steps}, input_size)")

        representations: Dict[int, List[Tensor]] = {level.level_id: [] for level in self.levels}

        for t in range(time_steps):
            current_input = input_data[t]

            # Process through each level
            level_output = None
            for level in self.levels:
                if level.level_id == 0:
                    # Input level
                    level_output = level.process_time_step(current_input, t)
                else:
                    # Higher levels process previous level's output
                    if level_output is not None:
                        level_output = level.process_time_step(level_output, t)
                    else:
                        # Skip if no input from previous level
                        continue

                representations[level.level_id].append(level_output)

            # Hierarchical communication
            self._hierarchical_communication(t)

            self.global_time_step = t

        return representations

    def _hierarchical_communication(self, t: int) -> None:
        """
        Handle communication between hierarchical levels.

        Args:
            t: Current time step
        """
        # 1. Bottom-up communication (Level N -> Level N-1)
        # Levels send their summary/representation upwards
        for i in range(len(self.levels) - 1, 0, -1):
            current_level = self.levels[i]
            if current_level.parent_level is not None:
                comm_data = current_level.communicate_with_parent()
                if comm_data:
                    # Parent receives data from child
                    current_level.parent_level.communication_buffer[f'child_{current_level.level_id}'] = comm_data

        # 2. Top-down communication (Level N-1 -> Level N)
        # Levels send contextual modulation downwards
        for i in range(len(self.levels) - 1):
            current_level = self.levels[i]
            if current_level.child_levels:
                top_down_data = {
                    'level_id': current_level.level_id,
                    'representation': current_level.get_representation(),
                    'time_step': t
                }
                current_level.communicate_with_children(top_down_data)

    def get_all_representations(self) -> Dict[int, Optional[Tensor]]:
        """
        Get current representations from all levels.

        Returns:
            Dictionary of current representations by level_id
        """
        return {level.level_id: level.get_representation() for level in self.levels}

    def reset_state(self) -> None:
        """Reset state of all levels."""
        for level in self.levels:
            level.reset_state()
        self.global_time_step = 0
        self.performance_metrics = {
            'processing_time': [],
            'communication_overhead': [],
            'memory_usage': []
        }

    def get_level(self, level_id: int) -> Optional[HierarchicalLevel]:
        """
        Get level by id.

        Args:
            level_id: Level identifier

        Returns:
            HierarchicalLevel instance or None if not found
        """
        for level in self.levels:
            if level.level_id == level_id:
                return level
        return None

    def remove_level(self, level_id: int) -> bool:
        """
        Remove level from hierarchy.

        Args:
            level_id: Level identifier

        Returns:
            True if level was removed, False if not found
        """
        for i, level in enumerate(self.levels):
            if level.level_id == level_id:
                self.levels.pop(i)
                self._update_hierarchy_relationships()
                return True
        return False

    def get_config(self) -> Dict[str, Any]:
        """
        Get hierarchy configuration.

        Returns:
            Dictionary containing hierarchy configuration
        """
        return {
            'num_levels': len(self.levels),
            'level_configs': [level.get_config() for level in self.levels],
            'communication_schedule': self.communication_schedule,
            'global_time_step': self.global_time_step
        }

    def set_communication_schedule(self, schedule: Dict[str, Any]) -> None:
        """
        Set communication schedule.

        Args:
            schedule: Communication schedule configuration
        """
        self.communication_schedule = schedule

    def log_performance_metric(self, metric_name: str, value: float) -> None:
        """
        Log performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        self.performance_metrics[metric_name].append(value)

    def get_performance_metrics(self) -> Dict[str, List[float]]:
        """
        Get performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return {k: v.copy() for k, v in self.performance_metrics.items()}

    def __str__(self) -> str:
        """String representation of the hierarchy."""
        level_strs = [str(level) for level in self.levels]
        return "HierarchyManager:\n  " + "\n  ".join(level_strs)