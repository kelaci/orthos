"""
ConsensusHierarchyManager implementation.

Extends the base HierarchyManager to include cross-level belief aggregation
using the HierarchicalConsensus engine.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple

from orthos.hierarchy.manager import HierarchyManager
from orthos.hierarchy.level import HierarchicalLevel
from orthos.consensus.engine import HierarchicalConsensus, LevelPrediction, ConsensusResult


class ConsensusHierarchyManager(HierarchyManager):
    """
    Managers hierarchy with inter-level consensus capabilities.

    Aggregates beliefs and predictions from multiple hierarchical levels
    to produce a robust global estimate.
    """

    def __init__(
        self,
        consensus_method: str = 'weighted_vote',
        outlier_threshold: float = 3.0,
        min_agreement: float = 0.6,
        use_temporal_weighting: bool = False,
        auto_projection: bool = False
    ):
        """
        Initialize ConsensusHierarchyManager.

        Args:
            consensus_method: Method for aggregation ('mean', 'median', 'weighted_vote').
            outlier_threshold: Robust Z-score threshold for outlier detection.
            min_agreement: Minimum agreement score to consider hierarchy stable.
            use_temporal_weighting: Enable temporal smoothing of consensus.
            auto_projection: Automatically project lower-dimensional outputs to higher dimensions.
        """
        super().__init__()
        self.consensus_engine = HierarchicalConsensus(
            outlier_threshold=outlier_threshold,
            min_agreement=min_agreement,
            use_temporal_weighting=use_temporal_weighting
        )
        self.consensus_method = consensus_method
        self.consensus_history: List[ConsensusResult] = []
        self.auto_projection = auto_projection
        self.consensus_prior: Optional[np.ndarray] = None

    def _validate_dimensions(self, level_predictions: List[LevelPrediction]) -> Tuple[int, bool]:
        """
        Validate that all level predictions have compatible dimensions.

        Args:
            level_predictions: List of level predictions to validate.

        Returns:
            Tuple of (target_dimension, is_valid).

        Raises:
            ValueError: If dimensions are incompatible and auto_projection is disabled.
        """
        if not level_predictions:
            raise ValueError("No level predictions provided")

        dimensions = []
        for pred in level_predictions:
            pred_array = np.asarray(pred.prediction)
            if pred_array.ndim == 0:
                dimensions.append(1)  # Scalar treated as 1-dimensional
            else:
                dimensions.append(pred_array.shape[0])
        
        unique_dims = set(dimensions)

        if len(unique_dims) == 1:
            return unique_dims.pop(), True
        elif self.auto_projection:
            # Project all to the maximum dimension
            target_dim = max(unique_dims)
            return target_dim, False
        else:
            raise ValueError(
                f"Level output dimensions are incompatible: {unique_dims}. "
                f"Either ensure all levels output the same dimension or enable auto_projection=True."
            )

    def _project_prediction(
        self,
        prediction: np.ndarray,
        target_dim: int
    ) -> np.ndarray:
        """
        Project a prediction to the target dimension.

        Args:
            prediction: Original prediction vector.
            target_dim: Target dimension.

        Returns:
            Projected prediction vector.
        """
        pred_array = np.asarray(prediction)
        if pred_array.ndim == 0:
            pred_array = pred_array.reshape(1)
        
        current_dim = pred_array.shape[0]
        
        if current_dim == target_dim:
            return pred_array
        elif current_dim < target_dim:
            # Upsample with repetition (simple but effective for many cases)
            repetition = target_dim // current_dim
            remainder = target_dim % current_dim
            projected = np.tile(pred_array, repetition)
            if remainder > 0:
                projected = np.concatenate([projected, pred_array[:remainder]])
            return projected
        else:
            # Downsample by taking every nth element
            step = current_dim // target_dim
            projected = pred_array[::step][:target_dim]
            return projected

    def get_consensus_prediction(
        self,
        input_data: np.ndarray,
        method: Optional[str] = None
    ) -> ConsensusResult:
        """
        Generate a consensus prediction by aggregating state from all levels.

        Args:
            input_data: Current input data step.
            method: Override for the default consensus method.

        Returns:
            ConsensusResult containing the aggregated prediction and metrics.
        """
        method = method or self.consensus_method
        level_predictions: List[LevelPrediction] = []

        for level_obj in self.levels:
            # We check for 'forward_filtered' or similar if it exists,
            # otherwise fallback to base representation.
            # Note: In a cleaner impl, we'd use a shared interface.
            
            if hasattr(level_obj, 'forward_filtered'):
                # Handle filtered levels (e.g., using Kalman/Particle filters)
                pred, unc = getattr(level_obj, 'forward_filtered')(input_data)
                conf = getattr(level_obj, 'get_confidence', lambda: 0.5)()
            else:
                # Fallback for standard levels
                pred = level_obj.process_time_step(input_data, self.global_time_step)
                unc = 0.5 # Default uncertainty
                conf = 0.5 # Default confidence

            level_predictions.append(LevelPrediction(
                level=level_obj.level_id,
                prediction=pred,
                confidence=float(conf),
                uncertainty=float(unc)
            ))

        # Validate and project dimensions if needed
        target_dim, is_valid = self._validate_dimensions(level_predictions)
        if not is_valid:
            # Project all predictions to target dimension
            for pred in level_predictions:
                pred.prediction = self._project_prediction(pred.prediction, target_dim)

        # Aggregate using consensus engine
        result = self.consensus_engine.aggregate(level_predictions, method=method) # type: ignore
        
        # Store as prior for next iteration (top-down feedback)
        self.consensus_prior = result.prediction
        
        self.consensus_history.append(result)
        self.global_time_step += 1

        return result

    def distribute_prior(self, levels: List[HierarchicalLevel]) -> None:
        """
        Distribute the consensus result back to lower levels as a top-down prior.

        This enables the feedback loop where high-level consensus influences
        lower-level processing in the next timestep.

        Args:
            levels: List of HierarchicalLevel objects to receive the prior.
        """
        if self.consensus_prior is None:
            return

        for level_obj in levels:
            # Check if level accepts top-down priors
            if hasattr(level_obj, 'set_top_down_prior'):
                # Project prior to the level's output dimension if needed
                level_dim = getattr(level_obj, 'output_size', None)
                if level_dim is not None:
                    prior_projected = self._project_prediction(self.consensus_prior, level_dim)
                    level_obj.set_top_down_prior(prior_projected)
            elif hasattr(level_obj, 'forward_filtered'):
                # For FilteredHierarchicalLevel, pass as top_down_prior parameter
                # Store it as an attribute that will be used in next forward_filtered call
                level_obj._top_down_prior = self._project_prediction(
                    self.consensus_prior,
                    getattr(level_obj, 'output_size', self.consensus_prior.shape[0])
                )

    def is_hierarchy_stable(self) -> bool:
        """
        Check if the hierarchy consensus is stable based on recent history.

        Returns:
            True if the average agreement score over the last 5 steps 
            exceeds the minimum threshold.
        """
        if len(self.consensus_history) < 5:
            return False
            
        recent_scores = [r.agreement_score for r in self.consensus_history[-5:]]
        return float(np.mean(recent_scores)) > self.consensus_engine.min_agreement

    def reset_state(self) -> None:
        """Reset manager and consensus history."""
        super().reset_state()
        self.consensus_history = []
        self.consensus_engine.reset_state()
        self.consensus_prior = None
