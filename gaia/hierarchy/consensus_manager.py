"""
ConsensusHierarchyManager implementation.

Extends the base HierarchyManager to include cross-level belief aggregation
using the HierarchicalConsensus engine.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple

from gaia.hierarchy.manager import HierarchyManager
from gaia.hierarchy.level import HierarchicalLevel
from gaia.consensus.engine import HierarchicalConsensus, LevelPrediction, ConsensusResult


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
        use_temporal_weighting: bool = False
    ):
        """
        Initialize ConsensusHierarchyManager.

        Args:
            consensus_method: Method for aggregation ('mean', 'median', 'weighted_vote').
            outlier_threshold: Robust Z-score threshold for outlier detection.
            min_agreement: Minimum agreement score to consider hierarchy stable.
            use_temporal_weighting: Enable temporal smoothing of consensus.
        """
        super().__init__()
        self.consensus_engine = HierarchicalConsensus(
            outlier_threshold=outlier_threshold,
            min_agreement=min_agreement,
            use_temporal_weighting=use_temporal_weighting
        )
        self.consensus_method = consensus_method
        self.consensus_history: List[ConsensusResult] = []

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

        # Aggregate using consensus engine
        # We use the aggregate method directly
        result = self.consensus_engine.aggregate(level_predictions, method=method) # type: ignore
        
        self.consensus_history.append(result)
        self.global_time_step += 1

        return result

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
