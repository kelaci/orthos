"""
Hierarchical Consensus Engine.

This module provides the core logic for aggregating predictions from multiple
hierarchical levels into a single robust estimate. It handles uncertainty,
confidence weighting, and outlier detection.
"""

import numpy as np
from typing import List, Optional, Literal, Dict, Any, Union
from dataclasses import dataclass

from gaia.core.base import Module


@dataclass
class LevelPrediction:
    """
    Container for a prediction from a single hierarchical level.
    
    Attributes:
        level (int): The ID of the hierarchical level.
        prediction (np.ndarray): The predicted state vector.
        confidence (float): Confidence score in [0, 1].
        uncertainty (float): Uncertainty estimate (e.g., variance/entropy).
    """
    level: int
    prediction: np.ndarray
    confidence: float
    uncertainty: float


@dataclass
class ConsensusResult:
    """
    Result of a consensus aggregation.
    
    Attributes:
        prediction (np.ndarray): The aggregated state prediction.
        agreement_score (float): Fraction of levels in agreement.
        uncertainty (float): Aggregated uncertainty score.
        outlier_count (int): Number of levels identified as outliers.
        participating_levels (List[int]): IDs of non-outlier levels.
    """
    prediction: np.ndarray
    agreement_score: float
    uncertainty: float
    outlier_count: int
    participating_levels: List[int]


class HierarchicalConsensus(Module):
    """
    Hierarchical Consensus Engine for robust multi-level aggregation.

    This engine identifies and rejects outliers using robust statistics (MAD)
    and produces a weighted average of valid predictions.

    Attributes:
        outlier_threshold (float): Z-score threshold for outlier detection.
        min_agreement (float): Minimum required agreement for stability.
        use_temporal_weighting (bool): Whether to apply temporal smoothing.
    """

    def __init__(
        self,
        outlier_threshold: float = 3.0,
        min_agreement: float = 0.6,
        use_temporal_weighting: bool = False
    ):
        """
        Initialize HierarchicalConsensus.

        Args:
            outlier_threshold: Threshold for Z-score outlier rejection.
            min_agreement: Minimum agreement score for stability checks.
            use_temporal_weighting: Enable temporal smoothing (not fully implemented).
        """
        self.outlier_threshold = outlier_threshold
        self.min_agreement = min_agreement
        self.use_temporal_weighting = use_temporal_weighting
        self.history: List[ConsensusResult] = []

    def aggregate(
        self,
        predictions: List[LevelPrediction],
        method: Literal['mean', 'median', 'weighted_vote'] = 'weighted_vote'
    ) -> ConsensusResult:
        """
        Aggregate multiple level predictions into a single consensus.

        Args:
            predictions: List of LevelPrediction objects.
            method: Aggregation strategy ('mean', 'median', 'weighted_vote').

        Returns:
            ConsensusResult containing the aggregated estimate.

        Raises:
            ValueError: If the predictions list is empty or dimensions are incompatible.
        """
        if not predictions:
            raise ValueError("No predictions provided to consensus engine")

        # Convert predictions to numpy arrays and check dimensions
        pred_arrays = []
        for p in predictions:
            pred_array = np.asarray(p.prediction)
            if pred_array.ndim == 0:
                pred_array = pred_array.reshape(1)
            pred_arrays.append(pred_array)
        
        # Check if all predictions have the same dimension
        dimensions = [arr.shape[0] for arr in pred_arrays]
        unique_dims = set(dimensions)
        
        if len(unique_dims) > 1:
            raise ValueError(
                f"Cannot aggregate predictions with incompatible dimensions: {unique_dims}. "
                f"Use ConsensusHierarchyManager with auto_projection=True to handle mismatched dimensions."
            )
        
        # Now convert to 2D array safely
        preds = np.array(pred_arrays)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]

        confs = np.array([p.confidence for p in predictions])
        uncs = np.array([p.uncertainty for p in predictions])

        # 1. Outlier Detection (Z-score based on distances from median)
        median_pred = np.median(preds, axis=0)
        dists = np.linalg.norm(preds - median_pred, axis=1)

        # Robust Std estimation (MAD: Median Absolute Deviation)
        mad = np.median(np.abs(dists - np.median(dists)))
        std_est = 1.4826 * mad if mad > 1e-6 else 1.0

        z_scores = dists / std_est
        outlier_mask = z_scores > self.outlier_threshold

        # Filter out outliers
        valid_preds = preds[~outlier_mask]
        valid_confs = confs[~outlier_mask]
        valid_uncs = uncs[~outlier_mask]

        participating_levels = [
            p.level for i, p in enumerate(predictions) if not outlier_mask[i]
        ]

        # 2. Aggregation
        if len(valid_preds) == 0:
            # Fallback to median if everyone is an outlier
            final_pred = median_pred
            final_unc = float(np.mean(uncs))
            agreement_score = 0.0
        else:
            if method == 'mean':
                final_pred = np.mean(valid_preds, axis=0)
            elif method == 'median':
                final_pred = np.median(valid_preds, axis=0)
            elif method == 'weighted_vote':
                # Weight by confidence
                weights = valid_confs / (np.sum(valid_confs) + 1e-9)
                final_pred = np.average(valid_preds, axis=0, weights=weights)
            else:
                final_pred = np.mean(valid_preds, axis=0)

            final_unc = float(np.mean(valid_uncs))
            agreement_score = len(valid_preds) / len(predictions)

        # Handle squeeze/flatten for scalars if input was 0D
        if final_pred.size == 1 and predictions[0].prediction.ndim == 0:
            final_pred = np.array(final_pred).reshape(())

        result = ConsensusResult(
            prediction=final_pred,
            agreement_score=agreement_score,
            uncertainty=final_unc,
            outlier_count=int(np.sum(outlier_mask)),
            participating_levels=participating_levels
        )
        return result

    # --- Module Interface ---

    def forward(self, x: Any) -> np.ndarray:
        """
        Forward pass for consensus engine.
        
        Args:
            x: List[LevelPrediction] to aggregate.
            
        Returns:
            The aggregated prediction array.
        """
        if not isinstance(x, list):
            raise TypeError("ConsensusEngine.forward expects a List[LevelPrediction]")
        
        result = self.aggregate(x)
        return result.prediction

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass not supported."""
        raise NotImplementedError("ConsensusEngine does not support backpropagation.")

    def update(self, lr: float) -> None:
        """Consensus parameters are typically fixed or meta-learned, not via SGD."""
        pass

    def reset_state(self) -> None:
        """Clear consensus history."""
        self.history = []
