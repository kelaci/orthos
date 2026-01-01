"""
FilteredHierarchicalLevel implementation.

Extends HierarchicalLevel with Bayesian state estimation filters (Kalman, Particle).
Provides robust state estimation and uncertainty quantification for hierarchical levels.
"""

import numpy as np
from typing import Tuple, Optional, Literal, Dict, Any, Union

from gaia.hierarchy.level import HierarchicalLevel
from gaia.filters.kalman import KalmanFilter, ExtendedKalmanFilter
from gaia.filters.particle import ParticleFilter


class FilteredHierarchicalLevel(HierarchicalLevel):
    """
    Hierarchical level with integrated Bayesian filtering.

    Combines the neural processing of a HierarchicalLevel with probabilistic 
    state estimation to handle noisy sensory inputs and quantify uncertainty.

    Attributes:
        filter_type (str): Type of filter ('kalman', 'ekf', 'particle', 'none').
        filter (Optional[Union[KalmanFilter, ParticleFilter]]): The filter instance.
        uncertainty_history (List[float]): History of uncertainty estimates.
    """

    def __init__(
        self,
        level_id: int,
        input_size: int,
        output_size: int,
        temporal_resolution: int = 1,
        filter_type: Literal['kalman', 'ekf', 'particle', 'none'] = 'kalman',
        process_noise: float = 0.01,
        obs_noise: float = 0.1,
        adaptive_noise: bool = False,
        **kwargs: Any
    ):
        """
        Initialize FilteredHierarchicalLevel.

        Args:
            level_id: Unique level identifier.
            input_size: Size of input features.
            output_size: Size of output features/representation.
            temporal_resolution: Relative time scale of this level.
            filter_type: Estimation algorithm to use.
            process_noise: Estimated variance of the state dynamics.
            obs_noise: Estimated variance of the observations.
            adaptive_noise: Enable online noise adaptation if supported.
            **kwargs: Additional parameters for the base level.
        """
        super().__init__(level_id, input_size, output_size, temporal_resolution)

        self.filter_type = filter_type
        self.filter: Optional[Union[KalmanFilter, ParticleFilter]] = None

        # Initialize Filter based on type
        if filter_type == 'kalman':
            self.filter = KalmanFilter(
                state_dim=output_size,
                obs_dim=output_size,
                process_noise=process_noise,
                obs_noise=obs_noise,
                adaptive=adaptive_noise
            )
        elif filter_type == 'ekf':
            self.filter = ExtendedKalmanFilter(
                state_dim=output_size,
                obs_dim=output_size,
                dynamics_fn=lambda x, u: x,  # Default to Random Walk
                observation_fn=lambda x: x,
                process_noise=process_noise,
                obs_noise=obs_noise
            )
        elif filter_type == 'particle':
            self.filter = ParticleFilter(
                n_particles=500,
                state_dim=output_size,
                dynamics_fn=lambda x, u, n: x + n,
                observation_fn=lambda x, z: np.exp(
                    -0.5 * np.sum((z - x)**2) / (obs_noise + 1e-9)
                )
            )

        self.uncertainty_history: List[float] = []

    def forward_filtered(
        self,
        input_data: np.ndarray,
        top_down_prior: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Perform a forward pass through neural layers followed by filtering.

        Args:
            input_data: Input features for this step.
            top_down_prior: Optional prior from higher levels (unused currently).

        Returns:
            Tuple of (Filtered Representation, Current Uncertainty).
        """
        # 1. Base neural processing
        self.time_step += 1
        raw_output = super().process_time_step(input_data, self.time_step)

        # Ensure prediction is 1D for the filtering stage if it's a single sample
        if raw_output.ndim > 1:
            raw_output = raw_output.flatten()

        if self.filter is None:
            return raw_output, 0.0

        # 2. Filtering cycle
        # We perform prediction and update cycle
        if isinstance(self.filter, KalmanFilter):
            # Identity transition assumed for random walk
            self.filter.predict(F=np.eye(self.output_size))
            prediction, P = self.filter.update(raw_output)
            uncertainty = float(np.trace(P))
        elif isinstance(self.filter, ParticleFilter):
            self.filter.predict(process_noise_std=0.01)
            self.filter.update(raw_output)
            prediction = self.filter.get_mean()
            uncertainty = self.filter.get_uncertainty()
        else:
            prediction = raw_output
            uncertainty = 0.0

        self.uncertainty_history.append(uncertainty)
        return prediction, uncertainty

    def get_confidence(self) -> float:
        """
        Calculate a confidence score based on current estimation uncertainty.

        Returns:
            Confidence score in range [0, 1].
        """
        if not self.uncertainty_history:
            return 0.5
        curr_unc = self.uncertainty_history[-1]
        # Invert uncertainty: higher uncertainty -> lower confidence
        return 1.0 / (1.0 + curr_unc)

    def reset_state(self) -> None:
        """Reset neural layers, filter state, and history."""
        super().reset_state()
        if self.filter:
            self.filter.reset_state()
        self.uncertainty_history = []
