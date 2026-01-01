"""
FilteredHierarchicalLevel implementation.

Extends HierarchicalLevel with Bayesian state estimation filters (Kalman, Particle).
Provides robust state estimation and uncertainty quantification for hierarchical levels.
"""

import numpy as np
from typing import Tuple, Optional, Literal, Dict, Any, Union, List

from orthos.hierarchy.level import HierarchicalLevel
from orthos.filters.kalman import KalmanFilter, ExtendedKalmanFilter
from orthos.filters.particle import ParticleFilter


class FilteredHierarchicalLevel(HierarchicalLevel):
    """
    Hierarchical level with integrated Bayesian filtering.

    Combines the neural processing of a HierarchicalLevel with probabilistic 
    state estimation to handle noisy sensory inputs and quantify uncertainty.

    Attributes:
        filter_type (str): Type of filter ('kalman', 'ekf', 'particle', 'none').
        filter (Optional[Union[KalmanFilter, ParticleFilter]]): The filter instance.
        uncertainty_history (List[float]): History of uncertainty estimates.
        _top_down_prior (Optional[np.ndarray]): Prior from higher-level consensus.
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
        use_diagonal_covariance: bool = None,
        min_obs_noise: float = 1e-6,
        use_joseph_form: bool = False,
        top_down_weight: float = 0.5,
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
            use_diagonal_covariance: Use diagonal covariance for high dimensions.
            min_obs_noise: Minimum threshold for observation noise.
            use_joseph_form: Use Joseph form for numerical stability.
            top_down_weight: Weight for top-down prior fusion (0-1).
            **kwargs: Additional parameters for the base level.
        """
        super().__init__(level_id, input_size, output_size, temporal_resolution)

        self.filter_type = filter_type
        self.filter: Optional[Union[KalmanFilter, ParticleFilter]] = None
        self.top_down_weight = top_down_weight
        self._top_down_prior: Optional[np.ndarray] = None

        # Initialize Filter based on type
        if filter_type == 'kalman':
            self.filter = KalmanFilter(
                state_dim=output_size,
                obs_dim=output_size,
                process_noise=process_noise,
                obs_noise=obs_noise,
                adaptive=adaptive_noise,
                use_diagonal_covariance=use_diagonal_covariance,
                min_obs_noise=min_obs_noise,
                use_joseph_form=use_joseph_form
            )
        elif filter_type == 'ekf':
            self.filter = ExtendedKalmanFilter(
                state_dim=output_size,
                obs_dim=output_size,
                dynamics_fn=lambda x, u: x,  # Default to Random Walk
                observation_fn=lambda x: x,
                process_noise=process_noise,
                obs_noise=obs_noise,
                adaptive=adaptive_noise,
                use_diagonal_covariance=use_diagonal_covariance,
                min_obs_noise=min_obs_noise,
                use_joseph_form=use_joseph_form
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

    def set_top_down_prior(self, prior: np.ndarray) -> None:
        """
        Set a top-down prior from higher-level consensus.

        Args:
            prior: Prior vector from higher levels.
        """
        self._top_down_prior = prior

    def forward_filtered(
        self,
        input_data: np.ndarray,
        top_down_prior: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Perform a forward pass through neural layers followed by filtering.

        Implements Bayesian fusion for robust state estimation:
        1. Bottom-Up: Update filter with neural output (measurement)
        2. Top-Down: Bayesian fusion with prior from higher levels (single update)

        This approach is more mathematically elegant and efficient than calling
        filter.update() twice. It uses the parallel combination rule for Bayesian
        fusion, treating the prior as a second information source with higher
        uncertainty (lower trust).

        Args:
            input_data: Input features for this step.
            top_down_prior: Optional prior from higher levels.

        Returns:
            Tuple of (Filtered Representation, Current Uncertainty).

        Notes:
            The Bayesian fusion formula for diagonal covariance:
                x_fused = (x_bu / R_bu + x_td / R_td) / (1/R_bu + 1/R_td)
                unc_fused = 1 / (1/R_bu + 1/R_td)
            
            Where top-down is treated as 2x less certain than bottom-up by default.
        """
        # Use provided prior or stored prior
        if top_down_prior is not None:
            prior = top_down_prior
        elif self._top_down_prior is not None:
            prior = self._top_down_prior
        else:
            prior = None

        # 1. Base neural processing
        self.time_step += 1
        raw_output = super().process_time_step(input_data, self.time_step)

        # Ensure prediction is 1D for the filtering stage if it's a single sample
        if raw_output.ndim > 1:
            raw_output = raw_output.flatten()

        if self.filter is None:
            return raw_output, 0.0

        # 2. Filtering cycle with Bayesian fusion
        if isinstance(self.filter, KalmanFilter):
            # Identity transition assumed for random walk
            self.filter.predict(F=np.eye(self.output_size))
            
            # First update: Bottom-Up (neural output as measurement)
            state_est, P = self.filter.update(raw_output)
            
            # Calculate uncertainty as trace or sum of diagonal
            unc = float(np.trace(P)) if P.ndim == 2 else float(np.sum(P))
            
            # Bayesian fusion for top-down prior
            # This replaces the second filter.update() with a mathematically elegant fusion
            if prior is not None:
                # Ensure prior matches dimension
                if prior.shape[0] != self.output_size:
                    # Simple projection: repeat or truncate
                    if prior.shape[0] < self.output_size:
                        prior_repeated = np.repeat(prior, 
                            (self.output_size + prior.shape[0] - 1) // prior.shape[0])[:self.output_size]
                        prior = prior_repeated
                    else:
                        prior = prior[:self.output_size]
                
                # Bayesian fusion using parallel combination rule
                # This is mathematically equivalent to treating prior as a measurement
                # but more efficient and clearer in intent
                
                # R_bu: Uncertainty from bottom-up (neural network measurement)
                # R_td: Uncertainty from top-down (prior - less trusted by default)
                r_bu = unc
                r_td = unc * (1.0 / (self.top_down_weight + 1e-9))  # Higher weight = lower noise
                
                # Numerical safety
                r_bu = max(r_bu, 1e-6)
                r_td = max(r_td, 1e-6)
                
                # Inverse variances (weights)
                inv_r_bu = 1.0 / r_bu
                inv_r_td = 1.0 / r_td
                inv_sum = inv_r_bu + inv_r_td
                
                # Bayesian fusion: weighted average by inverse variance
                # x_fused = (x_td / R_td + x_bu / R_bu) / (1/R_td + 1/R_bu)
                state_est = (state_est * inv_r_bu + prior * inv_r_td) / inv_sum
                
                # Fused uncertainty using parallel combination rule
                # 1/unc_fused = 1/unc_bu + 1/unc_td
                unc = 1.0 / inv_sum
            
            prediction = state_est
            uncertainty = unc
            
        elif isinstance(self.filter, ParticleFilter):
            self.filter.predict(process_noise_std=0.01)
            self.filter.update(raw_output)
            
            # For particle filter, incorporate prior by adjusting weights
            if prior is not None:
                # Incorporate prior into particle weights
                particles = self.filter.particles
                prior_likelihood = np.exp(-0.5 * np.sum((prior - particles)**2, axis=1))
                self.filter.weights *= prior_likelihood
                self.filter.weights /= np.sum(self.filter.weights) + 1e-10
            
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
        self._top_down_prior = None
