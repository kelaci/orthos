"""
FilteredHierarchicalLevel implementation.

Extends HierarchicalLevel with Bayesian state estimation filters (Kalman, Particle).
Provides robust state estimation and uncertainty quantification for hierarchical levels.
"""

import numpy as np
from typing import Tuple, Optional, Literal, Dict, Any, Union, List

from orthos.hierarchy.level import HierarchicalLevel
from orthos.filters.kalman import KalmanFilter, ExtendedKalmanFilter, SquareRootKalmanFilter, BlockDiagonalKalmanFilter
from orthos.filters.particle import ParticleFilter
from orthos.meta_learning.hybrid_manager import HybridMetaManager


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
        filter_type: Literal['kalman', 'sr_kalman', 'block_diagonal', 'ekf', 'particle', 'none'] = 'kalman',
        process_noise: float = 0.01,
        obs_noise: float = 0.1,
        adaptive_noise: bool = False,
        use_diagonal_covariance: Optional[bool] = None,
        min_obs_noise: float = 1e-6,
        use_joseph_form: bool = False,
        top_down_weight: float = 0.5,
        dynamic_modulation: bool = True,
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

        # Initialize Filter using dispatch table (Standard 2.6.1)
        filter_params = dict(
            state_dim=output_size,
            obs_dim=output_size,
            process_noise=process_noise,
            obs_noise=obs_noise,
            adaptive=adaptive_noise,
            use_diagonal_covariance=use_diagonal_covariance,
            min_obs_noise=min_obs_noise,
            use_joseph_form=use_joseph_form
        )
        
        self.filter = self._init_filter(filter_type, output_size, obs_noise, filter_params)
        
        self.dynamic_modulation = dynamic_modulation
        self.uncertainty_history: List[float] = []
        self.state_history: List[np.ndarray] = []
        self._current_concept: Optional[str] = None

        # Hybrid Meta-Learning Manager
        base_meta_params = {
            'adaptation_rate': kwargs.get('adaptation_rate', 0.01),
            'exploration_noise': kwargs.get('exploration_noise', 0.1),
            'consensus_threshold': kwargs.get('consensus_threshold', 3.0),
            'process_noise_scale': process_noise,
            'obs_noise_scale': obs_noise
        }
        self.meta_manager = HybridMetaManager(
            base_parameters=base_meta_params,
            use_bandit=kwargs.get('use_meta_bandit', True),
            use_nes=kwargs.get('use_meta_nes', False)
        )

    def _init_filter(
        self, 
        filter_type: str, 
        output_size: int, 
        obs_noise: float, 
        filter_params: Dict[str, Any]
    ) -> Optional[Union[KalmanFilter, ParticleFilter]]:
        """Initialize appropriate filter using a dispatch map."""
        
        def init_ekf():
            return ExtendedKalmanFilter(
                dynamics_fn=lambda x, u: x,
                observation_fn=lambda x: x,
                **filter_params
            )
            
        def init_particle():
            return ParticleFilter(
                n_particles=500,
                state_dim=output_size,
                dynamics_fn=lambda x, u, n: x + n,
                observation_fn=lambda x, z: np.exp(
                    -0.5 * np.sum((z - x)**2) / (obs_noise + 1e-9)
                )
            )

        dispatch = {
            'kalman': lambda: KalmanFilter(**filter_params),
            'sr_kalman': lambda: SquareRootKalmanFilter(**filter_params),
            'block_diagonal': lambda: BlockDiagonalKalmanFilter(**filter_params),
            'ekf': init_ekf,
            'particle': init_particle,
            'none': lambda: None
        }
        
        return dispatch.get(filter_type, lambda: None)()

    def process_time_step(
        self, 
        input_data: np.ndarray, 
        t: int
    ) -> np.ndarray:
        """
        Polymorphic interface for hierarchical processing.
        
        Args:
            input_data: Current input features.
            t: Current time step.
            
        Returns:
            Processed and filtered representation.
        """
        # forward_filtered handles the neural pass and the filtering
        prediction, _ = self.forward_filtered(input_data)
        return prediction

    def set_top_down_prior(self, prior: np.ndarray) -> None:
        """
        Set a top-down prior from higher-level consensus.

        Args:
            prior: Prior vector from higher levels.
        """
        self._top_down_prior = prior

    def set_concept_state(self, concept: str) -> None:
        """
        Set the conceptual state (overrides local detection).
        
        Args:
            concept: "stable", "transition", or "storm".
        """
        self._current_concept = concept

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

        # 1.5. Hybrid Meta-Learning Hook: Dynamic Adaptation
        if self.meta_manager is not None:
            # A. Compute Context Features
            # 1. Prediction Error (Reconstruction)
            error = 0.0
            if self.state_history:
                last_pred = self.state_history[-1]
                # Simple L2 error if dimensions match (otherwise 0.0)
                if last_pred.shape == raw_output.shape:
                    error = float(np.linalg.norm(raw_output - last_pred))
            
            # 2. Uncertainty
            unc = self.uncertainty_history[-1] if self.uncertainty_history else 0.0
            
            # 3. Sparsity (Ratio of near-zero activations)
            sparsity = float(np.mean(np.abs(raw_output) < 1e-3))
            
            # 4. Drift (Rate of change in representation)
            drift = 0.0
            if len(self.state_history) >= 2:
                drift = float(np.linalg.norm(self.state_history[-1] - self.state_history[-2]))
                
            context = np.array([error, unc, sparsity, drift])
            
            # B. Get Modulated Parameters
            modulated = self.meta_manager.step(context)
            
            # C. Apply Modulation
            # 1. Apply to neural layers (Plasticity)
            for layer in self.processing_layers:
                if hasattr(layer, 'set_plasticity_params'):
                    layer.set_plasticity_params({
                        'learning_rate': modulated.get('adaptation_rate', 0.01)
                    })
            
            # 2. Apply to Filter (Noise levels)
            if self.filter is not None:
                # Update R and Q based on meta-scales (Relative to base parameters)
                # This prevents cumulative explosion/collapse
                base_r = self.meta_manager.base_params.get('obs_noise_scale', 0.1)
                base_q = self.meta_manager.base_params.get('process_noise_scale', 0.01)
                
                # Modulated multiplier
                scale_r = modulated.get('obs_noise_scale', base_r) / (base_r + 1e-9)
                scale_q = modulated.get('process_noise_scale', base_q) / (base_q + 1e-9)
                
                # Set values on filter (maintaining correct array/diagonal structure)
                target_r = base_r * scale_r
                target_q = base_q * scale_q
                
                if self.filter.use_diagonal_covariance:
                    self.filter.R = np.ones(self.filter.obs_dim) * target_r
                    self.filter.Q = np.ones(self.filter.state_dim) * target_q
                else:
                    self.filter.R = np.eye(self.filter.obs_dim) * target_r
                    self.filter.Q = np.eye(self.filter.state_dim) * target_q
                
            # D. Feedback (Reward)
            # Reward is negative loss and negative uncertainty
            reward = -(error + 0.5 * unc + 0.1 * drift)
            self.meta_manager.update_feedback(reward)

        # 2. Filtering cycle with Bayesian fusion
        if self.filter is not None and not isinstance(self.filter, ParticleFilter):
            # Dynamic Modulation based on concept state
            if self.dynamic_modulation:
                concept = self._detect_concept_state()
                if concept == "storm":
                    # Increase R (trust sensors less during "storm"/chaos)
                    self.filter.R *= 1.5
                elif concept == "stable":
                    # Decrease R (trust sensors more during stability)
                    self.filter.R *= 0.95
                
                # Reset externally set concept to allow fresh detection if not updated
                self._current_concept = None
                
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
        self.state_history.append(prediction)
        return prediction, uncertainty

    def _detect_concept_state(self) -> str:
        """
        Analyze current state history to detect conceptual regime.
        
        Returns:
            "stable", "transition", or "storm".
        """
        # Return externally set concept if available
        if self._current_concept is not None:
            return self._current_concept

        if len(self.state_history) < 5:
            return "stable"
        
        # Recent volatility
        recent_states = np.array(self.state_history[-5:])
        diffs = np.diff(recent_states, axis=0)
        volatility = np.mean(np.linalg.norm(diffs, axis=1))
        
        # Recent uncertainty trend
        if len(self.uncertainty_history) >= 2:
            unc_trend = self.uncertainty_history[-1] - self.uncertainty_history[-2]
        else:
            unc_trend = 0
            
        if volatility > 0.5 or unc_trend > 0.1:
            return "storm"
        elif volatility > 0.2:
            return "transition"
        else:
            return "stable"

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
