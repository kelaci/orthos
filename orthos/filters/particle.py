"""
Particle Filter (Sequential Monte Carlo) implementation.

This module provides a robust implementation of the Particle Filter algorithm
for non-linear, non-Gaussian state estimation. It follows the ORTHOS
coding standards and inherits from the base Module class.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any, Union

from orthos.core.base import Module
from orthos.core.gpu_utils import get_array_module


class ParticleFilter(Module):
    """
    Particle Filter (Sequential Monte Carlo) for non-linear state estimation.

    The Particle Filter approximates the posterior density of the state
    using a set of weighted particles. It is capable of representing
    arbitrary, multi-modal distributions.

    Attributes:
        n_particles (int): Number of particles.
        state_dim (int): Dimension of the state vector.
        dynamics_fn (Callable): State transition function f(x, u, noise).
        observation_fn (Callable): Observation likelihood function p(z|x).
        particles (np.ndarray): Particle states (n_particles, state_dim).
        weights (np.ndarray): Particle weights (n_particles,).
    """

    def __init__(
        self,
        n_particles: int,
        state_dim: int,
        dynamics_fn: Callable[[np.ndarray, Optional[np.ndarray], np.ndarray], np.ndarray],
        observation_fn: Callable[[np.ndarray, np.ndarray], float],
        seed: Optional[int] = None
    ):
        """
        Initialize Particle Filter.

        Args:
            n_particles: Number of particles to use.
            state_dim: Dimension of the state vector.
            dynamics_fn: Function returning new particles given (particles, u, noise).
            observation_fn: Function returning likelihood scalar given (particle, z).
            seed: Random seed for reproducibility.
        """
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.dynamics_fn = dynamics_fn
        self.observation_fn = observation_fn
        
        if seed is not None:
            np.random.seed(seed)
            
        self.particles: np.ndarray = np.zeros((n_particles, state_dim))
        self.weights: np.ndarray = np.ones(n_particles) / n_particles
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset particles to initial distribution.
        
        Initializes particles from a standard normal distribution N(0, 1).
        """
        self.particles = np.random.randn(self.n_particles, self.state_dim)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(
        self,
        process_noise_std: float = 0.1,
        u: Optional[np.ndarray] = None
    ) -> None:
        """
        Propagate particles through the dynamics model.

        Args:
            process_noise_std: Standard deviation of process noise.
            u: Control input vector (optional).
        """
        # Vectorized noise generation
        noise = np.random.randn(self.n_particles, self.state_dim) * process_noise_std
        
        # Apply dynamics
        self.particles = self.dynamics_fn(self.particles, u, noise)

    def update(self, z: np.ndarray) -> None:
        """
        Update particle weights based on observation likelihood.

        Args:
            z: Observation vector.
        """
        # Vectorized or loop-based likelihood computation
        # Assuming observation_fn might not be vectorized, we loop for safety
        # unless optimized later.
        for i in range(self.n_particles):
            self.weights[i] = self.observation_fn(self.particles[i], z)
        
        # Normalize weights
        w_sum = np.sum(self.weights)
        if w_sum < 1e-10:
            # Rescue: reset weights to uniform if degeneracy occurs (particle collapse)
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights /= w_sum
            
        # Resample if effective sample size is too low
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.n_particles / 2.0:
            self._resample()

    def _resample(self) -> None:
        """
        Perform systematic resampling of particles.
        
        Reduces particle degeneracy by duplicating high-weight particles
        and removing low-weight ones.
        """
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Avoid round-off error
        
        # Systematic uniform random numbers
        step = 1.0 / self.n_particles
        start = np.random.rand() * step
        positions = np.arange(self.n_particles) * step + start
        
        indexes = np.searchsorted(cumulative_sum, positions)
        
        # Resample particles
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0 / self.n_particles)

    def get_mean(self) -> np.ndarray:
        """
        Calculate weighted mean of the state.

        Returns:
            Weighted mean state vector.
        """
        return np.average(self.particles, weights=self.weights, axis=0)

    def get_uncertainty(self) -> float:
        """
        Calculate approximate uncertainty (trace of covariance).

        Returns:
            Trace of the weighted covariance matrix.
        """
        cov = np.cov(
            self.particles, 
            rowvar=False, 
            aweights=self.weights
        )
        return float(np.trace(cov))

    # --- Module Interface Implementation ---

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform one full filter step: Predict -> Update.
        
        In the context of a Module, 'x' is treated as the observation 'z'.
        
        Args:
            x: Observation vector z.
            
        Returns:
            Estimated state vector (mean).
        """
        # Assume x contains observation. Control u is None for basic forward.
        self.predict()
        self.update(x)
        return self.get_mean()

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass is not applicable for standard Particle Filters.
        
        Raises:
            NotImplementedError
        """
        raise NotImplementedError("ParticleFilter does not support backpropagation.")

    # 'update' is already defined but conflicts with Module.update(lr).
    # In Module, update(lr) is for parameter updates (weights).
    # In Filter, update(z) is for state updates.
    # We resolve this by renaming the Module update to 'update_parameters'.
    # However, Python doesn't support overloading well.
    # We will alias the Filter update to 'update_state' and implement 'update' roughly.
    
    # Actually, proper design:
    # Rename filter update to 'observe' or 'correction' to avoid conflict?
    # Or check type of arg?
    # For now, let's strictly follow inherited signature for 'update'
    # and use a different name for the measurement update.
    
    # REVISION:
    # To avoid breaking the existing API usage in tests (if any), we keep `update(z)`.
    # But Module expects `update(lr)`.
    # Since Python is dynamic, we can inspect argument or use optional args.
    # But best practice: The concept of "Parameter Update" (Plasticity) maps poorly 
    # to a fixed Filter.
    # We will implement Module.update as a no-op matching signature.
    
    def update_parameters(self, lr: float) -> None:
        """
        Update module parameters.
        
        Particle Filter has no learnable parameters in this implementation.
        
        Args:
            lr: Learning rate.
        """
        pass
        
    def update(self, z: Union[np.ndarray, float]) -> None:  # type: ignore[override]
        """
        Update particle weights based on observation likelihood (Measurement Update).
        
        Args:
            z: Observation vector.
        """
        # Implementation duplicated from above for safety, or we rename methods.
        # Ideally, we rename this to "correct" or "assimilate" but standard Kalman terminology is "update".
        # We will suppress the linter warning for signature mismatch as this is intentional for a Filter.
        
        if isinstance(z, float):
             z = np.array([z])
             
        for i in range(self.n_particles):
            self.weights[i] = self.observation_fn(self.particles[i], z)
        
        w_sum = np.sum(self.weights)
        if w_sum < 1e-10:
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights /= w_sum
            
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.n_particles / 2.0:
            self._resample()


