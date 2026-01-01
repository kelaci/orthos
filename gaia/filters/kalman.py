"""
Kalman Filter implementations for GAIA.

This module implements:
- Standard Kalman Filter (KF)
- Extended Kalman Filter (EKF)

All implementations follow the GAIA coding standard and inherit from Module.
"""

import numpy as np
from typing import Optional, Tuple, Callable, Union, List

from gaia.core.base import Module


class KalmanFilter(Module):
    """
    Standard Linear Kalman Filter.

    Implements optimal recursive Bayesian estimation for linear Gaussian systems.

    Equations:
        x' = Fx + Bu          (State Prediction)
        P' = FPF' + Q         (Covariance Prediction)
        y = z - Hx'           (Innovation)
        S = HP'H' + R         (Innovation Covariance)
        K = P'H' inv(S)       (Kalman Gain)
        x = x' + Ky           (State Update)
        P = (I - KH)P'        (Covariance Update)

    Attributes:
        state_dim (int): Dimensionality of state.
        obs_dim (int): Dimensionality of observation.
        x (np.ndarray): State vector.
        P (np.ndarray): State covariance matrix.
        Q (np.ndarray): Process noise covariance.
        R (np.ndarray): Observation noise covariance.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        process_noise: float = 0.01,
        obs_noise: float = 0.1,
        adaptive: bool = False
    ):
        """
        Initialize Kalman Filter.

        Args:
            state_dim: Size of state vector.
            obs_dim: Size of observation vector.
            process_noise: Initial variance for process noise Q (identity scale).
            obs_noise: Initial variance for observation noise R (identity scale).
            adaptive: Whether to adapt Q based on innovation magnitude (simple heuristic).
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.adaptive = adaptive

        # State (Mean) and Covariance
        self.x: np.ndarray = np.zeros(state_dim)
        self.P: np.ndarray = np.eye(state_dim) * 1.0  # Initial uncertainty

        # Noise Covariances
        self.Q: np.ndarray = np.eye(state_dim) * process_noise
        self.R: np.ndarray = np.eye(obs_dim) * obs_noise

        # Identity matrix cache
        self.I: np.ndarray = np.eye(state_dim)

    def reset_state(self) -> None:
        """Reset state and covariance to initial conditions."""
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1.0

    def predict(
        self,
        F: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None
    ) -> None:
        """
        Perform prediction step (Time Update).

        Args:
            F: State transition matrix (default: Identity).
            B: Control input matrix.
            u: Control input vector.
        """
        if F is None:
            F = self.I

        # x = Fx
        self.x = F @ self.x

        if B is not None and u is not None:
            self.x += B @ u

        # P = FPF' + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(
        self,
        z: np.ndarray,
        H: Optional[np.ndarray] = None,
        R_override: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform update step (Measurement Update).

        Args:
            z: Observation vector.
            H: Observation matrix (default: Identity mapping state to obs).
            R_override: Optional override for observation noise covariance.

        Returns:
            Tuple of (Updated State x, Updated Covariance P).
        """
        if H is None:
            # Assume 1-to-1 observation of first obs_dim states
            H = np.eye(self.obs_dim, self.state_dim)

        R = R_override if R_override is not None else self.R

        # y = z - Hx (Innovation)
        y = z - H @ self.x

        # S = HP'H' + R
        S = H @ self.P @ H.T + R

        # K = P'H' inv(S) (Kalman Gain)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix using Pseudo-inverse
            K = self.P @ H.T @ np.linalg.pinv(S)

        # x = x' + Ky
        self.x = self.x + K @ y

        # P = (I - KH)P'
        self.P = (self.I - K @ H) @ self.P

        if self.adaptive:
            self._adapt_noise(y, S)

        return self.x, self.P

    def _adapt_noise(self, y: np.ndarray, S: np.ndarray) -> None:
        """
        Adapt process noise Q based on innovation.
        
        Args:
            y: Innovation vector.
            S: Innovation covariance.
        """
        innovation_sq = float(np.dot(y, y))
        if innovation_sq > 5.0 * np.trace(S):
            self.Q *= 1.1
        else:
            self.Q *= 0.99

    # --- Module Interface Implementation ---

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Standard Module forward pass.
        
        Interpret 'x' as observation 'z'.
        Performs Predict -> Update cycle assuming Identity drift and Obs.
        
        Args:
            x: Observation vector.
            
        Returns:
            Estimated state vector.
        """
        self.predict()
        self.update(x)
        return self.x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass not implemented for Kalman Filter.
        
        Raises:
            NotImplementedError
        """
        raise NotImplementedError("KalmanFilter does not support backpropagation.")
    
    def update_parameters(self, lr: float) -> None:
        """
        No learnable parameters to update via SGD.
        
        Args:
            lr: Learning rate (ignored).
        """
        pass


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter (EKF) for non-linear systems.

    Uses Jacobian linearization for state and covariance propagation.
    
    Attributes:
        dynamics_fn (Callable): Non-linear state transition f(x, u).
        observation_fn (Callable): Non-linear observation h(x).
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        dynamics_fn: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        observation_fn: Callable[[np.ndarray], np.ndarray],
        process_noise: float = 0.01,
        obs_noise: float = 0.1,
        adaptive: bool = False
    ):
        """
        Initialize EKF.
        
        Args:
            state_dim: State dimension.
            obs_dim: Observation dimension.
            dynamics_fn: Function f(x, u) -> x_next.
            observation_fn: Function h(x) -> z_pred.
            process_noise: Initial Q scale.
            obs_noise: Initial R scale.
            adaptive: Enable adaptive Q.
        """
        super().__init__(state_dim, obs_dim, process_noise, obs_noise, adaptive)
        self.dynamics_fn = dynamics_fn
        self.observation_fn = observation_fn

    def predict(
        self,
        F: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None
    ) -> None:
        """
        EKF Prediction.

        Args:
            F: Jacobian of dynamics_fn w.r.t state at current x.
               REQUIRED for EKF in this implementation if not computed auto-diff.
            B: Ignored for non-linear f(x, u) unless used in Jacobian.
            u: Control input.
        """
        if F is None:
            # In a full diff-programming framework, we would auto-compute Jacobian.
            # Here we default to Identity which reduces EKF to KF for this step if user fails to provide F.
            F = self.I

        # 1. Update state using non-linear dynamics: x = f(x, u)
        self.x = self.dynamics_fn(self.x, u)

        # 2. Propagate covariance using Jacobian F: P = FPF' + Q
        self.P = F @ self.P @ F.T + self.Q
