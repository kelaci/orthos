"""
Kalman Filter implementations for ORTHOS.

This module implements:
- Standard Kalman Filter (KF)
- Extended Kalman Filter (EKF)

All implementations follow the ORTHOS coding standard and inherit from Module.
"""

import numpy as np
from typing import Optional, Tuple, Callable, Union, List

from orthos.core.base import Module


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
        adaptive: bool = False,
        use_diagonal_covariance: bool = None,
        min_obs_noise: float = 1e-6,
        use_joseph_form: bool = False
    ):
        """
        Initialize Kalman Filter.

        Args:
            state_dim: Size of state vector.
            obs_dim: Size of observation vector.
            process_noise: Initial variance for process noise Q (identity scale).
            obs_noise: Initial variance for observation noise R (identity scale).
            adaptive: Whether to adapt Q based on innovation magnitude (simple heuristic).
            use_diagonal_covariance: If True, use diagonal covariance approximation (faster for high dimensions).
                                     If None, automatically use diagonal when state_dim > 64.
            min_obs_noise: Minimum threshold for observation noise R (prevents overconfidence).
            use_joseph_form: If True, use Joseph form for covariance update (numerically stable).
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.adaptive = adaptive
        self.min_obs_noise = min_obs_noise
        self.use_joseph_form = use_joseph_form
        
        # Auto-select diagonal covariance for high dimensions (>64)
        if use_diagonal_covariance is None:
            self.use_diagonal_covariance = state_dim > 64
        else:
            self.use_diagonal_covariance = use_diagonal_covariance

        # State (Mean) and Covariance
        self.x: np.ndarray = np.zeros(state_dim)
        if self.use_diagonal_covariance:
            self.P: np.ndarray = np.ones(state_dim) * 1.0  # Diagonal as 1D array
        else:
            self.P: np.ndarray = np.eye(state_dim) * 1.0  # Full covariance matrix

        # Noise Covariances
        if self.use_diagonal_covariance:
            self.Q: np.ndarray = np.ones(state_dim) * process_noise
            self.R: np.ndarray = np.ones(obs_dim) * obs_noise
        else:
            self.Q: np.ndarray = np.eye(state_dim) * process_noise
            self.R: np.ndarray = np.eye(obs_dim) * obs_noise

        # Identity matrix cache (only needed for full covariance)
        self.I: np.ndarray = np.eye(state_dim)

    def reset_state(self) -> None:
        """Reset state and covariance to initial conditions."""
        self.x = np.zeros(self.state_dim)
        if self.use_diagonal_covariance:
            self.P = np.ones(self.state_dim) * 1.0
        else:
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
        if self.use_diagonal_covariance:
            # For diagonal covariance, simplify: P[i] = sum_j F[i,j]^2 * P[j] + Q[i]
            # Assuming diagonal Q and ignoring cross-terms for efficiency
            P_new = np.zeros_like(self.P)
            for i in range(self.state_dim):
                P_new[i] = np.sum(F[i, :] ** 2 * self.P) + self.Q[i]
            self.P = P_new
        else:
            self.P = F @ self.P @ F.T + self.Q

    def update(
        self,
        z: np.ndarray,
        H: Optional[np.ndarray] = None,
        R_override: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform update step (Measurement Update).

        This method implements the Kalman filter measurement update with full support
        for both diagonal and full covariance matrices. For diagonal covariance (O(N)),
        it uses element-wise operations for maximum efficiency with high-dimensional
        state spaces (e.g., 256+ dimensions).

        Args:
            z: Observation vector.
            H: Observation matrix (default: Identity mapping state to obs).
            R_override: Optional override for observation noise covariance.

        Returns:
            Tuple of (Updated State x, Updated Covariance P).

        Raises:
            ValueError: If observation dimension mismatch.
        """
        if H is None:
            # Assume 1-to-1 observation of first obs_dim states
            H = np.eye(self.obs_dim, self.state_dim)

        R = R_override if R_override is not None else self.R
        
        # Enforce minimum observation noise to prevent overconfidence
        if self.use_diagonal_covariance:
            R = np.maximum(R, self.min_obs_noise)
        else:
            R = np.maximum(R, np.eye(R.shape[0]) * self.min_obs_noise)

        # Optimization: Fully diagonal update for O(N) performance
        # Only applies when H is identity and dimensions match
        if self.use_diagonal_covariance and np.array_equal(H, np.eye(self.state_dim)):
            # TRULY DIAGONAL UPDATE - O(N) instead of O(N³)
            # This is the critical optimization for high-dimensional systems
            
            # Innovation: y = z - x
            y = z - self.x
            
            # Innovation covariance: S = P + R (element-wise)
            S = self.P + R
            
            # Numerical safety: prevent division by zero
            S_safe = np.maximum(S, 1e-6)
            
            # Kalman gain: K = P / S (element-wise division)
            K = self.P / S_safe
            
            # State update: x = x + K * y
            self.x += K * y
            
            # Covariance update: P = (I - K) * P
            # For diagonal: P_new = (1 - K) * P_old (element-wise)
            self.P = (1.0 - K) * self.P
            
            # Floor value: prevent filter "lock-up" by maintaining minimum uncertainty
            # This ensures the filter remains responsive to new measurements
            self.P = np.maximum(self.P, self.min_obs_noise)
            
            # Adaptive noise estimation if enabled
            if self.adaptive:
                # Use diagonal-friendly adaptive noise
                self._adapt_noise_diagonal(y, S)
            
            return self.x, self.P
        
        # Standard update for general case (or full covariance)
        return self._update_standard(z, H, R)

    def _update_standard(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standard Kalman filter update for non-identity observation matrices.
        
        This method implements the full Kalman filter update equations with O(N³)
        complexity. Used when H is not identity or full covariance is required.

        Args:
            z: Observation vector.
            H: Observation matrix.
            R: Observation noise covariance.

        Returns:
            Tuple of (Updated State x, Updated Covariance P).
        """
        # y = z - Hx (Innovation)
        y = z - H @ self.x

        # S = HP'H' + R
        if self.use_diagonal_covariance:
            # Simplified for diagonal P and R
            # For diagonal P: HP'H' = H * diag(P) * H' (element-wise)
            # This gives a full matrix, not diagonal, due to cross-terms
            P_full = np.diag(self.P)
            S = H @ P_full @ H.T + np.diag(R)
        else:
            S = H @ self.P @ H.T + R

        # K = P'H' inv(S) (Kalman Gain)
        try:
            if self.use_diagonal_covariance:
                # For diagonal P, use full covariance in this step for accuracy
                P_full = np.diag(self.P)
                K = P_full @ H.T @ np.linalg.inv(S)
            else:
                K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix using Pseudo-inverse
            if self.use_diagonal_covariance:
                P_full = np.diag(self.P)
                K = P_full @ H.T @ np.linalg.pinv(S)
            else:
                K = self.P @ H.T @ np.linalg.pinv(S)

        # x = x' + Ky
        self.x = self.x + K @ y

        # P = (I - KH)P'  (or Joseph form for numerical stability)
        if self.use_joseph_form:
            # Joseph form: P = (I-KH)P'(I-KH)' + KRK'
            # Numerically more stable, guarantees positive semi-definite P
            KH = K @ H
            IKH = self.I - KH
            if self.use_diagonal_covariance:
                # For diagonal case, simplify: P_new = (I-KH)^2 * P_old + diag(KRK')
                P_full = np.diag(self.P)
                KRK_full = K @ np.diag(R) @ K.T
                P_new_full = IKH @ P_full @ IKH.T + KRK_full
                self.P = np.diag(P_new_full)
            else:
                self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        else:
            # Standard form (faster)
            if self.use_diagonal_covariance:
                # Simplified for diagonal covariance
                KH = K @ H
                self.P = np.diag((1 - np.diag(KH)) * self.P)
            else:
                self.P = (self.I - K @ H) @ self.P

        if self.adaptive:
            self._adapt_noise(y, S)

        return self.x, self.P

    def _adapt_noise_diagonal(
        self,
        y: np.ndarray,
        S: np.ndarray
    ) -> None:
        """
        Adapt observation noise R based on innovation (diagonal version).
        
        This implements adaptive noise estimation where R is updated based on
        the magnitude of innovations. Uses element-wise operations for efficiency.

        Args:
            y: Innovation vector (z - x').
            S: Innovation covariance (diagonal).
        """
        innovation_sq = float(np.dot(y, y))
        trace_S = float(np.sum(S))
        
        # Adaptation rate (alpha)
        alpha = 0.1
        
        if innovation_sq > 5.0 * trace_S:
            # Innovation is large - increase observation noise (decrease trust in measurements)
            self.R *= 1.1
        else:
            # Innovation is small - decrease observation noise (increase trust in measurements)
            self.R *= 0.99
        
        # CRITICAL: Enforce minimum observation noise floor to prevent filter lock-up
        # This ensures the filter always remains responsive to new measurements
        self.R = np.maximum(self.R, self.min_obs_noise)

    def _adapt_noise(self, y: np.ndarray, S: np.ndarray) -> None:
        """
        Adapt observation noise R based on innovation (Innovation Adaptation).
        
        This implements adaptive noise estimation where R is updated based on
        the magnitude of innovations. The adaptation follows:
            R_t+1 = (1 - alpha) * R_t + alpha * innovation_t^2
        
        A floor value (min_obs_noise) prevents filter "lock-up" by ensuring
        the filter remains responsive to new measurements.
        
        Args:
            y: Innovation vector (z - Hx').
            S: Innovation covariance.
        """
        innovation_sq = float(np.dot(y, y))
        if self.use_diagonal_covariance:
            trace_S = float(np.sum(S))
        else:
            trace_S = float(np.trace(S))
        
        # Adaptation rate (alpha)
        alpha = 0.1
        
        if innovation_sq > 5.0 * trace_S:
            # Innovation is large - increase observation noise (decrease trust in measurements)
            self.R *= 1.1
        else:
            # Innovation is small - decrease observation noise (increase trust in measurements)
            self.R *= 0.99
        
        # CRITICAL: Enforce minimum observation noise floor to prevent filter lock-up
        # This ensures the filter always remains responsive to new measurements
        self.R = np.maximum(self.R, self.min_obs_noise)

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
        adaptive: bool = False,
        use_diagonal_covariance: bool = None,
        min_obs_noise: float = 1e-6,
        use_joseph_form: bool = False
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
            use_diagonal_covariance: If True, use diagonal covariance approximation.
            min_obs_noise: Minimum threshold for observation noise.
            use_joseph_form: If True, use Joseph form for covariance update.
        """
        super().__init__(
            state_dim, obs_dim, process_noise, obs_noise, adaptive,
            use_diagonal_covariance, min_obs_noise, use_joseph_form
        )
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
        if self.use_diagonal_covariance:
            # Simplified for diagonal covariance
            P_new = np.zeros_like(self.P)
            for i in range(self.state_dim):
                P_new[i] = np.sum(F[i, :] ** 2 * self.P) + self.Q[i]
            self.P = P_new
        else:
            self.P = F @ self.P @ F.T + self.Q
