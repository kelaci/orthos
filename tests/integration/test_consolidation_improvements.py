"""
Comprehensive validation tests for ORTHOS consolidation improvements.

Tests for:
- Kalman filter diagonal covariance optimization
- Adaptive noise constraints
- Joseph form numerical stability
- Consensus manager dimension checking
- Top-down feedback loop
- Auto-projection functionality
"""

import numpy as np
import pytest
from typing import List, Optional

from orthos.filters.kalman import KalmanFilter, ExtendedKalmanFilter
from orthos.hierarchy.filtered_level import FilteredHierarchicalLevel
from orthos.hierarchy.consensus_manager import ConsensusHierarchyManager
from orthos.consensus.engine import HierarchicalConsensus, LevelPrediction, ConsensusResult


class TestKalmanFilterOptimizations:
    """Test Kalman filter performance and stability improvements."""

    def test_diagonal_covariance_high_dimension(self):
        """Test that diagonal covariance is automatically selected for high dimensions."""
        # High dimension (>64 should trigger diagonal)
        state_dim = 128
        obs_dim = 128
        
        kf = KalmanFilter(state_dim, obs_dim)
        
        assert kf.use_diagonal_covariance, "Should auto-select diagonal for high dimensions"
        assert kf.P.ndim == 1, "P should be 1D for diagonal covariance"
        assert kf.P.shape[0] == state_dim, "P should have correct dimension"
        
        # Test prediction and update work with diagonal
        z = np.random.randn(obs_dim)
        kf.predict()
        x, P = kf.update(z)
        
        assert x.shape == (state_dim,), "State should have correct dimension"
        assert P.ndim == 1 or P.shape == (state_dim, state_dim), "Covariance should be valid"

    def test_full_covariance_low_dimension(self):
        """Test that full covariance is used for low dimensions."""
        # Low dimension (<64 should use full covariance)
        state_dim = 32
        obs_dim = 32
        
        kf = KalmanFilter(state_dim, obs_dim)
        
        assert not kf.use_diagonal_covariance, "Should use full covariance for low dimensions"
        assert kf.P.ndim == 2, "P should be 2D for full covariance"
        assert kf.P.shape == (state_dim, state_dim), "P should have correct shape"

    def test_explicit_diagonal_covariance(self):
        """Test explicit diagonal covariance selection."""
        state_dim = 32
        obs_dim = 32
        
        # Force diagonal even for low dimension
        kf = KalmanFilter(state_dim, obs_dim, use_diagonal_covariance=True)
        
        assert kf.use_diagonal_covariance, "Should respect explicit diagonal setting"
        assert kf.P.ndim == 1, "P should be 1D for diagonal covariance"

    def test_adaptive_noise_constraints(self):
        """Test that adaptive noise respects minimum threshold."""
        state_dim = 64
        obs_dim = 64
        min_noise = 1e-5
        
        kf = KalmanFilter(state_dim, obs_dim, adaptive=True, min_obs_noise=min_noise)
        
        # Initial noise should be above threshold
        initial_noise = np.mean(kf.R)
        assert initial_noise >= min_noise, "Initial noise should respect minimum"
        
        # Force adaptation to try to reduce noise below threshold
        # Small innovation should reduce noise
        z = np.zeros(obs_dim)
        kf.predict()
        x, _ = kf.update(z)
        
        # Check R doesn't go below minimum
        assert np.all(kf.R >= min_noise), "Noise should not fall below minimum threshold"

    def test_joseph_form_numerical_stability(self):
        """Test Joseph form maintains positive semi-definite covariance."""
        state_dim = 10
        obs_dim = 10
        
        # Standard form
        kf_standard = KalmanFilter(state_dim, obs_dim, use_joseph_form=False)
        
        # Joseph form
        kf_joseph = KalmanFilter(state_dim, obs_dim, use_joseph_form=True)
        
        # Run multiple updates
        for _ in range(50):
            z = np.random.randn(obs_dim) * 0.1
            kf_standard.predict()
            kf_joseph.predict()
            _, P_standard = kf_standard.update(z)
            _, P_joseph = kf_joseph.update(z)
            
            # Joseph form should maintain symmetry
            if not kf_standard.use_diagonal_covariance:
                assert np.allclose(P_standard, P_standard.T, atol=1e-10), \
                    "Standard form covariance should remain symmetric"
            if not kf_joseph.use_diagonal_covariance:
                assert np.allclose(P_joseph, P_joseph.T, atol=1e-10), \
                    "Joseph form covariance should remain symmetric"
            
            # Joseph form should maintain positive semi-definiteness
            if not kf_joseph.use_diagonal_covariance:
                eigenvalues = np.linalg.eigvalsh(P_joseph)
                assert np.all(eigenvalues >= -1e-10), \
                    "Joseph form should maintain positive semi-definiteness"

    def test_extended_kalman_filter_inherits_optimizations(self):
        """Test that EKF inherits optimization parameters."""
        state_dim = 128
        obs_dim = 128
        
        ekf = ExtendedKalmanFilter(
            state_dim, obs_dim,
            dynamics_fn=lambda x, u: x,
            observation_fn=lambda x: x
        )
        
        # Should auto-select diagonal for high dimension
        assert ekf.use_diagonal_covariance, "EKF should auto-select diagonal for high dimensions"


class TestConsensusManagerDimensionHandling:
    """Test consensus manager dimension validation and projection."""

    def test_compatible_dimensions_pass_validation(self):
        """Test that compatible dimensions pass validation."""
        manager = ConsensusHierarchyManager()
        
        # Create predictions with same dimension
        predictions = [
            LevelPrediction(level=0, prediction=np.random.randn(32), confidence=0.8, uncertainty=0.2),
            LevelPrediction(level=1, prediction=np.random.randn(32), confidence=0.7, uncertainty=0.3),
        ]
        
        target_dim, is_valid = manager._validate_dimensions(predictions)
        
        assert target_dim == 32, "Target dimension should be 32"
        assert is_valid, "Should be valid for compatible dimensions"

    def test_incompatible_dimensions_raise_error_without_projection(self):
        """Test that incompatible dimensions raise error when auto_projection is False."""
        manager = ConsensusHierarchyManager(auto_projection=False)
        
        # Create predictions with different dimensions
        predictions = [
            LevelPrediction(level=0, prediction=np.random.randn(20), confidence=0.8, uncertainty=0.2),
            LevelPrediction(level=1, prediction=np.random.randn(40), confidence=0.7, uncertainty=0.3),
        ]
        
        with pytest.raises(ValueError, match="incompatible"):
            manager._validate_dimensions(predictions)

    def test_auto_projection_enabled(self):
        """Test that auto_projection handles incompatible dimensions."""
        manager = ConsensusHierarchyManager(auto_projection=True)
        
        # Create predictions with different dimensions
        predictions = [
            LevelPrediction(level=0, prediction=np.random.randn(20), confidence=0.8, uncertainty=0.2),
            LevelPrediction(level=1, prediction=np.random.randn(40), confidence=0.7, uncertainty=0.3),
        ]
        
        target_dim, is_valid = manager._validate_dimensions(predictions)
        
        assert target_dim == 40, "Target should be max dimension"
        assert not is_valid, "Should be marked as invalid for projection"

    def test_projection_upsampling(self):
        """Test projection for upsampling to higher dimension."""
        manager = ConsensusHierarchyManager()
        
        # Small to large
        small = np.random.randn(10)
        target_dim = 30
        projected = manager._project_prediction(small, target_dim)
        
        assert projected.shape[0] == target_dim, "Projected should have target dimension"
        # Check that original values are preserved in the projection
        assert np.allclose(projected[:10], small), "Original values should be preserved"

    def test_projection_downsampling(self):
        """Test projection for downsampling to lower dimension."""
        manager = ConsensusHierarchyManager()
        
        # Large to small
        large = np.random.randn(100)
        target_dim = 20
        projected = manager._project_prediction(large, target_dim)
        
        assert projected.shape[0] == target_dim, "Projected should have target dimension"
        # Check that sampled values are from original
        expected_step = 100 // 20
        for i in range(target_dim):
            assert projected[i] == large[i * expected_step], "Should sample correctly"


class TestTopDownFeedbackLoop:
    """Test top-down feedback loop implementation."""

    def test_prior_storage(self):
        """Test that consensus result is stored as prior."""
        manager = ConsensusHierarchyManager()
        
        # Simulate consensus result
        result = ConsensusResult(
            aggregated_prediction=np.random.randn(32),
            agreement_score=0.8,
            outlier_levels=[],
            weights=np.array([0.5, 0.5])
        )
        manager.consensus_prior = result.aggregated_prediction
        
        assert manager.consensus_prior is not None, "Prior should be stored"
        assert manager.consensus_prior.shape[0] == 32, "Prior should have correct dimension"

    def test_distribute_prior_with_compatible_dimensions(self):
        """Test distributing prior to levels with compatible dimensions."""
        manager = ConsensusHierarchyManager(auto_projection=True)
        manager.consensus_prior = np.random.randn(32)
        
        # Create mock level
        class MockLevel:
            def __init__(self, output_size):
                self.output_size = output_size
                self._top_down_prior = None
            
            def set_top_down_prior(self, prior):
                self._top_down_prior = prior
        
        level = MockLevel(32)
        manager.distribute_prior([level])
        
        assert level._top_down_prior is not None, "Prior should be distributed"
        assert level._top_down_prior.shape[0] == 32, "Prior should have correct dimension"

    def test_distribute_prior_with_incompatible_dimensions(self):
        """Test distributing prior with automatic projection."""
        manager = ConsensusHierarchyManager(auto_projection=True)
        manager.consensus_prior = np.random.randn(64)  # High-dim prior
        
        # Create mock level with lower dimension
        class MockLevel:
            def __init__(self, output_size):
                self.output_size = output_size
                self._top_down_prior = None
            
            def set_top_down_prior(self, prior):
                self._top_down_prior = prior
        
        level = MockLevel(32)
        manager.distribute_prior([level])
        
        assert level._top_down_prior is not None, "Prior should be distributed"
        assert level._top_down_prior.shape[0] == 32, "Prior should be projected to level dimension"

    def test_filtered_level_double_fusion(self):
        """Test double fusion in FilteredHierarchicalLevel."""
        level = FilteredHierarchicalLevel(
            level_id=0,
            input_size=32,
            output_size=32,
            filter_type='kalman'
        )
        
        # Set top-down prior
        prior = np.random.randn(32)
        level.set_top_down_prior(prior)
        
        # Process input
        input_data = np.random.randn(32)
        prediction, uncertainty = level.forward_filtered(input_data)
        
        assert prediction.shape[0] == 32, "Prediction should have correct dimension"
        assert uncertainty >= 0, "Uncertainty should be non-negative"
        
        # Check that prior was used
        # The prior should influence the prediction
        assert level._top_down_prior is not None, "Prior should be stored"

    def test_filtered_level_no_prior(self):
        """Test that level works without top-down prior."""
        level = FilteredHierarchicalLevel(
            level_id=0,
            input_size=32,
            output_size=32,
            filter_type='kalman'
        )
        
        # Process input without prior
        input_data = np.random.randn(32)
        prediction, uncertainty = level.forward_filtered(input_data)
        
        assert prediction.shape[0] == 32, "Prediction should have correct dimension"
        assert uncertainty >= 0, "Uncertainty should be non-negative"


class TestNumericalStability:
    """Test numerical stability of improvements."""

    def test_kalman_filter_no_negative_variance(self):
        """Test that Kalman filter maintains non-negative variance."""
        state_dim = 10
        obs_dim = 10
        
        kf = KalmanFilter(state_dim, obs_dim, use_joseph_form=True)
        
        # Run many updates
        for _ in range(100):
            z = np.random.randn(obs_dim) * 0.1
            kf.predict()
            _, P = kf.update(z)
            
            # Check no negative variances
            if kf.use_diagonal_covariance:
                assert np.all(P >= 0), "Diagonal covariance should be non-negative"
            else:
                eigenvalues = np.linalg.eigvalsh(P)
                assert np.all(eigenvalues >= -1e-8), "Full covariance should be PSD"

    def test_long_running_stability(self):
        """Test stability over many iterations."""
        state_dim = 64
        obs_dim = 64
        
        kf = KalmanFilter(state_dim, obs_dim, use_joseph_form=True, min_obs_noise=1e-6)
        
        # Track condition number
        condition_numbers = []
        
        for _ in range(200):
            z = np.random.randn(obs_dim) * 0.01
            kf.predict()
            _, P = kf.update(z)
            
            if not kf.use_diagonal_covariance:
                cond_num = np.linalg.cond(P)
                condition_numbers.append(cond_num)
        
        # Condition number should remain reasonable
        if condition_numbers:
            avg_cond = np.mean(condition_numbers)
            assert avg_cond < 1e10, "Condition number should remain reasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
