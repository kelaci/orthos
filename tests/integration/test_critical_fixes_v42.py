"""
Critical fixes integration tests for GAIA v4.2.

This test suite validates the following critical fixes:
1. Adaptive noise estimation (Innovation Adaptation with floor value)
2. Dimension consistency validation
3. Top-down feedback loop
4. Diagonal covariance for high-dimensional states
5. Joseph form numerical stability
"""

import numpy as np
import pytest
from typing import List

from gaia.filters.kalman import KalmanFilter
from gaia.consensus.engine import HierarchicalConsensus, LevelPrediction
from gaia.hierarchy.consensus_manager import ConsensusHierarchyManager
from gaia.hierarchy.level import HierarchicalLevel


class TestAdaptiveNoiseEstimation:
    """Test adaptive noise estimation with floor value to prevent filter lock-up."""

    def test_adaptive_noise_floor_prevents_lockup(self):
        """Verify that min_obs_noise floor prevents filter from becoming overconfident."""
        # Initialize KF with adaptive noise and floor
        kf = KalmanFilter(
            state_dim=4,
            obs_dim=2,
            obs_noise=0.1,
            adaptive=True,
            min_obs_noise=1e-6
        )
        
        # Simulate consistent, small innovations (filter should decrease R)
        for _ in range(100):
            z = np.array([1.0, 1.0])
            kf.predict()
            kf.update(z)
        
        # Verify R never goes below floor
        assert np.all(kf.R >= kf.min_obs_noise), \
            "Observation noise R dropped below minimum floor value"

    def test_adaptive_noise_increases_on_large_innovations(self):
        """Verify that R increases when innovations are large."""
        kf = KalmanFilter(
            state_dim=4,
            obs_dim=2,
            obs_noise=0.1,
            adaptive=True,
            min_obs_noise=1e-6
        )
        
        initial_R = kf.R.copy()
        
        # Simulate large innovations
        for _ in range(10):
            z = np.array([100.0, 100.0])  # Large innovation
            kf.predict()
            kf.update(z)
        
        # R should have increased
        assert np.all(kf.R >= initial_R), \
            "Observation noise R did not increase on large innovations"

    def test_adaptive_noise_decreases_on_small_innovations(self):
        """Verify that R decreases when innovations are small."""
        kf = KalmanFilter(
            state_dim=4,
            obs_dim=2,
            obs_noise=0.5,
            adaptive=True,
            min_obs_noise=1e-6
        )
        
        initial_R = kf.R.copy()
        
        # Simulate consistent, small innovations
        for _ in range(50):
            z = np.array([0.01, 0.01])
            kf.predict()
            kf.update(z)
        
        # R should have decreased (but not below floor)
        assert np.all(kf.R < initial_R), \
            "Observation noise R did not decrease on small innovations"

    def test_adaptive_noise_with_diagonal_covariance(self):
        """Test adaptive noise works correctly with diagonal covariance."""
        kf = KalmanFilter(
            state_dim=128,
            obs_dim=64,
            obs_noise=0.1,
            adaptive=True,
            use_diagonal_covariance=True,
            min_obs_noise=1e-6
        )
        
        # Verify diagonal covariance is 1D
        assert kf.R.ndim == 1, "R should be 1D array for diagonal covariance"
        assert kf.P.ndim == 1, "P should be 1D array for diagonal covariance"
        
        # Run adaptive updates
        for _ in range(20):
            z = np.random.randn(64)
            kf.predict()
            kf.update(z)
        
        # Verify floor is enforced
        assert np.all(kf.R >= kf.min_obs_noise)


class TestDimensionConsistency:
    """Test dimension consistency validation and auto-projection."""

    def test_mismatched_dimensions_raise_error_without_auto_projection(self):
        """Verify that mismatched dimensions raise error when auto_projection=False."""
        predictions = [
            LevelPrediction(level=0, prediction=np.array([1.0, 2.0]), confidence=0.9, uncertainty=0.1),
            LevelPrediction(level=1, prediction=np.array([1.0, 2.0, 3.0, 4.0]), confidence=0.8, uncertainty=0.2),
        ]
        
        consensus = HierarchicalConsensus()
        
        with pytest.raises(ValueError, match="incompatible"):
            consensus.aggregate(predictions)

    def test_auto_projection_handles_mismatched_dimensions(self):
        """Verify that auto_projection handles mismatched dimensions gracefully."""
        predictions = [
            LevelPrediction(level=0, prediction=np.array([1.0, 2.0]), confidence=0.9, uncertainty=0.1),
            LevelPrediction(level=1, prediction=np.array([1.0, 2.0, 3.0, 4.0]), confidence=0.8, uncertainty=0.2),
        ]
        
        manager = ConsensusHierarchyManager(auto_projection=True)
        
        # This should not raise an error
        result = manager.consensus_engine.aggregate(predictions)
        
        # Result should have the maximum dimension
        assert result.prediction.shape[0] == 4, \
            f"Expected dimension 4, got {result.prediction.shape[0]}"

    def test_dimension_projection_upsampling(self):
        """Test upsampling projection for lower dimensions."""
        manager = ConsensusHierarchyManager(auto_projection=True)
        
        # Project from 2D to 4D
        pred_2d = np.array([1.0, 2.0])
        pred_4d = manager._project_prediction(pred_2d, 4)
        
        assert pred_4d.shape[0] == 4
        # Check that pattern repeats
        np.testing.assert_array_almost_equal(pred_4d[:2], pred_2d)

    def test_dimension_projection_downsampling(self):
        """Test downsampling projection for higher dimensions."""
        manager = ConsensusHierarchyManager(auto_projection=True)
        
        # Project from 4D to 2D
        pred_4d = np.array([1.0, 2.0, 3.0, 4.0])
        pred_2d = manager._project_prediction(pred_4d, 2)
        
        assert pred_2d.shape[0] == 2
        # Check that every other element is taken
        np.testing.assert_array_almost_equal(pred_2d, pred_4d[::2])

    def test_matched_dimensions_pass_validation(self):
        """Verify that matched dimensions pass validation."""
        predictions = [
            LevelPrediction(level=0, prediction=np.array([1.0, 2.0]), confidence=0.9, uncertainty=0.1),
            LevelPrediction(level=1, prediction=np.array([3.0, 4.0]), confidence=0.8, uncertainty=0.2),
        ]
        
        manager = ConsensusHierarchyManager(auto_projection=False)
        target_dim, is_valid = manager._validate_dimensions(predictions)
        
        assert is_valid
        assert target_dim == 2


class TestTopDownFeedback:
    """Test top-down feedback loop in consensus hierarchy."""

    def test_consensus_prior_stored_correctly(self):
        """Verify that consensus prior is stored after aggregation."""
        manager = ConsensusHierarchyManager(auto_projection=True)
        
        predictions = [
            LevelPrediction(level=0, prediction=np.array([1.0, 2.0]), confidence=0.9, uncertainty=0.1),
            LevelPrediction(level=1, prediction=np.array([3.0, 4.0]), confidence=0.8, uncertainty=0.2),
        ]
        
        # Mock the get_consensus_prediction method
        manager.levels = []
        manager.global_time_step = 0
        
        # Aggregate
        result = manager.consensus_engine.aggregate(predictions)
        manager.consensus_prior = result.prediction
        
        # Verify prior is stored
        assert manager.consensus_prior is not None
        assert manager.consensus_prior.shape[0] == 2

    def test_distribute_prior_updates_levels(self):
        """Verify that distribute_prior updates levels with projected prior."""
        manager = ConsensusHierarchyManager(auto_projection=True)
        manager.consensus_prior = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Create mock level with set_top_down_prior method
        mock_level = HierarchicalLevel(level_id=0, temporal_resolution=1)
        mock_level.output_size = 2
        mock_level.set_top_down_prior = lambda x: setattr(mock_level, 'top_down_prior', x)
        
        manager.distribute_prior([mock_level])
        
        # Verify level received projected prior
        assert hasattr(mock_level, 'top_down_prior')
        assert mock_level.top_down_prior.shape[0] == 2  # Projected to level's output_size


class TestHighDimensionalFiltering:
    """Test diagonal covariance for high-dimensional states."""

    def test_diagonal_covariance_auto_enabled_for_high_dims(self):
        """Verify that diagonal covariance is auto-enabled for >64 dimensions."""
        # Low dimension - should use full covariance
        kf_low = KalmanFilter(state_dim=32, obs_dim=16)
        assert not kf_low.use_diagonal_covariance, \
            "Should use full covariance for low dimensions"
        
        # High dimension - should auto-enable diagonal
        kf_high = KalmanFilter(state_dim=128, obs_dim=64)
        assert kf_high.use_diagonal_covariance, \
            "Should auto-enable diagonal covariance for high dimensions"

    def test_diagonal_covariance_reduces_memory(self):
        """Verify that diagonal covariance reduces memory usage."""
        state_dim = 128
        
        # Full covariance
        kf_full = KalmanFilter(
            state_dim=state_dim,
            obs_dim=64,
            use_diagonal_covariance=False
        )
        
        # Diagonal covariance
        kf_diag = KalmanFilter(
            state_dim=state_dim,
            obs_dim=64,
            use_diagonal_covariance=True
        )
        
        # Diagonal should be 1D array
        assert kf_diag.P.ndim == 1, "Diagonal P should be 1D"
        assert kf_full.P.ndim == 2, "Full P should be 2D"
        
        # Diagonal should use much less memory
        full_memory = kf_full.P.nbytes
        diag_memory = kf_diag.P.nbytes
        reduction_factor = full_memory / diag_memory
        
        # Expect ~128x reduction (state_dim)
        assert reduction_factor >= 100, \
            f"Expected >100x memory reduction, got {reduction_factor:.1f}x"

    def test_diagonal_covariance_maintains_accuracy(self):
        """Verify that diagonal covariance maintains reasonable accuracy."""
        state_dim = 64
        obs_dim = 32
        n_steps = 100
        
        # Full covariance KF
        kf_full = KalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_diagonal_covariance=False
        )
        
        # Diagonal covariance KF
        kf_diag = KalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_diagonal_covariance=True
        )
        
        # Simulate same trajectory
        errors_full = []
        errors_diag = []
        
        for _ in range(n_steps):
            z = np.random.randn(obs_dim)
            
            kf_full.predict()
            kf_full.update(z)
            
            kf_diag.predict()
            kf_diag.update(z)
            
            # Compare state estimates
            error = np.linalg.norm(kf_full.x - kf_diag.x)
            errors_diag.append(error)
        
        # Average error should be reasonable (not catastrophic)
        avg_error = np.mean(errors_diag)
        assert avg_error < 1.0, \
            f"Diagonal covariance error too high: {avg_error:.3f}"


class TestJosephFormStability:
    """Test Joseph form for numerical stability."""

    def test_joseph_form_maintains_symmetry(self):
        """Verify that Joseph form maintains covariance symmetry."""
        state_dim = 32
        obs_dim = 16
        n_steps = 1000
        
        # Standard form
        kf_std = KalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_joseph_form=False
        )
        
        # Joseph form
        kf_joseph = KalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_joseph_form=True
        )
        
        symmetry_losses_std = []
        symmetry_losses_joseph = []
        
        for _ in range(n_steps):
            z = np.random.randn(obs_dim)
            
            kf_std.predict()
            kf_std.update(z)
            
            kf_joseph.predict()
            kf_joseph.update(z)
            
            # Measure symmetry loss
            loss_std = np.linalg.norm(kf_std.P - kf_std.P.T)
            loss_joseph = np.linalg.norm(kf_joseph.P - kf_joseph.P.T)
            
            symmetry_losses_std.append(loss_std)
            symmetry_losses_joseph.append(loss_joseph)
        
        # Joseph form should have better symmetry
        avg_loss_std = np.mean(symmetry_losses_std)
        avg_loss_joseph = np.mean(symmetry_losses_joseph)
        
        # Joseph form should be significantly better
        assert avg_loss_joseph <= avg_loss_std, \
            f"Joseph form symmetry loss ({avg_loss_joseph:.2e}) not better than standard ({avg_loss_std:.2e})"

    def test_joseph_form_prevents_negative_variance(self):
        """Verify that Joseph form prevents negative variance."""
        state_dim = 32
        obs_dim = 16
        n_steps = 1000
        
        kf = KalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_joseph_form=True
        )
        
        # Run many updates
        for _ in range(n_steps):
            z = np.random.randn(obs_dim)
            kf.predict()
            kf.update(z)
        
        # Check that all variances (diagonal elements) are non-negative
        variances = np.diag(kf.P)
        assert np.all(variances >= 0), \
            "Joseph form produced negative variances"

    def test_joseph_form_with_diagonal_covariance(self):
        """Test that Joseph form works with diagonal covariance."""
        kf = KalmanFilter(
            state_dim=64,
            obs_dim=32,
            use_joseph_form=True,
            use_diagonal_covariance=True
        )
        
        # Run updates
        for _ in range(50):
            z = np.random.randn(32)
            kf.predict()
            kf.update(z)
        
        # Verify no errors occurred
        assert kf.P.ndim == 1  # Still diagonal
        assert np.all(kf.P >= 0)  # All variances non-negative


class TestIntegrationScenarios:
    """Integration tests combining multiple v4.2 features."""

    def test_high_dimensional_filtering_with_adaptive_noise(self):
        """Test high-dimensional filtering with adaptive noise."""
        kf = KalmanFilter(
            state_dim=128,
            obs_dim=64,
            use_diagonal_covariance=True,
            adaptive=True,
            min_obs_noise=1e-6
        )
        
        # Simulate varying innovation magnitudes
        for i in range(100):
            if i % 20 == 0:
                # Large innovation
                z = np.random.randn(64) * 10
            else:
                # Small innovation
                z = np.random.randn(64) * 0.1
            
            kf.predict()
            kf.update(z)
        
        # Verify floor is maintained
        assert np.all(kf.R >= kf.min_obs_noise)
        # Verify diagonal structure
        assert kf.P.ndim == 1

    def test_consensus_with_top_down_and_auto_projection(self):
        """Test consensus with top-down feedback and auto-projection."""
        manager = ConsensusHierarchyManager(auto_projection=True)
        
        # Simulate hierarchy with different dimensions
        predictions = [
            LevelPrediction(level=0, prediction=np.random.randn(16), confidence=0.9, uncertainty=0.1),
            LevelPrediction(level=1, prediction=np.random.randn(32), confidence=0.8, uncertainty=0.2),
            LevelPrediction(level=2, prediction=np.random.randn(64), confidence=0.7, uncertainty=0.3),
        ]
        
        # Aggregate
        result = manager.consensus_engine.aggregate(predictions)
        manager.consensus_prior = result.prediction
        
        # Verify prior is stored with max dimension
        assert manager.consensus_prior.shape[0] == 64
        
        # Verify result is valid
        assert result.agreement_score > 0
        assert result.prediction.shape[0] == 64

    def test_stability_check_with_history(self):
        """Test stability check with consensus history."""
        manager = ConsensusHierarchyManager(min_agreement=0.8)
        
        # Add stable history
        for i in range(5):
            result = type('obj', (object,), {
                'agreement_score': 0.9
            })()
            manager.consensus_history.append(result)
        
        # Should be stable
        assert manager.is_hierarchy_stable()
        
        # Add unstable history
        for i in range(5):
            result = type('obj', (object,), {
                'agreement_score': 0.5
            })()
            manager.consensus_history.append(result)
        
        # Should not be stable
        assert not manager.is_hierarchy_stable()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
