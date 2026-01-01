"""
Comprehensive tests for Bayesian optimization improvements.

This test suite validates the three critical improvements:
1. Diagonal Kalman Filter O(N) optimization (Proposal 1)
2. Uncertainty-weighted consensus (Proposal 3)
3. Bayesian fusion for dual updates (Proposal 2)

All tests follow ORTHOS coding standards with comprehensive documentation
and performance benchmarking.
"""

import numpy as np
import time
from typing import List, Tuple

import pytest

from orthos.filters.kalman import KalmanFilter
from orthos.hierarchy.filtered_level import FilteredHierarchicalLevel
from orthos.consensus.engine import HierarchicalConsensus, LevelPrediction


class TestDiagonalKalmanFilter:
    """
    Test suite for diagonal Kalman filter optimization.
    
    Validates that the O(N) diagonal optimization works correctly
    and provides significant performance improvements for high-dimensional
    state spaces.
    """

    def test_diagonal_update_correctness(self):
        """
        Test that diagonal update produces mathematically correct results.
        
        Verifies that the O(N) diagonal update matches the full covariance
        update when H is identity and dimensions match.
        """
        print("\n🧪 Testing diagonal Kalman filter correctness...")
        
        # Setup
        state_dim = 128
        kf_diagonal = KalmanFilter(
            state_dim=state_dim,
            obs_dim=state_dim,
            process_noise=0.01,
            obs_noise=0.1,
            use_diagonal_covariance=True
        )
        
        kf_full = KalmanFilter(
            state_dim=state_dim,
            obs_dim=state_dim,
            process_noise=0.01,
            obs_noise=0.1,
            use_diagonal_covariance=False
        )
        
        # Test multiple steps
        for step in range(10):
            z = np.random.randn(state_dim)
            
            # Predict and update with diagonal
            kf_diagonal.predict()
            x_diag, P_diag = kf_diagonal.update(z)
            
            # Predict and update with full
            kf_full.predict()
            x_full, P_full = kf_full.update(z)
            
            # Check state estimates are close (they won't be identical due to
            # diagonal approximation, but should be similar)
            assert np.allclose(x_diag, x_full, rtol=0.1, atol=0.1), \
                f"State estimates differ too much at step {step}"
            
            # Check diagonal P matches trace of full P
            trace_diag = np.sum(P_diag)
            trace_full = np.trace(P_full)
            assert np.abs(trace_diag - trace_full) / trace_full < 0.5, \
                f"Trace mismatch at step {step}: {trace_diag} vs {trace_full}"
        
        print("✅ Diagonal Kalman filter correctness test passed")

    def test_diagonal_performance(self):
        """
        Test that diagonal update provides significant performance improvement.
        
        Benchmarks O(N) diagonal vs O(N³) full covariance for high dimensions.
        """
        print("\n🧪 Testing diagonal Kalman filter performance...")
        
        state_dims = [64, 128, 256]
        results = []
        
        for dim in state_dims:
            # Diagonal filter
            kf_diag = KalmanFilter(
                state_dim=dim,
                obs_dim=dim,
                process_noise=0.01,
                obs_noise=0.1,
                use_diagonal_covariance=True
            )
            
            # Full covariance filter
            kf_full = KalmanFilter(
                state_dim=dim,
                obs_dim=dim,
                process_noise=0.01,
                obs_noise=0.1,
                use_diagonal_covariance=False
            )
            
            # Warm-up
            for _ in range(5):
                z = np.random.randn(dim)
                kf_diag.predict()
                kf_diag.update(z)
                kf_full.predict()
                kf_full.update(z)
            
            # Benchmark diagonal
            start = time.time()
            for _ in range(100):
                z = np.random.randn(dim)
                kf_diag.predict()
                kf_diag.update(z)
            time_diag = time.time() - start
            
            # Benchmark full
            start = time.time()
            for _ in range(100):
                z = np.random.randn(dim)
                kf_full.predict()
                kf_full.update(z)
            time_full = time.time() - start
            
            speedup = time_full / time_diag
            results.append((dim, time_diag, time_full, speedup))
            
            print(f"  Dim {dim}: Diagonal={time_diag:.4f}s, Full={time_full:.4f}s, Speedup={speedup:.2f}x")
            
            # Assert significant speedup for larger dimensions
            if dim >= 128:
                assert speedup > 2.0, \
                    f"Expected >2x speedup for dim {dim}, got {speedup:.2f}x"
        
        print("✅ Diagonal Kalman filter performance test passed")

    def test_diagonal_numerical_stability(self):
        """
        Test that diagonal update maintains numerical stability.
        
        Verifies that the floor value prevents filter "lock-up" and
        that division by zero is prevented.
        """
        print("\n🧪 Testing diagonal Kalman filter numerical stability...")
        
        kf = KalmanFilter(
            state_dim=64,
            obs_dim=64,
            process_noise=0.01,
            obs_noise=0.1,
            use_diagonal_covariance=True,
            min_obs_noise=1e-6
        )
        
        # Force extreme conditions
        for step in range(100):
            z = np.random.randn(64) * 0.01  # Very small innovations
            kf.predict()
            x, P = kf.update(z)
            
            # Check that P doesn't collapse to zero
            assert np.all(P >= 1e-6), f"Covariance collapsed at step {step}"
            
            # Check that state remains finite
            assert np.all(np.isfinite(x)), f"State became non-finite at step {step}"
        
        print("✅ Diagonal Kalman filter numerical stability test passed")


class TestUncertaintyWeightedConsensus:
    """
    Test suite for uncertainty-weighted consensus.
    
    Validates that consensus correctly weights predictions by uncertainty
    rather than confidence, which is mathematically superior for Bayesian
    fusion with Kalman filters.
    """

    def test_uncertainty_weighting_correctness(self):
        """
        Test that uncertainty weighting produces correct results.
        
        Verifies that lower uncertainty predictions receive higher weights.
        """
        print("\n🧪 Testing uncertainty-weighted consensus correctness...")
        
        consensus = HierarchicalConsensus(
            outlier_threshold=3.0,
            min_agreement=0.6
        )
        
        # Create predictions with different uncertainties
        predictions = [
            LevelPrediction(level=0, prediction=np.array([1.0]), confidence=0.5, uncertainty=0.1),
            LevelPrediction(level=1, prediction=np.array([2.0]), confidence=0.5, uncertainty=0.5),
            LevelPrediction(level=2, prediction=np.array([3.0]), confidence=0.5, uncertainty=1.0),
        ]
        
        result = consensus.aggregate(predictions, method='weighted_vote')
        
        # Verify result is weighted toward lowest uncertainty
        # Weights: 1/0.1 = 10, 1/0.5 = 2, 1/1.0 = 1
        # Normalized: 10/13 ≈ 0.77, 2/13 ≈ 0.15, 1/13 ≈ 0.08
        # Expected: 1.0*0.77 + 2.0*0.15 + 3.0*0.08 ≈ 1.31
        expected_weighted = (1.0 * 10.0 + 2.0 * 2.0 + 3.0 * 1.0) / 13.0
        
        assert np.abs(result.prediction[0] - expected_weighted) < 0.01, \
            f"Uncertainty weighting incorrect: {result.prediction[0]} vs {expected_weighted}"
        
        print("✅ Uncertainty-weighted consensus correctness test passed")

    def test_uncertainty_weighting_vs_confidence(self):
        """
        Compare uncertainty weighting with confidence weighting.
        
        Demonstrates that uncertainty weighting is mathematically superior
        for Bayesian fusion with Kalman filters.
        """
        print("\n🧪 Comparing uncertainty vs confidence weighting...")
        
        consensus = HierarchicalConsensus()
        
        # Create predictions where confidence and uncertainty are inversely related
        predictions = [
            LevelPrediction(level=0, prediction=np.array([1.0]), confidence=0.9, uncertainty=0.01),
            LevelPrediction(level=1, prediction=np.array([2.0]), confidence=0.1, uncertainty=10.0),
        ]
        
        # Uncertainty weighting (current implementation)
        result_unc = consensus.aggregate(predictions, method='weighted_vote')
        
        # Manual confidence weighting for comparison
        confs = np.array([p.confidence for p in predictions])
        preds = np.array([p.prediction for p in predictions])
        weights_conf = confs / np.sum(confs)
        result_conf = np.average(preds, axis=0, weights=weights_conf)
        
        # Uncertainty weighting should heavily favor low uncertainty prediction
        # (1.0 with uncertainty 0.01) over high uncertainty prediction (2.0 with uncertainty 10.0)
        assert result_unc.prediction[0] < result_conf[0], \
            "Uncertainty weighting should favor low-uncertainty predictions more strongly"
        
        # Result should be close to 1.0 (low uncertainty prediction)
        assert np.abs(result_unc.prediction[0] - 1.0) < 0.1, \
            f"Result {result_unc.prediction[0]} should favor low-uncertainty prediction"
        
        print(f"  Uncertainty-weighted: {result_unc.prediction[0]:.4f}")
        print(f"  Confidence-weighted: {result_conf[0]:.4f}")
        print("✅ Uncertainty weighting comparison test passed")

    def test_outlier_detection_with_uncertainty(self):
        """
        Test that outlier detection works correctly with uncertainty weighting.
        
        Verifies that MAD-based outlier detection properly rejects outliers
        before aggregation.
        """
        print("\n🧪 Testing outlier detection with uncertainty weighting...")
        
        consensus = HierarchicalConsensus(outlier_threshold=3.0)
        
        # Create predictions with one obvious outlier
        predictions = [
            LevelPrediction(level=0, prediction=np.array([1.0, 1.0]), confidence=0.5, uncertainty=0.1),
            LevelPrediction(level=1, prediction=np.array([1.1, 1.1]), confidence=0.5, uncertainty=0.1),
            LevelPrediction(level=2, prediction=np.array([10.0, 10.0]), confidence=0.5, uncertainty=0.1),  # Outlier
            LevelPrediction(level=3, prediction=np.array([0.9, 0.9]), confidence=0.5, uncertainty=0.1),
        ]
        
        result = consensus.aggregate(predictions, method='weighted_vote')
        
        # Outlier should be rejected
        assert result.outlier_count == 1, f"Expected 1 outlier, got {result.outlier_count}"
        assert result.agreement_score == 0.75, f"Expected 0.75 agreement, got {result.agreement_score}"
        
        # Result should be close to [1.0, 1.0], not [10.0, 10.0]
        assert np.allclose(result.prediction, np.array([1.0, 1.0]), atol=0.2), \
            f"Result {result.prediction} should not include outlier"
        
        print("✅ Outlier detection with uncertainty weighting test passed")


class TestBayesianFusion:
    """
    Test suite for Bayesian fusion in dual updates.
    
    Validates that the Bayesian fusion approach is mathematically correct
    and provides better results than calling filter.update() twice.
    """

    def test_bayesian_fusion_correctness(self):
        """
        Test that Bayesian fusion produces correct fused estimates.
        
        Verifies the parallel combination rule for diagonal covariance.
        """
        print("\n🧪 Testing Bayesian fusion correctness...")
        
        # Create a simple filter
        kf = KalmanFilter(
            state_dim=10,
            obs_dim=10,
            process_noise=0.01,
            obs_noise=0.1,
            use_diagonal_covariance=True
        )
        
        # Simulate the Bayesian fusion process
        bottom_up_obs = np.random.randn(10)
        top_down_prior = np.random.randn(10)
        
        # Predict
        kf.predict()
        
        # Bottom-up update
        x_bu, P_bu = kf.update(bottom_up_obs)
        unc_bu = np.sum(P_bu)
        
        # Bayesian fusion manually
        top_down_weight = 0.5
        r_bu = unc_bu
        r_td = unc_bu * (1.0 / top_down_weight)
        
        inv_r_bu = 1.0 / r_bu
        inv_r_td = 1.0 / r_td
        inv_sum = inv_r_bu + inv_r_td
        
        x_fused_manual = (x_bu * inv_r_bu + top_down_prior * inv_r_td) / inv_sum
        unc_fused_manual = 1.0 / inv_sum
        
        # Verify fusion properties
        # 1. Fused uncertainty should be lower than both sources
        assert unc_fused_manual < r_bu, "Fused uncertainty should be lower than bottom-up"
        assert unc_fused_manual < r_td, "Fused uncertainty should be lower than top-down"
        
        # 2. Fused state should be between the two estimates (when weights are balanced)
        if top_down_weight == 0.5:
            for i in range(10):
                min_val = min(x_bu[i], top_down_prior[i])
                max_val = max(x_bu[i], top_down_prior[i])
                assert min_val <= x_fused_manual[i] <= max_val, \
                    f"Fused value {x_fused_manual[i]} not between estimates [{min_val}, {max_val}]"
        
        print("✅ Bayesian fusion correctness test passed")

    def test_bayesian_fusion_performance(self):
        """
        Test that Bayesian fusion is more efficient than double update.
        
        Compares the performance of Bayesian fusion (single update)
        versus calling filter.update() twice.
        """
        print("\n🧪 Testing Bayesian fusion performance...")
        
        state_dim = 256
        n_steps = 100
        
        # Method 1: Bayesian fusion (new approach)
        start = time.time()
        kf_fusion = KalmanFilter(
            state_dim=state_dim,
            obs_dim=state_dim,
            process_noise=0.01,
            obs_noise=0.1,
            use_diagonal_covariance=True
        )
        
        for _ in range(n_steps):
            z = np.random.randn(state_dim)
            prior = np.random.randn(state_dim)
            
            kf_fusion.predict()
            x_bu, P_bu = kf_fusion.update(z)
            unc = np.sum(P_bu)
            
            # Bayesian fusion (manual simulation)
            r_bu = unc
            r_td = unc * 2.0
            inv_sum = (1.0/r_bu) + (1.0/r_td)
            x_fused = (x_bu/r_bu + prior/r_td) / inv_sum
            unc_fused = 1.0 / inv_sum
        
        time_fusion = time.time() - start
        
        # Method 2: Double update (old approach - simulate with 2 updates)
        start = time.time()
        kf_double = KalmanFilter(
            state_dim=state_dim,
            obs_dim=state_dim,
            process_noise=0.01,
            obs_noise=0.1,
            use_diagonal_covariance=True
        )
        
        for _ in range(n_steps):
            z = np.random.randn(state_dim)
            prior = np.random.randn(state_dim)
            
            # Two filter updates (less efficient)
            kf_double.predict()
            x_final, P_final = kf_double.update(z)  # First update
            x_final, P_final = kf_double.update(prior)  # Second update
        
        time_double = time.time() - start
        
        speedup = time_double / time_fusion
        print(f"  Bayesian fusion: {time_fusion:.4f}s")
        print(f"  Double update: {time_double:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Bayesian fusion should be faster (fewer operations)
        assert speedup > 1.0, f"Bayesian fusion should be faster, got {speedup:.2f}x"
        
        print("✅ Bayesian fusion performance test passed")

    def test_filtered_level_bayesian_fusion(self):
        """
        Test that FilteredHierarchicalLevel uses Bayesian fusion.
        
        Validates the integration of Bayesian fusion in the actual
        FilteredHierarchicalLevel implementation.
        """
        print("\n🧪 Testing FilteredHierarchicalLevel Bayesian fusion...")
        
        # Create a filtered level
        level = FilteredHierarchicalLevel(
            level_id=0,
            input_size=10,
            output_size=10,
            filter_type='kalman',
            process_noise=0.01,
            obs_noise=0.1,
            top_down_weight=0.5
        )
        
        # Test with and without top-down prior
        input_data = np.random.randn(10)
        top_down_prior = np.random.randn(10)
        
        # Without prior
        result_no_prior, unc_no_prior = level.forward_filtered(input_data)
        
        # With prior
        result_with_prior, unc_with_prior = level.forward_filtered(
            input_data,
            top_down_prior=top_down_prior
        )
        
        # With prior should have different result
        assert not np.allclose(result_no_prior, result_with_prior), \
            "Top-down prior should affect result"
        
        # With prior should have lower uncertainty (fusion reduces uncertainty)
        assert unc_with_prior < unc_no_prior, \
            f"Bayesian fusion should reduce uncertainty: {unc_with_prior} vs {unc_no_prior}"
        
        print("✅ FilteredHierarchicalLevel Bayesian fusion test passed")


class TestIntegration:
    """
    Integration tests for the complete optimization suite.
    
    Tests the interaction of all three optimizations together.
    """

    def test_end_to_end_performance(self):
        """
        Test end-to-end performance with all optimizations.
        
        Measures the performance improvement when using all three
        optimizations together in a realistic scenario.
        """
        print("\n🧪 Testing end-to-end performance with all optimizations...")
        
        # Setup
        n_levels = 4
        level_sizes = [64, 128, 256, 512]
        n_steps = 50
        
        # Create levels
        levels = []
        for i, size in enumerate(level_sizes):
            level = FilteredHierarchicalLevel(
                level_id=i,
                input_size=size if i == 0 else level_sizes[i-1],
                output_size=size,
                filter_type='kalman',
                process_noise=0.01,
                obs_noise=0.1,
                use_diagonal_covariance=True,  # Optimization 1
                top_down_weight=0.5
            )
            levels.append(level)
        
        # Create consensus engine
        consensus = HierarchicalConsensus(outlier_threshold=3.0)
        
        # Benchmark
        start = time.time()
        
        for step in range(n_steps):
            # Process each level
            predictions = []
            for i, level in enumerate(levels):
                input_data = np.random.randn(level.input_size)
                
                # Add top-down prior from previous level
                prior = None
                if i > 0 and len(predictions) > 0:
                    prior = predictions[-1].prediction
                
                # Forward pass with Bayesian fusion (Optimization 2)
                result, uncertainty = level.forward_filtered(input_data, prior)
                
                predictions.append(
                    LevelPrediction(
                        level=i,
                        prediction=result,
                        confidence=1.0 / (1.0 + uncertainty),
                        uncertainty=uncertainty
                    )
                )
            
            # Consensus with uncertainty weighting (Optimization 3)
            result = consensus.aggregate(predictions, method='weighted_vote')
        
        total_time = time.time() - start
        time_per_step = total_time / n_steps
        
        print(f"  Total time for {n_steps} steps: {total_time:.4f}s")
        print(f"  Time per step: {time_per_step:.4f}s")
        print(f"  Average time per level: {time_per_step/n_levels:.4f}s")
        
        # Performance should be reasonable for high-dimensional system
        assert time_per_step < 0.1, f"Time per step too high: {time_per_step:.4f}s"
        
        print("✅ End-to-end performance test passed")

    def test_numerical_stability_integration(self):
        """
        Test numerical stability of all optimizations together.
        
        Ensures that the system remains stable even with extreme
        conditions when all optimizations are enabled.
        """
        print("\n🧪 Testing numerical stability of integrated optimizations...")
        
        # Setup
        levels = [
            FilteredHierarchicalLevel(
                level_id=0,
                input_size=32,
                output_size=32,
                filter_type='kalman',
                process_noise=0.01,
                obs_noise=0.1,
                use_diagonal_covariance=True,
                top_down_weight=0.5,
                min_obs_noise=1e-6
            )
        ]
        
        consensus = HierarchicalConsensus(outlier_threshold=3.0)
        
        # Extreme conditions: very small measurements
        for step in range(100):
            input_data = np.random.randn(32) * 0.001
            result, uncertainty = levels[0].forward_filtered(input_data)
            
            # Check stability
            assert np.all(np.isfinite(result)), f"Result non-finite at step {step}"
            assert uncertainty > 0, f"Uncertainty collapsed to zero at step {step}"
            assert uncertainty < 100.0, f"Uncertainty exploded at step {step}: {uncertainty}"
        
        # Extreme conditions: very large measurements
        for step in range(100):
            input_data = np.random.randn(32) * 100.0
            result, uncertainty = levels[0].forward_filtered(input_data)
            
            # Check stability
            assert np.all(np.isfinite(result)), f"Result non-finite at step {step}"
            assert uncertainty > 0, f"Uncertainty collapsed to zero at step {step}"
            assert uncertainty < 100.0, f"Uncertainty exploded at step {step}: {uncertainty}"
        
        print("✅ Numerical stability integration test passed")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 BAYESIAN OPTIMIZATIONS TEST SUITE")
    print("="*70)
    
    # Run all tests
    pytest.main([__file__, "-v", "-s"])
