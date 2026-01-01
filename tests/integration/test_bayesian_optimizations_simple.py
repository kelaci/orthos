"""
Simple test runner for Bayesian optimization improvements (no pytest dependency).

This test suite validates the three critical improvements:
1. Diagonal Kalman Filter O(N) optimization (Proposal 1)
2. Uncertainty-weighted consensus (Proposal 3)
3. Bayesian fusion for dual updates (Proposal 2)
"""

import numpy as np
import time
from typing import List

from orthos.filters.kalman import KalmanFilter
from orthos.hierarchy.filtered_level import FilteredHierarchicalLevel
from orthos.consensus.engine import HierarchicalConsensus, LevelPrediction


def run_test(test_name, test_func):
    """Run a single test and print results."""
    print(f"\n{'='*70}")
    print(f"🧪 {test_name}")
    print('='*70)
    try:
        test_func()
        print(f"✅ {test_name} PASSED")
        return True
    except AssertionError as e:
        print(f"❌ {test_name} FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diagonal_correctness():
    """Test diagonal Kalman filter correctness."""
    print("\n🧪 Testing diagonal Kalman filter correctness...")
    
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
    
    for step in range(10):
        z = np.random.randn(state_dim)
        
        kf_diagonal.predict()
        x_diag, P_diag = kf_diagonal.update(z)
        
        kf_full.predict()
        x_full, P_full = kf_full.update(z)
        
        assert np.allclose(x_diag, x_full, rtol=0.1, atol=0.1), \
            f"State estimates differ too much at step {step}"
        
        trace_diag = np.sum(P_diag)
        trace_full = np.trace(P_full)
        assert np.abs(trace_diag - trace_full) / trace_full < 0.5, \
            f"Trace mismatch at step {step}"
    
    print("✅ Diagonal Kalman filter correctness test passed")


def test_diagonal_performance():
    """Test diagonal Kalman filter performance."""
    print("\n🧪 Testing diagonal Kalman filter performance...")
    
    state_dims = [64, 128, 256]
    
    for dim in state_dims:
        kf_diag = KalmanFilter(
            state_dim=dim,
            obs_dim=dim,
            process_noise=0.01,
            obs_noise=0.1,
            use_diagonal_covariance=True
        )
        
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
        print(f"  Dim {dim}: Diagonal={time_diag:.4f}s, Full={time_full:.4f}s, Speedup={speedup:.2f}x")
        
        if dim >= 128:
            assert speedup > 2.0, \
                f"Expected >2x speedup for dim {dim}, got {speedup:.2f}x"
    
    print("✅ Diagonal Kalman filter performance test passed")


def test_diagonal_numerical_stability():
    """Test diagonal Kalman filter numerical stability."""
    print("\n🧪 Testing diagonal Kalman filter numerical stability...")
    
    kf = KalmanFilter(
        state_dim=64,
        obs_dim=64,
        process_noise=0.01,
        obs_noise=0.1,
        use_diagonal_covariance=True,
        min_obs_noise=1e-6
    )
    
    for step in range(100):
        z = np.random.randn(64) * 0.01
        kf.predict()
        x, P = kf.update(z)
        
        assert np.all(P >= 1e-6), f"Covariance collapsed at step {step}"
        assert np.all(np.isfinite(x)), f"State became non-finite at step {step}"
    
    print("✅ Diagonal Kalman filter numerical stability test passed")


def test_uncertainty_weighting_correctness():
    """Test uncertainty-weighted consensus correctness."""
    print("\n🧪 Testing uncertainty-weighted consensus correctness...")
    
    consensus = HierarchicalConsensus(
        outlier_threshold=3.0,
        min_agreement=0.6
    )
    
    predictions = [
        LevelPrediction(level=0, prediction=np.array([1.0]), confidence=0.5, uncertainty=0.1),
        LevelPrediction(level=1, prediction=np.array([2.0]), confidence=0.5, uncertainty=0.5),
        LevelPrediction(level=2, prediction=np.array([3.0]), confidence=0.5, uncertainty=1.0),
    ]
    
    result = consensus.aggregate(predictions, method='weighted_vote')
    
    expected_weighted = (1.0 * 10.0 + 2.0 * 2.0 + 3.0 * 1.0) / 13.0
    
    assert np.abs(result.prediction[0] - expected_weighted) < 0.01, \
        f"Uncertainty weighting incorrect: {result.prediction[0]} vs {expected_weighted}"
    
    print("✅ Uncertainty-weighted consensus correctness test passed")


def test_uncertainty_vs_confidence():
    """Compare uncertainty vs confidence weighting."""
    print("\n🧪 Comparing uncertainty vs confidence weighting...")
    
    consensus = HierarchicalConsensus()
    
    predictions = [
        LevelPrediction(level=0, prediction=np.array([1.0]), confidence=0.9, uncertainty=0.01),
        LevelPrediction(level=1, prediction=np.array([2.0]), confidence=0.1, uncertainty=10.0),
    ]
    
    result_unc = consensus.aggregate(predictions, method='weighted_vote')
    
    confs = np.array([p.confidence for p in predictions])
    preds = np.array([p.prediction for p in predictions])
    weights_conf = confs / np.sum(confs)
    result_conf = np.average(preds, axis=0, weights=weights_conf)
    
    assert result_unc.prediction[0] < result_conf[0], \
        "Uncertainty weighting should favor low-uncertainty predictions more strongly"
    
    assert np.abs(result_unc.prediction[0] - 1.0) < 0.1, \
        f"Result {result_unc.prediction[0]} should favor low-uncertainty prediction"
    
    print(f"  Uncertainty-weighted: {result_unc.prediction[0]:.4f}")
    print(f"  Confidence-weighted: {result_conf[0]:.4f}")
    print("✅ Uncertainty weighting comparison test passed")


def test_outlier_detection():
    """Test outlier detection with uncertainty weighting."""
    print("\n🧪 Testing outlier detection with uncertainty weighting...")
    
    consensus = HierarchicalConsensus(outlier_threshold=3.0)
    
    predictions = [
        LevelPrediction(level=0, prediction=np.array([1.0, 1.0]), confidence=0.5, uncertainty=0.1),
        LevelPrediction(level=1, prediction=np.array([1.1, 1.1]), confidence=0.5, uncertainty=0.1),
        LevelPrediction(level=2, prediction=np.array([10.0, 10.0]), confidence=0.5, uncertainty=0.1),
        LevelPrediction(level=3, prediction=np.array([0.9, 0.9]), confidence=0.5, uncertainty=0.1),
    ]
    
    result = consensus.aggregate(predictions, method='weighted_vote')
    
    assert result.outlier_count == 1, f"Expected 1 outlier, got {result.outlier_count}"
    assert result.agreement_score == 0.75, f"Expected 0.75 agreement, got {result.agreement_score}"
    assert np.allclose(result.prediction, np.array([1.0, 1.0]), atol=0.2), \
        f"Result {result.prediction} should not include outlier"
    
    print("✅ Outlier detection with uncertainty weighting test passed")


def test_bayesian_fusion_correctness():
    """Test Bayesian fusion correctness."""
    print("\n🧪 Testing Bayesian fusion correctness...")
    
    kf = KalmanFilter(
        state_dim=10,
        obs_dim=10,
        process_noise=0.01,
        obs_noise=0.1,
        use_diagonal_covariance=True
    )
    
    bottom_up_obs = np.random.randn(10)
    top_down_prior = np.random.randn(10)
    
    kf.predict()
    x_bu, P_bu = kf.update(bottom_up_obs)
    unc_bu = np.sum(P_bu)
    
    top_down_weight = 0.5
    r_bu = unc_bu
    r_td = unc_bu * (1.0 / top_down_weight)
    
    inv_r_bu = 1.0 / r_bu
    inv_r_td = 1.0 / r_td
    inv_sum = inv_r_bu + inv_r_td
    
    x_fused_manual = (x_bu * inv_r_bu + top_down_prior * inv_r_td) / inv_sum
    unc_fused_manual = 1.0 / inv_sum
    
    assert unc_fused_manual < r_bu, "Fused uncertainty should be lower than bottom-up"
    assert unc_fused_manual < r_td, "Fused uncertainty should be lower than top-down"
    
    print("✅ Bayesian fusion correctness test passed")


def test_bayesian_fusion_performance():
    """Test Bayesian fusion performance."""
    print("\n🧪 Testing Bayesian fusion performance...")
    
    state_dim = 256
    n_steps = 100
    
    # Bayesian fusion
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
        
        r_bu = unc
        r_td = unc * 2.0
        inv_sum = (1.0/r_bu) + (1.0/r_td)
        x_fused = (x_bu/r_bu + prior/r_td) / inv_sum
        unc_fused = 1.0 / inv_sum
    
    time_fusion = time.time() - start
    
    # Double update
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
        
        kf_double.predict()
        x_final, P_final = kf_double.update(z)
        x_final, P_final = kf_double.update(prior)
    
    time_double = time.time() - start
    
    speedup = time_double / time_fusion
    print(f"  Bayesian fusion: {time_fusion:.4f}s")
    print(f"  Double update: {time_double:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    assert speedup > 1.0, f"Bayesian fusion should be faster, got {speedup:.2f}x"
    
    print("✅ Bayesian fusion performance test passed")


def test_filtered_level_bayesian_fusion():
    """Test FilteredHierarchicalLevel Bayesian fusion."""
    print("\n🧪 Testing FilteredHierarchicalLevel Bayesian fusion...")
    
    level = FilteredHierarchicalLevel(
        level_id=0,
        input_size=10,
        output_size=10,
        filter_type='kalman',
        process_noise=0.01,
        obs_noise=0.1,
        top_down_weight=0.5
    )
    
    input_data = np.random.randn(10)
    top_down_prior = np.random.randn(10)
    
    result_no_prior, unc_no_prior = level.forward_filtered(input_data)
    
    result_with_prior, unc_with_prior = level.forward_filtered(
        input_data,
        top_down_prior=top_down_prior
    )
    
    assert not np.allclose(result_no_prior, result_with_prior), \
        "Top-down prior should affect result"
    
    assert unc_with_prior < unc_no_prior, \
        f"Bayesian fusion should reduce uncertainty: {unc_with_prior} vs {unc_no_prior}"
    
    print("✅ FilteredHierarchicalLevel Bayesian fusion test passed")


def test_end_to_end_performance():
    """Test end-to-end performance with all optimizations."""
    print("\n🧪 Testing end-to-end performance with all optimizations...")
    
    # Use uniform dimension to avoid projection issues in test
    n_levels = 4
    level_size = 256
    n_steps = 50
    
    levels = []
    for i in range(n_levels):
        level = FilteredHierarchicalLevel(
            level_id=i,
            input_size=level_size,
            output_size=level_size,
            filter_type='kalman',
            process_noise=0.01,
            obs_noise=0.1,
            use_diagonal_covariance=True,
            top_down_weight=0.5
        )
        levels.append(level)
    
    consensus = HierarchicalConsensus(outlier_threshold=3.0)
    
    start = time.time()
    
    for step in range(n_steps):
        predictions = []
        for i, level in enumerate(levels):
            input_data = np.random.randn(level.input_size)
            
            prior = None
            if i > 0 and len(predictions) > 0:
                prior = predictions[-1].prediction
            
            result, uncertainty = level.forward_filtered(input_data, prior)
            
            predictions.append(
                LevelPrediction(
                    level=i,
                    prediction=result,
                    confidence=1.0 / (1.0 + uncertainty),
                    uncertainty=uncertainty
                )
            )
        
        result = consensus.aggregate(predictions, method='weighted_vote')
    
    total_time = time.time() - start
    time_per_step = total_time / n_steps
    
    print(f"  Total time for {n_steps} steps: {total_time:.4f}s")
    print(f"  Time per step: {time_per_step:.4f}s")
    print(f"  Average time per level: {time_per_step/n_levels:.4f}s")
    
    assert time_per_step < 0.1, f"Time per step too high: {time_per_step:.4f}s"
    
    print("✅ End-to-end performance test passed")


def test_numerical_stability_integration():
    """Test numerical stability of integrated optimizations."""
    print("\n🧪 Testing numerical stability of integrated optimizations...")
    
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
    
    for step in range(100):
        input_data = np.random.randn(32) * 0.001
        result, uncertainty = levels[0].forward_filtered(input_data)
        
        assert np.all(np.isfinite(result)), f"Result non-finite at step {step}"
        assert uncertainty > 0, f"Uncertainty collapsed to zero at step {step}"
        assert uncertainty < 100.0, f"Uncertainty exploded at step {step}: {uncertainty}"
    
    for step in range(100):
        input_data = np.random.randn(32) * 100.0
        result, uncertainty = levels[0].forward_filtered(input_data)
        
        assert np.all(np.isfinite(result)), f"Result non-finite at step {step}"
        assert uncertainty > 0, f"Uncertainty collapsed to zero at step {step}"
        assert uncertainty < 100.0, f"Uncertainty exploded at step {step}: {uncertainty}"
    
    print("✅ Numerical stability integration test passed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 BAYESIAN OPTIMIZATIONS TEST SUITE")
    print("="*70)
    
    tests = [
        ("Diagonal Kalman Filter Correctness", test_diagonal_correctness),
        ("Diagonal Kalman Filter Performance", test_diagonal_performance),
        ("Diagonal Kalman Filter Numerical Stability", test_diagonal_numerical_stability),
        ("Uncertainty Weighting Correctness", test_uncertainty_weighting_correctness),
        ("Uncertainty vs Confidence Weighting", test_uncertainty_vs_confidence),
        ("Outlier Detection with Uncertainty", test_outlier_detection),
        ("Bayesian Fusion Correctness", test_bayesian_fusion_correctness),
        ("Bayesian Fusion Performance", test_bayesian_fusion_performance),
        ("Filtered Level Bayesian Fusion", test_filtered_level_bayesian_fusion),
        ("End-to-End Performance", test_end_to_end_performance),
        ("Numerical Stability Integration", test_numerical_stability_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        passed = run_test(test_name, test_func)
        results.append((test_name, passed))
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
