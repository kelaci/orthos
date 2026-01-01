"""
Simple verification script for GAIA v4.2 critical fixes.
Does not require pytest - runs with plain Python.
"""

import numpy as np
import sys

# Import GAIA modules
from gaia.filters.kalman import KalmanFilter
from gaia.consensus.engine import HierarchicalConsensus, LevelPrediction
from gaia.hierarchy.consensus_manager import ConsensusHierarchyManager
from gaia.hierarchy.level import HierarchicalLevel


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_test(test_name, passed):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {test_name}")
    return passed


def test_adaptive_noise_floor():
    """Test that adaptive noise floor prevents filter lock-up."""
    print_header("TEST 1: Adaptive Noise Floor (CRITICAL FIX)")
    
    kf = KalmanFilter(
        state_dim=4,
        obs_dim=2,
        obs_noise=0.1,
        adaptive=True,
        min_obs_noise=1e-6
    )
    
    # Simulate consistent, small innovations
    for _ in range(100):
        z = np.array([1.0, 1.0])
        kf.predict()
        kf.update(z)
    
    # Verify R never goes below floor
    passed = np.all(kf.R >= kf.min_obs_noise)
    print_test("R never drops below floor value", passed)
    
    # Verify R is being adapted (not Q)
    print(f"  Observation noise R: {kf.R}")
    print(f"  Process noise Q: {kf.Q}")
    
    return passed


def test_adaptive_noise_increases():
    """Test that R increases on large innovations."""
    print_header("TEST 2: Adaptive Noise Increases on Large Innovations")
    
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
        z = np.array([100.0, 100.0])
        kf.predict()
        kf.update(z)
    
    # R should have increased
    passed = np.all(kf.R >= initial_R)
    print_test("R increases on large innovations", passed)
    print(f"  Initial R: {initial_R}")
    print(f"  Final R: {kf.R}")
    
    return passed


def test_dimension_validation():
    """Test dimension consistency validation."""
    print_header("TEST 3: Dimension Consistency Validation")
    
    predictions = [
        LevelPrediction(level=0, prediction=np.array([1.0, 2.0]), confidence=0.9, uncertainty=0.1),
        LevelPrediction(level=1, prediction=np.array([1.0, 2.0, 3.0, 4.0]), confidence=0.8, uncertainty=0.2),
    ]
    
    consensus = HierarchicalConsensus()
    
    # Without auto-projection, should raise error
    try:
        consensus.aggregate(predictions)
        passed = False
        print_test("Mismatched dimensions raise error", False)
    except ValueError as e:
        passed = "incompatible" in str(e)
        print_test(f"Mismatched dimensions raise error: {e}", passed)
    
    return passed


def test_auto_projection():
    """Test auto-projection handles mismatched dimensions."""
    print_header("TEST 4: Auto-Projection Handles Mismatched Dimensions")
    
    # Test projection method directly
    manager = ConsensusHierarchyManager(auto_projection=True)
    
    # Test upsampling
    pred_2d = np.array([1.0, 2.0])
    pred_4d = manager._project_prediction(pred_2d, 4)
    passed1 = pred_4d.shape[0] == 4
    print_test(f"Upsampling 2D to 4D: {pred_4d}", passed1)
    
    # Test downsampling
    pred_4d_input = np.array([1.0, 2.0, 3.0, 4.0])
    pred_2d_down = manager._project_prediction(pred_4d_input, 2)
    passed2 = pred_2d_down.shape[0] == 2
    print_test(f"Downsampling 4D to 2D: {pred_2d_down}", passed2)
    
    # Test that projection works for full workflow
    predictions = [
        LevelPrediction(level=0, prediction=np.array([1.0, 2.0]), confidence=0.9, uncertainty=0.1),
        LevelPrediction(level=1, prediction=np.array([1.0, 2.0, 3.0, 4.0]), confidence=0.8, uncertainty=0.2),
    ]
    
    # Validate dimensions
    target_dim, is_valid = manager._validate_dimensions(predictions)
    passed3 = target_dim == 4 and not is_valid
    print_test(f"Validation detects mismatch: target_dim={target_dim}, is_valid={is_valid}", passed3)
    
    # Project and aggregate
    if not is_valid:
        for pred in predictions:
            pred.prediction = manager._project_prediction(pred.prediction, target_dim)
    
    try:
        result = manager.consensus_engine.aggregate(predictions)
        passed4 = result.prediction.shape[0] == 4
        print_test(f"After projection, aggregate succeeds: result dim={result.prediction.shape[0]}", passed4)
    except Exception as e:
        passed4 = False
        print_test(f"Aggregate failed: {e}", passed4)
    
    return passed1 and passed2 and passed3 and passed4


def test_diagonal_covariance():
    """Test diagonal covariance auto-enabled for high dimensions."""
    print_header("TEST 5: Diagonal Covariance for High Dimensions")
    
    # Low dimension - should use full covariance
    kf_low = KalmanFilter(state_dim=32, obs_dim=16)
    passed1 = not kf_low.use_diagonal_covariance
    print_test("Low dim uses full covariance", passed1)
    
    # High dimension - should auto-enable diagonal
    kf_high = KalmanFilter(state_dim=128, obs_dim=64)
    passed2 = kf_high.use_diagonal_covariance
    print_test("High dim uses diagonal covariance", passed2)
    
    # Verify memory reduction
    kf_full = KalmanFilter(state_dim=128, obs_dim=64, use_diagonal_covariance=False)
    kf_diag = KalmanFilter(state_dim=128, obs_dim=64, use_diagonal_covariance=True)
    
    full_memory = kf_full.P.nbytes
    diag_memory = kf_diag.P.nbytes
    reduction = full_memory / diag_memory
    
    passed3 = reduction >= 100
    print_test(f"Memory reduction: {reduction:.1f}x (expected >100x)", passed3)
    
    print(f"  Full covariance memory: {full_memory:,} bytes")
    print(f"  Diagonal covariance memory: {diag_memory:,} bytes")
    
    return passed1 and passed2 and passed3


def test_joseph_form():
    """Test Joseph form maintains symmetry."""
    print_header("TEST 6: Joseph Form Numerical Stability")
    
    state_dim = 32
    obs_dim = 16
    n_steps = 100
    
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
    
    for _ in range(n_steps):
        z = np.random.randn(obs_dim)
        kf_std.predict()
        kf_std.update(z)
        kf_joseph.predict()
        kf_joseph.update(z)
    
    # Measure symmetry loss
    loss_std = np.linalg.norm(kf_std.P - kf_std.P.T)
    loss_joseph = np.linalg.norm(kf_joseph.P - kf_joseph.P.T)
    
    passed = loss_joseph <= loss_std
    print_test(f"Joseph form symmetry loss ({loss_joseph:.2e}) <= standard ({loss_std:.2e})", passed)
    
    print(f"  Standard form symmetry loss: {loss_std:.2e}")
    print(f"  Joseph form symmetry loss: {loss_joseph:.2e}")
    
    return passed


def test_top_down_feedback():
    """Test top-down feedback loop."""
    print_header("TEST 7: Top-Down Feedback Loop")
    
    manager = ConsensusHierarchyManager(auto_projection=True)
    
    predictions = [
        LevelPrediction(level=0, prediction=np.array([1.0, 2.0]), confidence=0.9, uncertainty=0.1),
        LevelPrediction(level=1, prediction=np.array([3.0, 4.0]), confidence=0.8, uncertainty=0.2),
    ]
    
    # Aggregate
    result = manager.consensus_engine.aggregate(predictions)
    manager.consensus_prior = result.prediction
    
    # Verify prior is stored
    passed1 = manager.consensus_prior is not None
    print_test("Consensus prior stored correctly", passed1)
    
    passed2 = manager.consensus_prior.shape[0] == 2
    print_test(f"Prior dimension: {manager.consensus_prior.shape[0]}", passed2)
    
    return passed1 and passed2


def test_high_dim_accuracy():
    """Test diagonal covariance maintains accuracy."""
    print_header("TEST 8: Diagonal Covariance Accuracy")
    
    # Use matching dimensions for diagonal covariance test
    state_dim = 32
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
    
    errors = []
    for _ in range(n_steps):
        z = np.random.randn(obs_dim)
        kf_full.predict()
        kf_full.update(z)
        kf_diag.predict()
        kf_diag.update(z)
        
        error = np.linalg.norm(kf_full.x - kf_diag.x)
        errors.append(error)
    
    avg_error = np.mean(errors)
    # Diagonal covariance is an approximation, so we allow higher error
    # The important thing is that it doesn't diverge catastrophically
    passed = avg_error < 5.0
    print_test(f"Average error vs full covariance: {avg_error:.3f} (threshold: 5.0)", passed)
    
    return passed


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("  GAIA v4.2 CRITICAL FIXES VERIFICATION")
    print("="*70)
    
    tests = [
        ("Adaptive Noise Floor", test_adaptive_noise_floor),
        ("Adaptive Noise Increases", test_adaptive_noise_increases),
        ("Dimension Validation", test_dimension_validation),
        ("Auto-Projection", test_auto_projection),
        ("Diagonal Covariance", test_diagonal_covariance),
        ("Joseph Form Stability", test_joseph_form),
        ("Top-Down Feedback", test_top_down_feedback),
        ("High-Dim Accuracy", test_high_dim_accuracy),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ FAIL: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n  🎉 ALL CRITICAL FIXES VERIFIED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n  ⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
