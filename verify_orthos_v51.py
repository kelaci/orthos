"""
Verification script for ORTHOS v5.1.0 Master Improvements.

Tests:
- SquareRootKalmanFilter numerical stability.
- BlockDiagonalKalmanFilter execution.
- FilteredHierarchicalLevel dynamic modulation and concept detection.
- ConsensusHierarchyManager global concept broadcasting.
"""

import numpy as np
import time
from orthos.filters.kalman import KalmanFilter, SquareRootKalmanFilter, BlockDiagonalKalmanFilter
from orthos.hierarchy.filtered_level import FilteredHierarchicalLevel
from orthos.hierarchy.consensus_manager import ConsensusHierarchyManager
from orthos.layers.reactive import ReactiveLayer

def test_sr_kalman_stability():
    """Verify SR-KF is numerically stable and matches KF in simple cases."""
    print("🧪 Testing SquareRootKalmanFilter Stability...")
    
    dim = 4
    obs_dim = 2
    kf = KalmanFilter(dim, obs_dim, use_joseph_form=True)
    srkf = SquareRootKalmanFilter(dim, obs_dim)
    
    # Force H for both
    H = np.zeros((obs_dim, dim))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    
    # Run many steps
    for i in range(100):
        z = np.array([1.0, 0.5]) + np.random.randn(2) * 0.1
        
        kf.predict()
        kf.update(z, H=H)
        
        srkf.predict()
        srkf.update(z, H=H)
        
        # Check if state is reasonably close
        if i == 99:
            diff = np.linalg.norm(kf.x - srkf.x)
            print(f"  State difference after 100 steps: {diff:.6e}")
            assert diff < 1e-2, "SR-KF and KF state diverged significantly"

    print("✅ SR-KF stability test passed")

def test_block_diagonal_kf():
    """Verify BlockDiagonalKalmanFilter functionality."""
    print("\n🧪 Testing BlockDiagonalKalmanFilter...")
    
    dim = 8
    obs_dim = 8
    # Define blocks: [0,1,2,3] and [4,5,6,7]
    block_structure = [[0,1,2,3], [4,5,6,7]]
    bdkf = BlockDiagonalKalmanFilter(dim, obs_dim, block_structure=block_structure)
    
    z = np.random.randn(8)
    bdkf.predict()
    x, P = bdkf.update(z)
    
    assert x.shape == (8,), "Output state shape mismatch"
    assert len(bdkf.blocks) == 2, "Blocks not initialized correctly"
    print("✅ Block-Diagonal KF test passed")

def test_dynamic_modulation():
    """Verify FilteredHierarchicalLevel dynamic modulation."""
    print("\n🧪 Testing Dynamic Top-Down Modulation...")
    
    output_size = 10
    level = FilteredHierarchicalLevel(
        level_id=1,
        input_size=10,
        output_size=output_size,
        filter_type='sr_kalman',
        dynamic_modulation=True
    )
    
    # Manually add a layer so process_time_step works
    level.add_layer(ReactiveLayer(10, output_size))
    
    # 1. Test Stable Regime
    print("  Situating in Stable regime...")
    for _ in range(10):
        input_data = np.ones(10)
        level.forward_filtered(input_data)
        
    concept = level._detect_concept_state()
    print(f"  Detected concept: {concept}")
    
    # 2. Test Storm Regime (sudden change)
    print("  Simulating 'Storm' (sudden jump)...")
    for _ in range(5):
        input_data = np.ones(10) * 10.0 # Huge jump
        level.forward_filtered(input_data)
        
    concept = level._detect_concept_state()
    print(f"  Detected concept: {concept}")
    assert concept in ["storm", "transition"], f"Expected storm/transition, got {concept}"
    
    print("✅ Dynamic Modulation test passed")

def test_global_consensus_modulation():
    """Verify ConsensusHierarchyManager broadcasts global concept."""
    print("\n🧪 Testing Global Consensus Modulation...")
    
    manager = ConsensusHierarchyManager()
    
    # Create two levels
    level1 = FilteredHierarchicalLevel(1, 10, 10, filter_type='kalman')
    level1.add_layer(ReactiveLayer(10, 10))
    
    level2 = FilteredHierarchicalLevel(2, 10, 10, filter_type='kalman')
    level2.add_layer(ReactiveLayer(10, 10))
    
    manager.add_level(level1)
    manager.add_level(level2)
    
    # Run some steps to build history
    input_data = np.random.randn(10)
    for _ in range(10):
        manager.get_consensus_prediction(input_data)
        
    print(f"  Global concept: {manager.global_concept}")
    
    # Verify levels received it (or will use it)
    # The manager calls set_concept_state during get_consensus_prediction
    assert level1._current_concept == manager.global_concept
    assert level2._current_concept == manager.global_concept
    
    print("✅ Global Consensus Modulation test passed")

def run_verification():
    print("🚀 Verifying ORTHOS v5.1.0 Master Improvements")
    print("=" * 50)
    
    test_sr_kalman_stability()
    test_block_diagonal_kf()
    test_dynamic_modulation()
    test_global_consensus_modulation()
    
    print("\n🎉 All v5.1.0 improvements verified successfully!")

if __name__ == "__main__":
    run_verification()
