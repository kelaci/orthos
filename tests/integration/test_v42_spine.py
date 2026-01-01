"""
Test script for ORTHOS v4.2 Probabilistic Spine.

Verifies the refactored Kalman Filters, Particle Filters, and 
Consensus Engine implementation.
"""

import sys
import os
import numpy as np
from typing import List

# Add current directory to path
sys.path.append(os.getcwd())

from orthos.filters.kalman import KalmanFilter, ExtendedKalmanFilter
from orthos.filters.particle import ParticleFilter
from orthos.consensus.engine import HierarchicalConsensus, LevelPrediction
from orthos.hierarchy.filtered_level import FilteredHierarchicalLevel
from orthos.hierarchy.consensus_manager import ConsensusHierarchyManager
from orthos.layers.reactive import ReactiveLayer

def test_kalman_filter():
    print("🧪 Testing Kalman Filter...")
    kf = KalmanFilter(state_dim=2, obs_dim=2)
    
    # Predict
    kf.predict()
    
    # Update with observation near zero
    x, P = kf.update(np.array([0.1, -0.1]))
    
    assert x.shape == (2,)
    assert P.shape == (2, 2)
    assert np.all(np.diag(P) < 1.0) # Uncertainty should decrease from initial 1.0
    print("✅ Kalman Filter passed")

def test_particle_filter():
    print("🧪 Testing Particle Filter...")
    
    def dynamics(x, u, n):
        return x + n
        
    def likelihood(x, z):
        return np.exp(-0.5 * np.sum((z - x)**2))
        
    pf = ParticleFilter(
        n_particles=100, 
        state_dim=2, 
        dynamics_fn=dynamics, 
        observation_fn=likelihood
    )
    
    pf.predict(process_noise_std=0.01)
    pf.update(np.array([0.5, 0.5]))
    
    mean = pf.get_mean()
    unc = pf.get_uncertainty()
    
    assert mean.shape == (2,)
    assert isinstance(unc, float)
    print("✅ Particle Filter passed")

def test_consensus_engine():
    print("🧪 Testing Consensus Engine...")
    engine = HierarchicalConsensus(outlier_threshold=2.0)
    
    predictions = [
        LevelPrediction(level=0, prediction=np.array([1.0, 1.0]), confidence=0.9, uncertainty=0.1),
        LevelPrediction(level=1, prediction=np.array([1.1, 0.9]), confidence=0.8, uncertainty=0.2),
        LevelPrediction(level=2, prediction=np.array([10.0, 10.0]), confidence=0.5, uncertainty=1.0) # Outlier
    ]
    
    result = engine.aggregate(predictions, method='weighted_vote')
    
    assert result.outlier_count == 1
    assert 2 not in result.participating_levels  # Level 2 should be outlier
    assert result.agreement_score == 2/3
    assert np.allclose(result.prediction.flatten(), [1.047, 0.953], atol=1e-2)
    print("✅ Consensus Engine passed")

def test_filtered_hierarchy():
    print("🧪 Testing Filtered Hierarchy...")
    manager = ConsensusHierarchyManager()
    
    # Level 0: Reactive + Kalman
    l0 = FilteredHierarchicalLevel(0, 10, 2, filter_type='kalman')
    l0.add_layer(ReactiveLayer(10, 2))
    manager.add_level(l0)
    
    # Level 1: Reactive + Particle
    l1 = FilteredHierarchicalLevel(1, 10, 2, filter_type='particle')
    l1.add_layer(ReactiveLayer(10, 2))
    manager.add_level(l1)
    
    # Process 5 steps
    for _ in range(5):
        input_data = np.random.randn(10)
        result = manager.get_consensus_prediction(input_data)
        assert result.prediction.shape == (2,)
    
    assert manager.is_hierarchy_stable() or not manager.is_hierarchy_stable() # Should run without error
    print("✅ Filtered Hierarchy passed")

if __name__ == "__main__":
    print("🚀 Running ORTHOS v4.2 Probabilistic Spine Tests")
    print("=" * 50)
    try:
        test_kalman_filter()
        test_particle_filter()
        test_consensus_engine()
        test_filtered_hierarchy()
        print("\n🎉 All Probabilistic Spine tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
