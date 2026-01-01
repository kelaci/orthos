"""
Unit tests for layers.
"""
import pytest
import numpy as np
from gaia.layers.reactive import ReactiveLayer
from gaia.layers.hebbian import HebbianCore
from gaia.layers.temporal import TemporalLayer
from gaia.core.exceptions import InputShapeError

def test_reactive_layer():
    """Test ReactiveLayer basic functionality."""
    layer = ReactiveLayer(10, 20, activation='relu')
    input_data = np.random.randn(5, 10)
    output = layer.forward(input_data)
    
    assert output.shape == (5, 20)
    assert np.all(output >= 0)  # ReLU check

def test_reactive_layer_errors():
    """Test error handling in ReactiveLayer."""
    layer = ReactiveLayer(10, 20)
    input_data = np.random.randn(5, 5) # Wrong shape
    
    with pytest.raises(InputShapeError):
        layer.forward(input_data)

def test_hebbian_core():
    """Test HebbianCore basic functionality."""
    layer = HebbianCore(20, 40, plasticity_rule='hebbian')
    input_data = np.random.randn(5, 20)
    output = layer.forward(input_data)
    
    assert output.shape == (5, 40)
    
    # Test update
    initial_weights = layer.get_weights().copy()
    layer.update(0.01)
    new_weights = layer.get_weights()
    
    assert not np.array_equal(initial_weights, new_weights)

def test_temporal_layer():
    """Test TemporalLayer basic functionality."""
    layer = TemporalLayer(40, 80, time_window=5)
    input_data = np.random.randn(5, 40)
    output = layer.forward(input_data, t=0)
    
    assert output.shape == (5, 80)
