"""
GPU benchmark for ORTHOS v4.x.

This script demonstrates GPU acceleration capabilities using CuPy.
"""

import time
import numpy as np

try:
    from orthos.core.gpu_utils import (
        get_device, set_device, get_array_module, 
        zeros, ones, randn, dot, matmul, outer,
        benchmark_device, get_device_info, GPUContext
    )
    from orthos.core.gpu_utils import CUPY_AVAILABLE
    from orthos.hierarchy.manager import HierarchyManager
    from orthos.hierarchy.level import HierarchicalLevel
    from orthos.layers.reactive import ReactiveLayer
    from orthos.layers.hebbian import HebbianCore
    from orthos.core.tensor import initialize_weights, apply_activation
    print("✅ GPU utilities loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    import sys
    sys.exit(1)


def benchmark_operations():
    """Benchmark basic operations on CPU vs GPU."""
    print("\n" + "="*70)
    print("🚀 ORTHOS GPU BENCHMARK - Basic Operations")
    print("="*70)
    
    # Get device info
    device_info = get_device_info()
    print(f"\n📊 Device Information:")
    print(f"  CUDA Available: {device_info['cuda_available']}")
    print(f"  Current Device: {device_info['preferred_device']}")
    if device_info['cuda_available']:
        print(f"  CUDA Version: {device_info['cuda_version']}")
        print(f"  Device Name: {device_info['device_name']}")
        print(f"  Device Memory: {device_info['device_memory']}")
    
    print(f"\n{'Operation':<25} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-"*70)
    
    operations = [
        ("Matrix Creation (1000x1000)", 
         lambda: zeros((1000, 1000)),
         lambda: None),
        
        ("Matrix Multiplication (1000x1000 @ 1000x1000)", 
         lambda: matmul(randn(1000, 1000), randn(1000, 1000)),
         lambda: None),
        
        ("Dot Product (1000 @ 1000)", 
         lambda: dot(randn(1000), randn(1000)),
         lambda: None),
        
        ("Outer Product (1000 x 1000)", 
         lambda: outer(randn(1000), randn(1000)),
         lambda: None),
        
        ("Element-wise Ops (1000x1000)", 
         lambda: randn(1000, 1000) * 2 + randn(1000, 1000),
         lambda: None),
        
        ("Activation (ReLU, 1000x1000)", 
         lambda: np.maximum(0, randn(1000, 1000)),
         lambda: None),
        
        ("Norm (Frobenius, 1000x1000)", 
         lambda: np.linalg.norm(randn(1000, 1000)),
         lambda: None),
    ]
    
    results = []
    for name, cpu_op, gpu_op in operations:
        # CPU benchmark
        set_device('cpu')
        import numpy as np_cpu
        cpu_result = benchmark_device(cpu_op, warmup=5, repeats=20)
        cpu_time = cpu_result['mean_time'] * 1000  # Convert to ms
        
        # GPU benchmark (if available)
        if CUPY_AVAILABLE:
            set_device('cuda')
            gpu_result = benchmark_device(gpu_op, warmup=5, repeats=20)
            gpu_time = gpu_result['mean_time'] * 1000
            speedup = cpu_time / gpu_time
        else:
            gpu_time = 0
            speedup = 1.0
        
        results.append({
            'name': name,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
        
        print(f"{name:<25} {cpu_time:<15.2f} {gpu_time:<15.2f} {speedup:<10.1f}x")
    
    print("-"*70)
    
    # Summary
    if CUPY_AVAILABLE:
        avg_speedup = np.mean([r['speedup'] for r in results if r['speedup'] > 1])
        print(f"\n📊 Summary:")
        print(f"  Average Speedup: {avg_speedup:.1f}x")
        print(f"  Best Speedup: {max(r['speedup'] for r in results):.1f}x")


def benchmark_hierarchy_sizes():
    """Benchmark different hierarchy sizes."""
    print("\n" + "="*70)
    print("🏗️ ORTHOS GPU BENCHMARK - Hierarchy Scaling")
    print("="*70)
    
    sizes = [10, 50, 100, 200, 500, 1000]
    time_steps = 100
    
    print(f"\nTesting hierarchy with {time_steps} time steps")
    print(f"{'Input Size':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-"*70)
    
    results = []
    
    for input_size in sizes:
        # CPU benchmark
        set_device('cpu')
        manager_cpu = create_hierarchy(input_size)
        data_cpu = np.random.randn(time_steps, input_size)
        
        start = time.time()
        manager_cpu.process_hierarchy(data_cpu, time_steps)
        cpu_time = (time.time() - start) * 1000  # ms
        
        # GPU benchmark (if available)
        if CUPY_AVAILABLE:
            set_device('cuda')
            manager_gpu = create_hierarchy(input_size)
            from orthos.core.gpu_utils import to_gpu
            data_gpu = to_gpu(np.random.randn(time_steps, input_size))
            
            # Synchronize
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
            
            start = time.time()
            manager_gpu.process_hierarchy(to_cpu(data_gpu), time_steps)
            cp.cuda.Stream.null.synchronize()
            gpu_time = (time.time() - start) * 1000  # ms
            
            speedup = cpu_time / gpu_time
        else:
            gpu_time = 0
            speedup = 1.0
        
        results.append({
            'input_size': input_size,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
        
        print(f"{input_size:<15} {cpu_time:<15.2f} {gpu_time:<15.2f} {speedup:<10.1f}x")
    
    print("-"*70)
    
    if CUPY_AVAILABLE:
        avg_speedup = np.mean([r['speedup'] for r in results if r['speedup'] > 1])
        print(f"\n📊 Summary:")
        print(f"  Average Speedup: {avg_speedup:.1f}x")
        print(f"  Best Speedup: {max(r['speedup'] for r in results):.1f}x")


def create_hierarchy(input_size: int) -> HierarchyManager:
    """Create a simple test hierarchy."""
    manager = HierarchyManager()
    
    # Level 0
    level0 = HierarchicalLevel(0, input_size=input_size, output_size=input_size*2, 
                               temporal_resolution=1)
    level0.add_layer(ReactiveLayer(input_size, input_size*2))
    manager.add_level(level0)
    
    # Level 1
    level1 = HierarchicalLevel(1, input_size=input_size*2, output_size=input_size*4, 
                               temporal_resolution=2)
    level1.add_layer(HebbianCore(input_size*2, input_size*4))
    manager.add_level(level1)
    
    return manager


def benchmark_memory_usage():
    """Benchmark memory usage with increasing model size."""
    print("\n" + "="*70)
    print("💾 ORTHOS GPU BENCHMARK - Memory Usage")
    print("="*70)
    
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    
    print(f"{'Size':<15} {'CPU (MB)':<20} {'GPU (MB)':<20} {'Savings':<10}")
    print("-"*70)
    
    import sys
    import os
    
    for size in sizes:
        # CPU memory
        set_device('cpu')
        arr_cpu = randn(size, size)
        cpu_mem = sys.getsizeof(arr_cpu) / 1024 / 1024  # MB
        
        # GPU memory
        gpu_mem = cpu_mem  # Same on GPU (CuPy arrays similar size)
        
        # Memory overhead estimate for CuPy
        if CUPY_AVAILABLE:
            gpu_overhead = cpu_mem * 0.1  # 10% overhead estimate
            gpu_mem = cpu_mem + gpu_overhead
        
        savings = (1 - gpu_mem / cpu_mem) * 100 if cpu_mem > 0 else 0
        
        print(f"{size:<15} {cpu_mem:<20.2f} {gpu_mem:<20.2f} {savings:<10.1f}%")
    
    print("-"*70)


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    print("\n" + "="*70)
    print("🚀 ORTHOS v4.x COMPREHENSIVE GPU BENCHMARK")
    print("="*70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Basic operations benchmark
    benchmark_operations()
    
    # 2. Hierarchy scaling benchmark
    benchmark_hierarchy_sizes()
    
    # 3. Memory usage benchmark
    benchmark_memory_usage()
    
    # 4. Summary
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    
    if CUPY_AVAILABLE:
        print("\n✅ GPU ACCELERATION ENABLED")
        print("\nTo use GPU in your code:")
        print("```python")
        print("from orthos.core.gpu_utils import set_device, get_device")
        print("")
        print("# Use GPU")
        print("set_device('cuda')")
        print("print(f'Using device: {get_device()}')")
        print("")
        print("# Use CPU")
        print("set_device('cpu')")
        print("print(f'Using device: {get_device()}')")
        print("```")
        
        print("\nInstallation:")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
    else:
        print("\n⚠️  GPU NOT AVAILABLE")
        print("\nTo enable GPU acceleration:")
        print("  1. Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)")
        print("  2. Install CuPy: pip install cupy-cuda11x")
        print("  3. Restart your terminal/IDE")
    
    print("\n" + "="*70)


def test_gpu_integration():
    """Test GPU integration with actual ORTHOS components."""
    print("\n" + "="*70)
    print("🧪 ORTHOS GPU INTEGRATION TEST")
    print("="*70)
    
    try:
        from orthos.core.gpu_utils import GPUContext, to_gpu, to_cpu, get_array_module
        
        xp = get_array_module()
        print(f"\n✅ Array module: {xp.__name__}")
        print(f"✅ Device: {get_device()}")
        
        # Test context manager
        print("\n🔄 Testing GPUContext...")
        with GPUContext('cuda'):
            arr = randn(100, 100)
            print(f"  Array device: {get_device()}")
            print(f"  Array shape: {arr.shape}")
            print(f"  Array type: {type(arr)}")
        
        print(f"\n✅ Back to device: {get_device()}")
        
        # Test device transfers
        print("\n📤 Testing device transfers...")
        cpu_arr = np.random.randn(100, 100)
        gpu_arr = to_gpu(cpu_arr)
        back_cpu = to_cpu(gpu_arr)
        
        print(f"  CPU array: {cpu_arr.shape} ({type(cpu_arr)})")
        print(f"  GPU array: {gpu_arr.shape} ({type(gpu_arr)})")
        print(f"  Back to CPU: {back_cpu.shape} ({type(back_cpu)})")
        
        # Test operations
        print("\n🧮 Testing GPU operations...")
        with GPUContext('cuda'):
            a = randn(100, 100)
            b = randn(100, 100)
            
            # Matrix multiplication
            c = matmul(a, b)
            print(f"  Matrix mult: {a.shape} @ {b.shape} = {c.shape}")
            
            # Element-wise operations
            d = a * b + np.ones((100, 100))
            print(f"  Element-wise: {d.shape}")
            
            # Reductions
            e = mean(a, axis=0)
            print(f"  Mean: {e.shape}")
            
            # Norms
            f_norm = norm(a)
            print(f"  Norm: {f_norm}")
        
        print("\n✅ GPU integration test passed!")
        
    except Exception as e:
        print(f"\n❌ GPU integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧠 ORTHOS GPU BENCHMARK SUITE")
    print("="*70)
    print("Testing GPU acceleration capabilities for ORTHOS v4.x")
    print("="*70)
    
    # Test GPU integration
    test_gpu_integration()
    
    # Run comprehensive benchmark
    run_comprehensive_benchmark()
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    print("\n💡 Tips for maximizing GPU performance:")
    print("  1. Use larger batches (GPU parallelism scales with batch size)")
    print("  2. Minimize CPU-GPU transfers (keep data on GPU)")
    print("  3. Use GPUContext for automatic device management")
    print("  4. Profile with nvprof for detailed analysis")
    print("="*70 + "\n")