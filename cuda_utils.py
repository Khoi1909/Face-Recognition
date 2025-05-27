"""
CUDA Utilities for Face Recognition System
Handles CUDA initialization, testing, and performance monitoring
"""

import cv2
import torch
import numpy as np
import psutil
import logging
import time
import gc
from typing import Dict, Any

# Try to import GPUtil, but don't fail if it's not available
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None

logger = logging.getLogger(__name__)

class CUDAManager:
    """Unified CUDA management system"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.opencv_cuda_available = self._setup_opencv_cuda()
        self.device = self._get_optimal_device()
        
    def _setup_opencv_cuda(self) -> bool:
        """Setup CUDA for OpenCV operations"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info(f"OpenCV CUDA: {cv2.cuda.getCudaEnabledDeviceCount()} devices available")
                return True
            else:
                logger.warning("OpenCV CUDA: No devices available")
                return False
        except:
            logger.warning("OpenCV CUDA: Not available")
            return False
    
    def _get_optimal_device(self) -> torch.device:
        """Get the optimal device for processing"""
        if not self.cuda_available:
            logger.info("CUDA not available, using CPU")
            return torch.device('cpu')
        
        # Find GPU with most free memory
        best_device = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_allocated(i)
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_device = i
        
        device = torch.device(f'cuda:{best_device}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(best_device)} (Device {best_device})")
        return device
    
    def get_cuda_info(self) -> Dict[str, Any]:
        """Get comprehensive CUDA information"""
        info = {
            'cuda_available': self.cuda_available,
            'device_count': 0,
            'current_device': None,
            'devices': [],
            'pytorch_version': torch.__version__,
            'cuda_version': None
        }
        
        if self.cuda_available:
            info['device_count'] = torch.cuda.device_count()
            info['current_device'] = torch.cuda.current_device()
            info['cuda_version'] = torch.version.cuda
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'major': props.major,
                    'minor': props.minor,
                    'multi_processor_count': props.multi_processor_count
                }
                info['devices'].append(device_info)
        
        return info
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        info = {
            'system_memory': {
                'total': psutil.virtual_memory().total / (1024**3),
                'available': psutil.virtual_memory().available / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'gpu_memory': []
        }
        
        if self.cuda_available:
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                gpu_info = {
                    'device': i,
                    'name': torch.cuda.get_device_name(i),
                    'total': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'allocated': torch.cuda.memory_allocated(i) / (1024**3),
                    'cached': torch.cuda.memory_reserved(i) / (1024**3)
                }
                gpu_info['free'] = gpu_info['total'] - gpu_info['allocated']
                info['gpu_memory'].append(gpu_info)
        
        return info
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        cuda_info = self.get_cuda_info()
        memory_info = self.get_memory_info()
        
        info = {
            'pytorch_cuda': cuda_info['cuda_available'],
            'opencv_cuda': self.opencv_cuda_available,
            'device': str(self.device),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': memory_info['system_memory']['total'],
            'memory_available': memory_info['system_memory']['available'],
        }
        
        if cuda_info['cuda_available'] and cuda_info['devices']:
            device = cuda_info['devices'][0]
            info.update({
                'gpu_count': cuda_info['device_count'],
                'gpu_name': device['name'],
                'gpu_memory_total': device['total_memory_gb'],
                'gpu_memory_allocated': memory_info['gpu_memory'][0]['allocated'] if memory_info['gpu_memory'] else 0,
            })
        
        return info
    
    def optimize_memory(self):
        """Optimize memory usage"""
        gc.collect()
        if self.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def benchmark_device(self, iterations: int = 100) -> Dict[str, float]:
        """Benchmark device performance"""
        logger.info(f"Benchmarking {self.device}...")
        
        # Warm up
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            for _ in range(10):
                x = torch.randn(1000, 1000, device=self.device)
                y = torch.randn(1000, 1000, device=self.device)
                _ = torch.matmul(x, y)
            torch.cuda.synchronize()
        
        # Matrix multiplication benchmark
        start_time = time.time()
        for _ in range(iterations):
            x = torch.randn(1000, 1000, device=self.device)
            y = torch.randn(1000, 1000, device=self.device)
            _ = torch.matmul(x, y)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        matmul_time = (time.time() - start_time) / iterations
        
        # Memory transfer benchmark (only for GPU)
        transfer_time = 0
        if self.device.type == 'cuda':
            start_time = time.time()
            for _ in range(iterations):
                x_cpu = torch.randn(1000, 1000)
                x_gpu = x_cpu.to(self.device)
                _ = x_gpu.cpu()
            torch.cuda.synchronize()
            transfer_time = (time.time() - start_time) / iterations
        
        return {
            'matmul_time_ms': matmul_time * 1000,
            'transfer_time_ms': transfer_time * 1000,
            'ops_per_second': 1.0 / matmul_time if matmul_time > 0 else 0
        }
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor system resources"""
        resources = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
        }
        
        if self.cuda_available and GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    resources.update({
                        'gpu_utilization': gpu.load * 100,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_temperature': gpu.temperature
                    })
            except:
                pass
        
        return resources

def print_system_info():
    """Print comprehensive system information"""
    manager = CUDAManager()
    
    print("\n" + "="*60)
    print("🖥️  SYSTEM INFORMATION")
    print("="*60)
    
    # CPU Information
    print(f"CPU: {psutil.cpu_count()} cores @ {psutil.cpu_freq().current:.0f} MHz")
    
    # Memory Information
    memory = manager.get_memory_info()
    print(f"RAM: {memory['system_memory']['total']:.1f} GB total, {memory['system_memory']['available']:.1f} GB available")
    
    # CUDA Information
    cuda_info = manager.get_cuda_info()
    print(f"PyTorch: {cuda_info['pytorch_version']}")
    
    if cuda_info['cuda_available']:
        print(f"CUDA: {cuda_info['cuda_version']} ✅")
        for device in cuda_info['devices']:
            print(f"  GPU {device['id']}: {device['name']} ({device['total_memory_gb']:.1f} GB)")
    else:
        print("CUDA: Not available ❌")
    
    print(f"OpenCV CUDA: {'✅' if manager.opencv_cuda_available else '❌'}")
    print("="*60 + "\n")
    
    return manager

def run_performance_benchmark() -> Dict[str, Any]:
    """Run comprehensive performance benchmark"""
    print("🏃‍♂️ Running Performance Benchmark...")
    print("=" * 50)
    
    manager = CUDAManager()
    
    # CPU benchmark - OpenCV operations
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    iterations = 50
    
    start_time = time.time()
    for _ in range(iterations):
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        _ = cv2.GaussianBlur(gray, (15, 15), 0)
    cpu_time = (time.time() - start_time) / iterations
    
    results = {
        'cpu_time_ms': cpu_time * 1000,
        'device': str(manager.device)
    }
    
    # GPU benchmark
    if manager.cuda_available:
        gpu_benchmarks = manager.benchmark_device(iterations=iterations)
        results.update({
            'gpu_time_ms': gpu_benchmarks['matmul_time_ms'],
            'speedup': results['cpu_time_ms'] / gpu_benchmarks['matmul_time_ms'],
            'gpu_ops_per_second': gpu_benchmarks['ops_per_second'],
            'gpu_transfer_time_ms': gpu_benchmarks['transfer_time_ms']
        })
    else:
        results.update({
            'gpu_time_ms': None,
            'speedup': None,
            'gpu_ops_per_second': None,
            'gpu_transfer_time_ms': None
        })
    
    # Print results
    print(f"CPU Processing Time: {results['cpu_time_ms']:.2f} ms")
    if results['gpu_time_ms'] is not None:
        print(f"GPU Processing Time: {results['gpu_time_ms']:.2f} ms")
        print(f"GPU Speedup: {results['speedup']:.2f}x")
    else:
        print("GPU benchmark not available")
    
    return results

# Legacy function aliases for backwards compatibility
def get_cuda_info() -> Dict[str, Any]:
    return cuda_manager.get_cuda_info()

def get_optimal_device() -> torch.device:
    return cuda_manager.device

def get_memory_info() -> Dict[str, Any]:
    return cuda_manager.get_memory_info()

def optimize_memory():
    return cuda_manager.optimize_memory()

def benchmark_device(device: torch.device, iterations: int = 100) -> Dict[str, float]:
    return cuda_manager.benchmark_device(iterations)

# Create global instance
cuda_manager = CUDAManager()

if __name__ == "__main__":
    # Test the CUDA utilities
    manager = print_system_info()
    benchmarks = run_performance_benchmark()
    
    print(f"\n📊 Final Results:")
    print(f"  Device: {benchmarks['device']}")
    print(f"  CPU Time: {benchmarks['cpu_time_ms']:.2f} ms")
    if benchmarks['gpu_time_ms']:
        print(f"  GPU Time: {benchmarks['gpu_time_ms']:.2f} ms")
        print(f"  Speedup: {benchmarks['speedup']:.2f}x") 