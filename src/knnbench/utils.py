import time
import numpy as np
import random
import tracemalloc

def set_seed(seed: int = 42):
    """set the same seed everytime for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def peak_memory_bytes(fn):
    """a function to show the peak memory usage of a function."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)

def benchmark(fn, warmup: int = 3, repeat: int = 10):
    """a function to benchmark the execution time of a function."""
    for _ in range(warmup):
        fn()
        
    times = []
    for _ in range(repeat):
        start_time = time.perf_counter()
        fn()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        "median_s": float(np.median(times)),
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
    }