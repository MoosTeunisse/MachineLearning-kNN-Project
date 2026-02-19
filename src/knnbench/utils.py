import time
import numpy as np
import random

def set_seed(seed: int = 42):
    """set the same seed everytime for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

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