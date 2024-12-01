# performance_measurement.py
import feature_extraction as fe
def measure_improvement(func, optimization, *args):
    # Measure the original performance
    original_features = fe.extract_features(func, *args)
    original_time = original_features['Execution Time']
    original_memory = original_features['Memory Usage (in MiB)']
    
    # Apply the optimization
    optimized_func = optimization(func)
    optimized_features = fe.extract_features(optimized_func, *args)
    optimized_time = optimized_features['Execution Time']
    optimized_memory = optimized_features['Memory Usage (in MiB)']
    
    # Calculate improvements
    time_improvement = original_time - optimized_time
    memory_improvement = original_memory - optimized_memory
    
    return {
        'Optimization': optimization.__name__,
        'Execution Time Improvement': time_improvement,
        'Memory Usage Improvement': memory_improvement
    }
