# optimization.py
import numpy as np
import multiprocessing
from functools import lru_cache

def memoize(func, arr):
    """
    A simple memoization example for optimizing repeated function calls.
    """
    cache = {}
    
    def memoized_function(arr):
        if tuple(arr) in cache:
            return cache[tuple(arr)]  # Return cached result
        result = func(arr)  # Call the original function (not a string)
        cache[tuple(arr)] = result  # Cache the result
        return result
    
    return memoized_function

def loop_unrolling(arr):
    """
    A simple loop unrolling optimization example for sorting.
    """
    for i in range(0, len(arr) - 1, 2):  # Skip 2 steps for unrolling
        arr[i], arr[i + 1] = arr[i + 1], arr[i]  # Perform item swapping
    return arr  # Return the modified array


def tail_recursive_fib(n, accumulator=0):
    """Optimizes recursion to a tail recursive function."""
    if n == 0:
        return accumulator
    else:
        return tail_recursive_fib(n - 1, n + accumulator)

def vectorized_addition(arr1, arr2):
    """Vectorized addition using NumPy."""
    return np.add(arr1, arr2)

def vectorized_multiply(arr1, arr2):
    """Vectorized multiplication using NumPy."""
    return np.multiply(arr1, arr2)

def in_place_sort(arr):
    """Sort the array in-place to save memory."""
    arr.sort()
    return arr

def parallel_map(func, arr):
    """Apply function in parallel across multiple processors."""
    with multiprocessing.Pool() as pool:
        return pool.map(func, arr)

def parallel_sort(arr):
    """Sort the array in parallel."""
    return parallel_map(sorted, arr)

@lru_cache(maxsize=100)
def fibonacci(n):
    """Memoized Fibonacci with LRU cache."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def iterative_fibonacci(n):
    """Iterative version of Fibonacci to avoid recursion."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def binary_search(arr, target):
    """Binary search for sorted arrays."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
