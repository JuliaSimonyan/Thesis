import time
import inspect
import ast
from memory_profiler import memory_usage

def count_function_calls(func):
    source = inspect.getsource(func)
    try:
        tree = ast.parse(source)
    except IndentationError:
        return 0  # Skip functions that have indentation issues

    class CallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.calls = 0

        def visit_Call(self, node):
            self.calls += 1
            self.generic_visit(node)

    visitor = CallVisitor()
    visitor.visit(tree)
    return visitor.calls

def estimate_complexity(func_name):
    if "sort" in func_name:
        return "O(n^2)"
    elif "search" in func_name:
        return "O(log n)"
    elif "fibonacci" in func_name or "factorial" in func_name:
        return "O(2^n)"
    elif "gcd" in func_name or "lcm" in func_name:
        return "O(log(min(a,b)))"
    elif "matrix_multiply" in func_name:
        return "O(n^3)"
    else:
        return "O(n)"  # Default assumption

def extract_features(func, *args, repeat=10000):
    try:
        start_time = time.time()
        mem_usage = memory_usage((func, args))

        # Run the function multiple times
        for _ in range(repeat):
            func(*args)

        execution_time = (time.time() - start_time) / repeat
        num_calls = count_function_calls(func)
        complexity = estimate_complexity(func.__name__)

        # Ensure valid data for all features
        if execution_time is None or num_calls is None or complexity is None:
            raise ValueError(f"Invalid feature values for {func.__name__}")

        return {
            "Function Name": func.__name__,
            "Execution Time": execution_time,
            "Memory Usage (in MiB)": max(mem_usage),  # Maximum memory usage during function execution
            "Number of Function Calls": num_calls,
            "Complexity": complexity
        }
    except Exception as e:
        print(f"Error extracting features for {func.__name__}: {e}")
        return {
            "Function Name": func.__name__,
            "Execution Time": None,
            "Memory Usage (in MiB)": None,
            "Number of Function Calls": None,
            "Complexity": None
        }
