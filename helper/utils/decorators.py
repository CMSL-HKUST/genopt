import time
from functools import wraps
from typing import Callable

def count_calls(func: Callable) -> Callable:
    """Decorator to count function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


def timer(func: Callable) -> Callable:
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.last_execution_time = end_time - start_time
        return result
    wrapper.last_execution_time = 0.0
    return wrapper
