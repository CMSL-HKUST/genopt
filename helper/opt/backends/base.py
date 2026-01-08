from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Any

import jax
import numpy as onp

from ..core.result import OptimizationResult
from ...utils.decorators import count_calls
from ...utils.logger import LoggerManager

class Backend(ABC):
    
    """Base class for all optimization backends."""
    
    def __init__(self, method: str, logger: LoggerManager):
        self.method = method
        self.logger = logger
    
    @abstractmethod
    def minimize(
        self,
        objective: Callable[[onp.ndarray], float],
        x0: onp.ndarray,
        tol: Optional[float] = None,
        bounds: Tuple[Tuple[float, float], ...] = None,
        constraints: Optional[list] = None,
        options: Optional[dict] = None,
        user_callback: Optional[Callable[..., Any]] = None
        ) -> OptimizationResult:
        """Implement optimization logic."""
        pass
    
    def create_value_and_grad(self, objective: Callable) -> Callable:
        """Create function that returns value and gradient."""
        @count_calls
        def value_and_grad_fn(x):
            val, grad = jax.value_and_grad(objective)(x)
            return float(val), onp.array(grad)
        return value_and_grad_fn
    
    def _create_result(self, x, fun, success, message, nfev=0, njev=0, nit=0, history=None, raw_result=None):
        """Helper method to create standardized OptimizationResult."""
        return OptimizationResult(
            x=x,
            fun=fun,
            success=success,
            message=message,
            nfev=nfev,
            njev=njev,
            nit=nit,
            history=history,
            raw_result=raw_result
        )
