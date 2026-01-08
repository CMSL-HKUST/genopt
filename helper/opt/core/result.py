from dataclasses import dataclass
from typing import Dict, Any

import numpy as onp

@dataclass
class OptimizationResult:
    """Standardized optimization result container"""
    x: onp.ndarray                   # Optimal solution
    fun: float                       # Optimal function value
    success: bool                    # Convergence status
    message: str                     # Termination message
    nfev: int = 0                    # Objective evaluations
    njev: int = 0                    # Gradient evaluations
    nit: int = 0                     # Iterations
    history: Dict[str, list] = None  # Optimization trajectory
    raw_result: Any = None           # Original optimizer output

    def __post_init__(self):
        if self.history is None:
            self.history = {'xs': [], 'funs': []}
            