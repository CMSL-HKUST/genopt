from typing import Callable, Optional, Tuple, Any

import numpy as onp
import scipy.optimize

from .base import Backend
from ..core.optimizer import Optimizer
from ..core.exceptions import Convergence
from ..core.result import OptimizationResult

@Optimizer.register_backend('scipy')
class SciPyBackend(Backend):
    """SciPy optimization backend."""
    
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
        
        assert len(x0.shape) == 1,f"Expected shape (n,) for x0, got {x0.shape}"
        
        # Initialize result tracking using create_result
        initial_fun = float(objective(x0))
        result = self._create_result(
            x=x0.copy(),
            fun=initial_fun,
            success=False,
            message='Optimization started',
            nfev=0,
            njev=0,
            nit=0,
            history={'xs': [x0.copy()], 'funs': [initial_fun]}
        )
        
        self.logger.info(f"Initial objective: {initial_fun:.6e}")
        
        # Prepare objective function
        objective_with_grad = self.create_value_and_grad(objective)
        jac = True
        
        def scipy_callback(intermediate_result):
            """Callback function with custom tol stopping."""
            
            result.nit += 1
            current_x = onp.array(intermediate_result.x.copy())
            current_fun = float(intermediate_result.fun)
            
            # Update tracking
            result.history['xs'].append(current_x)
            result.history['funs'].append(current_fun)
            
            # Update evaluation counts
            result.nfev = objective_with_grad.calls
            result.njev = objective_with_grad.calls
            
            self.logger.info(f"Iter {result.nit:03d} Obj: {current_fun:.6e}")
            
            # Custom stopping condition
            if tol is not None and current_fun < tol:
                result.x = current_x
                result.fun = current_fun
                result.success = True
                result.message = f"Optimization finished: objective < tol ({current_fun:.2e} < {tol:.2e})"
                self.logger.info(f"Custom convergence achieved: {current_fun:.2e} < {tol:.2e}")
                raise Convergence()
            
            # User callback
            if user_callback:
                user_callback(result, current_x, current_fun)
        
        # Run optimization
        try:
            scipy_result = scipy.optimize.minimize(
                fun=objective_with_grad,
                x0=x0,
                method=self.method,
                jac=jac,
                bounds=bounds,
                constraints=constraints,
                callback=scipy_callback,
                options=options or {}
            )
            
            result.x = scipy_result.x
            result.fun = scipy_result.fun
            result.success = scipy_result.success
            result.message = scipy_result.message
            result.nfev = objective_with_grad.calls
            result.njev = objective_with_grad.calls
            result.raw_result = scipy_result
            
            if result.success:
                self.logger.info(f"Success: {result.message}")
            else:
                self.logger.warning(f"Fail to converge: {result.message}")
            
        except Convergence:
            
            self.logger.info("Optimization stopped by custom tolerance condition")
        
        return result