import time
from typing import Dict, Type

from ...utils.logger import LoggerManager

class Optimizer:
    
    """Main optimizer class"""
    
    _backend_registry: Dict[str, Type] = {}  # backend_name -> backend_class
    
    def __init__(self, method: str, backend: str, logger: LoggerManager):
        self.method = method  
        self.backend_name = backend.lower()
        self.logger = logger
        self.backend = self._create_backend()
    
    def _create_backend(self):
        """Create backend instance using registry."""
        if self.backend_name not in self._backend_registry:
            available = list(self._backend_registry.keys())
            raise ValueError(f"Unsupported backend: {self.backend_name}. Available: {available}")
        
        backend_class = self._backend_registry[self.backend_name]
        return backend_class(self.method, self.logger)
    
    @classmethod
    def register_backend(cls, backend_name: str):
        """Decorator to register backend classes."""
        def decorator(backend_class):
            cls._backend_registry[backend_name.lower()] = backend_class
            return backend_class
        return decorator
    
    @classmethod
    def available_backends(cls):
        """Get list of available backend names."""
        return list(cls._backend_registry.keys())
    
    def minimize(self, objective, x0, tol=None, bounds=None, constraints=None, options=None, callback=None):
        """Delegate minimization to backend."""
        
        self.logger.info('=' * 60)
        self.logger.info(f"Starting {self.method} optimization with {self.backend_name} backend")
        
        if tol is not None:
            self.logger.info(f"Using custom tolerance: {tol:.2e}")
        
        start_time = time.time()
        
        result = self.backend.minimize(
            objective=objective,
            x0=x0,
            tol=tol,
            bounds=bounds,
            constraints=constraints,
            options=options,
            user_callback=callback
        )
        
        elapsed = time.time() - start_time
        self.logger.info(f"Optimization completed in {elapsed:.3f} seconds")
        self.logger.info('=' * 60)
        
        return result
    
    def __repr__(self) -> str:
        return f"Optimizer(method='{self.method}', backend='{self.backend_name}')"