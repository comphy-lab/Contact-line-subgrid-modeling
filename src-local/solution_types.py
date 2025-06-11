"""
Shared data structures for GLE solver and continuation methods.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple

@dataclass
class SolutionResult:
    """Container for solution results."""
    success: bool
    solution: Optional[Any] = None
    theta_min: Optional[float] = None
    x0: Optional[float] = None
    s_range: Optional[Any] = None
    y_guess: Optional[Any] = None
    Ca: Optional[float] = None
    message: Optional[str] = None

class SolutionCache:
    """Cache for storing and interpolating solutions."""
    
    def __init__(self, max_size: int = 20):
        self.cache: Dict[float, SolutionResult] = {}
        self.max_size = max_size
    
    def add(self, result: SolutionResult) -> None:
        """Add a solution to the cache."""
        if result.Ca is not None:
            self.cache[result.Ca] = result
            # Keep only most recent entries
            if len(self.cache) > self.max_size:
                oldest_ca = min(self.cache.keys())
                del self.cache[oldest_ca]
    
    def get_nearest_two(self, Ca: float) -> Optional[Tuple[float, float, SolutionResult, SolutionResult]]:
        """Get two nearest Ca values for interpolation."""
        cas = sorted(self.cache.keys())
        if len(cas) < 2:
            return None
            
        if Ca <= cas[0]:
            return cas[0], cas[1], self.cache[cas[0]], self.cache[cas[1]]
        elif Ca >= cas[-1]:
            return cas[-2], cas[-1], self.cache[cas[-2]], self.cache[cas[-1]]
        else:
            # Find bracketing values
            for i in range(len(cas)-1):
                if cas[i] <= Ca <= cas[i+1]:
                    return cas[i], cas[i+1], self.cache[cas[i]], self.cache[cas[i+1]]
        return None 