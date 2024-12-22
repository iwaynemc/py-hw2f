from typing import Any
import numpy as np

class YieldCurve():
    """
    Yield curve class representing interest rates at different time points
    
    This class stores and provides access to zero rates and their corresponding times
    """
    
    def __init__(self, t: np.ndarray, rt: np.ndarray) -> None:
        """
        Initialize the yield curve
        
        Args:
            t: Array of times
            rt: Array of corresponding interest rates
        """
        self._t = self._is_valid_attr(t)
        self._rt = self._is_valid_attr(rt)

    @property
    def get_time(self):
        """
        Get the time points of the yield curve
        
        Returns:
            np.ndarray: Array of time points
        """
        return self.__getattribute__("_t")

    @property
    def get_rate(self): 
        """
        Get the interest rates of the yield curve
        
        Returns:
            np.ndarray: Array of interest rates
        """
        return self.__getattribute__("_rt")

    @staticmethod
    def _is_valid_attr(attr: Any):
        """
        Validate that input attributes are numpy arrays or lists
        
        Args:
            attr: Attribute to validate
            
        Returns:
            The validated attribute
            
        Raises:
            AssertionError: If attribute is not a numpy array or list
        """
        assert isinstance(attr, (np.ndarray, list)), "Class Constructor takes only numpy arrays or list as arguments"
        return attr

    def plot_curve(self):
        """
        Plot the yield curve
        
        Plots interest rates against their corresponding time points
        """
        pass