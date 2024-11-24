from typing import Any
import numpy as np

class YieldCurve():
    def __init__(self, t: np.ndarray, rt: np.ndarray) -> None:
        self._t = self._is_valid_attr(t)
        self._rt = self._is_valid_attr(rt)

    @property
    def get_time(self):
        return self.__getattribute__("_t")

    @property
    def get_rate(self): 
        return self.__getattribute__("_rt")

    @staticmethod
    def _is_valid_attr(attr: Any):
        assert isinstance(attr, (np.ndarray, list)), "Class Constructor takes only numpy arrays or list as arguments"
        return attr

    def plot_curve(self):
        pass