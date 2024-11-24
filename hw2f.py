from ast import Yield
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
from yield_curve import YieldCurve
import yield_curve

@dataclass
class HW2FParameters:
    """Parameters for Hull-White 2 Factor model"""
    lambda1: float  # Mean reversion speed for first factor
    lambda2: float  # Mean reversion speed for second factor
    sigma1: float  # Volatility of first factor
    sigma2: float  # Volatility of second factor
    rho: float     # Correlation between factors
    
class HullWhite2Factor:
    """Hull-White 2 Factor interest rate model calibrator"""
    
    def __init__(self, 
                 yield_curve: YieldCurve,
                 swaption_vols: np.ndarray,
                 option_maturities: np.ndarray,
                 swap_tenors: np.ndarray):
        """
        Initialize the HW2F calibrator
        
        Args:
            zero_rates: Array of zero rates
            rate_times: Time points for zero rates
            swaption_vols: Matrix of ATM swaption volatilities
            option_maturities: Option expiry times
            swap_tenors: Underlying swap tenors
        """
        self.yield_curve = yield_curve
        self.swaption_vols = swaption_vols
        self.option_maturities = option_maturities
        self.swap_tenors = swap_tenors
        
    def _discount_factor(self, t: float) -> float:
        """Calculate discount factor for time t"""
        if t == 0:
            return 1.0
        idx = np.searchsorted(self.yield_curve.get_time, t)
        if idx == 0:
            r = self.yield_curve.get_rate[0]
        else:
            r = float(np.interp(t, self.yield_curve.get_time, self.yield_curve.get_rate))
        return np.exp(-r * t)
    
    def _hw2f_swaption_vol(self, params: HW2FParameters, 
                          option_maturity: float, 
                          swap_tenor: float) -> float:
        """
        Calculate model swaption volatility under HW2F
        Using the analytical approximation for ATM swaptions
        """
        k1, k2 = params.lambda1, params.lambda2
        s1, s2 = params.sigma1, params.sigma2
        rho = params.rho
        
        T = option_maturity
        S = swap_tenor
        
        # Calculate integrated variance terms
        var1 = (s1**2 / (2 * k1**3)) * (1 - np.exp(-2*k1*T))
        var2 = (s2**2 / (2 * k2**3)) * (1 - np.exp(-2*k2*T))
        
        # Calculate cross term
        cross = (2 * rho * s1 * s2 / (k1 * k2 * (k1 + k2))) * \
                (1 - np.exp(-(k1 + k2)*T))
        
        # Total integrated variance
        total_var = var1 + var2 + cross
        
        return np.sqrt(total_var / T)
    
    def _objective_function(self, x: np.ndarray) -> float:
        """Objective function for calibration"""
        params = HW2FParameters(
            lambda1=abs(x[0]),
            lambda2=abs(x[1]),
            sigma1=abs(x[2]),
            sigma2=abs(x[3]),
            rho=max(min(x[4], 1.0), -1.0)
        )
        
        error = 0.0
        for i, T in enumerate(self.option_maturities):
            for j, S in enumerate(self.swap_tenors):
                model_vol = self._hw2f_swaption_vol(params, T, S)
                market_vol = self.swaption_vols[i, j]
                error += (model_vol - market_vol)**2
                
        return error
    
    def calibrate(self, 
                 initial_guess: Optional[HW2FParameters] = None) -> HW2FParameters:
        """
        Calibrate the HW2F model to market swaption volatilities
        
        Args:
            initial_guess: Initial parameter guess (optional)
            
        Returns:
            Calibrated HW2F parameters
        """
        if initial_guess is None:
            x0 = np.array([0.01, 0.01, 0.01, 0.01, 0.0])
        else:
            x0 = np.array([
                initial_guess.lambda1,
                initial_guess.lambda2,
                initial_guess.sigma1,
                initial_guess.sigma2,
                initial_guess.rho
            ])
            
        # Parameter bounds
        bounds = [
            (1e-4, 1.0),  # lambda1
            (1e-4, 1.0),  # lambda2
            (1e-4, 1.0),  # sigma1
            (1e-4, 1.0),  # sigma2
            (-1.0, 1.0)   # rho
        ]
        
        result = minimize(
            self._objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if not result.success:
            warnings.warn("Calibration may not have converged")
            
        return HW2FParameters(
            lambda1=abs(result.x[0]),
            lambda2=abs(result.x[1]),
            sigma1=abs(result.x[2]),
            sigma2=abs(result.x[3]),
            rho=max(min(result.x[4], 1.0), -1.0)
        )
    
if __name__ == "__main__":
    yield_curve = YieldCurve(np.array([0, 1, 2, 3, 4]), np.array([0.01, 0.013, 0.021, 0.025, 0.027]))
    swaption_vols = np.array([[0.01, 0.014, 0.02, 0.025, 0.03],
                             [0.02, 0.022, 0.025, 0.03, 0.035],
                             [0.02, 0.024, 0.028, 0.035, 0.04]])
    option_maturities = np.array([1, 2, 3])
    swap_tenors = np.array([1, 2, 3])
    hw2f = HullWhite2Factor(yield_curve, swaption_vols, option_maturities, swap_tenors)
    params = hw2f.calibrate()
    print(params)
    plt.plot(yield_curve.get_time, yield_curve.get_rate)
    plt.show()
