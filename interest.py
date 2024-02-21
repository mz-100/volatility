from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize

# This function is needed to avoid division by zero problem when zero is passed as maturity
@np.vectorize
def _nelson_siegel_exp(x):
    if math.isclose(x, 0):
        return 1.0
    else:
        return (1 - math.exp(-x)) / x

@dataclass
class NelsonSiegel:
    """The Nelson-Siegel yield curve model.

    The Nelson-Siegel model is a parametric model defined by the formula
    ```
        y(t) = beta1 + beta2 * (1 - exp(-lambda*t)) / (lambda*t) 
            + beta3 * ((1 - exp(-lambda*t)) / (lambda*t) - exp(-lambda*t)) 
    ```
    where `t` is the time to maturity, `y(t)` is the yield, and `beta1`, `beta2`, `beta3`, `lambda`
    are parameters. The above form is from Diebold and Li (2006).

    Attributes:
        beta1 (float): Long-term factor (level).
        beta2 (float): Short-term factor (slope).
        beta3 (float): Medium-term factor (curvature).
        lambda_ (float): Decay factor.

    Methods:
        yield_curve: Returns the yield curve at time `t`.
        discount_factor: Returns the discount factor at time `t`.
        calibrate: Calibrates the Nelson-Siegel model to the given yield curve.
    """
    beta1: float
    beta2: float
    beta3: float
    lambda_: float

    def yield_curve(self, t):
        """Returns the yield curve at time `t`.

        Args:
            t (float or array): Time to maturity.

        Returns:
            float or array: Yield curve at time `t`.        
        """
        return (self.beta1 + (self.beta2 + self.beta3) * _nelson_siegel_exp(t*self.lambda_)
                - self.beta3 * np.exp(-self.lambda_*t))
    
    def __call__(self, t):
        """Returns the yield curve at time `t`.

        Args:
            t (float or array): Time to maturity.

        Returns:
            float or array: Yield curve at time `t`.        
        """
        return self.yield_curve(t)
    
    def discount_factor(self, t):
        """Returns the discount factor at time `t`.

        The discount factor is calculated as `exp(-t * yield_curve(t))`.

        Args:
            t (float or array): Time to maturity.

        Returns:
            float or array: Discount factor at time `t`.        
        """
        return np.exp(-t*self.yield_curve(t))
    
    @classmethod
    def calibrate(cls, t, y):
        """Calibrates the Nelson-Siegel model to the given yield curve.

        Args:
            t (array): Maturities.
            y (array): Yield rates.
        
        Returns:
            NelsonSiegel: Calibrated Nelson-Siegel model.        
        """
        # For a given decay factor, find betas by OLS
        # This function returns a tuple with the first element being the betas and the second
        # element being the squared residuals; we ignore other elements 
        def beta_ols(lambda_):
            A = np.vstack([
                np.ones_like(t), _nelson_siegel_exp(t*lambda_), 
                _nelson_siegel_exp(t*lambda_) - np.exp(-lambda_*t)]).T 
            return np.linalg.lstsq(A, y, rcond=None)
        
        # Initial guess maximizes the loading on the medium-term factor at 30 months
        # (from Diebold and Li, 2006)        
        res = minimize(lambda lambda_: beta_ols(lambda_)[1], 0.0609) 

        if not res.success:
            raise ValueError("Calibration failed")
        return cls(*beta_ols(res.x[0])[0], res.x[0])
    
