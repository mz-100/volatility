from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class NelsonSiegel:
    """The Nelson-Siegel yield curve model.

    The Nelson-Siegel model is a parametric model defined by the formula
    ```
        y(t) = beta2 + beta2 * (1 - exp(-lambda*t)) / (lambda*t) 
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

    # The next function is needed to avoid division by zero problem when zero is passed as maturity
    @staticmethod
    @np.vectorize
    def _beta2_loading(x):
        if np.isclose(x, 0):
            return 1
        else:
            return (1 - np.exp(-x)) / x

    def yield_curve(self, t):
        """Returns the yield curve at time `t`.

        Args:
            t (float or array): Time to maturity.

        Returns:
            float or array: Yield curve at time `t`.        
        """
        return (self.beta1 + (self.beta2 + self.beta3) * NelsonSiegel._beta2_loading(t*self.lambda_)
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
        # For a given decay factor, we find betas by linear regression
        def find_beta(lambda_):
            A = np.vstack([
                np.ones_like(t), NelsonSiegel._beta2_loading(t*lambda_), 
                NelsonSiegel._beta2_loading(t*lambda_) - np.exp(-lambda_*t)]).T 
            return np.linalg.lstsq(A, y, rcond=None)[0]
        
        # We minimize the mean square error of the yield curve to find the decay factor
        def loss(lambda_):
            beta = find_beta(lambda_)
            return np.sum((y - cls(*beta, lambda_).yield_curve(t))**2)
        
        res = minimize(loss, 1)
        if not res.success:
            raise ValueError("Calibration failed")
        
        return cls(*find_beta(res.x), res.x)
    
