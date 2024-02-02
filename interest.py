from dataclasses import dataclass
import numpy as np

@dataclass
class NelsonSiegel:
    """The Nelson-Siegel yield curve model.

    The Nelson-Siegel model is a parametric model for the yield curve defined by the formula
    ```
        y(t) = beta0 + beta1 * (1 - exp(-t / tau)) / (t / tau) + beta2 * ((1 - exp(-t / tau)) / (t / tau) - exp(-t / tau))
    ```
    where `t` is the time to maturity and `beta0`, `beta1`, `beta2`, and `tau` are the model parameters.

    Attributes:
        beta0 (float): Long-term factor (level).
        beta1 (float): Short-term factor (slope).
        beta2 (float): Medium-term factor (curvature).
        tau (float): Decay factor.

    Methods:
        yield_curve: Returns the yield curve at time `t`.
        discount_factor: Returns the discount factor at time `t`.
        calibrate: Calibrates the Nelson-Siegel model to the given yield curve.
    """
    beta0: float
    beta1: float
    beta2: float
    tau: float

    def yield_curve(self, t):
        """Returns the yield curve at time `t`.

        Args:
            t (float or array): Time to maturity.

        Returns:
            float or array: Yield curve at time `t`.        
        """
        return (self.beta0 + self.beta1 * (1 - np.exp(-t/self.tau)) / (t/self.tau) +
                self.beta2 * ((1 - np.exp(-t/self.tau)) / (t/ self.tau) - np.exp(-t/self.tau)))
    
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
        pass
    
