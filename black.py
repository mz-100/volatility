from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import norm


@dataclass
class Black:
    """The Black model.

    Attributes:
        sigma: Volatility.

    Methods:
        call_price: Price of a call option.
        put_price: Price of a put option.
        call_delta: Delta of a call option.
        put_delta: Delta of a put option.
        simulate: Simulates paths.
    """
    sigma: float

    def _d1(self, forward_price, maturity, strike):
        """Computes `d_1` from the Black formula."""
        return (np.log(forward_price/strike) + 0.5*maturity*self.sigma**2) / (self.sigma*math.sqrt(maturity))

    def _d2(self, forward_price, maturity, strike):
        """Computes `d_2` from the Black formula."""
        return self._d1(forward_price, maturity, strike) - self.sigma*math.sqrt(maturity)

    def call_price(self, forward_price, maturity, strike, discount_factor=1.0):
        """Computes the price of a call option.

        Args:
            forward_price: Forward price.
            maturity: Time to maturity.
            strike: Strike price.
            discount_factor: Discount factor.

        Returns:
            Call option price.
        """
        return discount_factor * (forward_price*norm.cdf(self._d1(forward_price, maturity, strike))
                                  - strike*norm.cdf(self._d2(forward_price, maturity, strike)))

    def put_price(self, forward_price, maturity, strike, discount_factor=1.0):
        """Computes the price of a put option.

        Args:
            forward_price: Forward price.
            maturity: Time to maturity.
            strike: Strike price.
            discount_factor: Discount factor.

        Returns:
            Put option price.
        """
        return discount_factor * (strike*norm.cdf(-self._d2(forward_price, maturity, strike))
                                  - forward_price*norm.cdf(-self._d1(forward_price, maturity, strike)))

    def call_delta(self, forward_price, maturity, strike, discount_factor=1.0):
        """Computes the price of a call option.

        Args:
            forward_price: Forward price.
            maturity: Time to maturity.
            strike: Strike price.
            discount_factor: Discount factor.

        Returns:
            Call option delta.
        """
        return discount_factor*norm.cdf(self._d1(forward_price, maturity, strike))

    def put_delta(self, forward_price, maturity, strike, discount_factor=1.0):
        """Computes the price of a call option.

        Args:
            forward_price: Forward price.
            maturity: Time to maturity.
            strike: Strike price.
            discount_factor: Discount factor.

        Returns:
            Call option delta.
        """
        return discount_factor*norm.cdf(self._d1(forward_price, maturity, strike) - 1)

    def simulate(self, x0, t, steps, paths, drift=0.0, rng=None):
        """Simulates paths of the price process.

        This function simulates the process
        ```
            dX_t = mu_t X_t dt + sigma X_t dW_t,
        ```
        where `mu_t` is a drift process (e.g., 0 for the forward price, or the risk-free rate for
        the spot price), and `sigma` is the model volatility.
        

        Args:
            t: Right end of the time interval.
            steps: Number of time steps, i.e. paths are sampled at `t_i = i*dt`, where
                `i = 0, ..., steps`, `dt = t/steps`.
            paths: Number of paths to simulate.
            drift (float or array or callable): Drift of the simulated process. If a scalar, the
                drift is assumed to be constant. If an array, then the drift at `t_i` is set equal
                to `drift[i]`; in this case the length of the array must be equal to `steps`. If a 
                callable the drift at `t_i` is computed as `drift(t_i)`.
            rng: A NumPy random number generator. If None, `numpy.random.default_rng` will be used.

        Returns:
            An array `X` of shape `(steps+1, paths)`, where `X[i, j]` is the value of the
            j-th path at point `t_i`.
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = t/steps

        if np.isscalar(drift):
            r = drift
        else:
            raise NotImplemented("Non-constant drift is not supported")

        # Increments of the log-price
        z = rng.normal(
            loc=(r - 0.5*self.sigma**2)*dt, scale=self.sigma*math.sqrt(dt), size=(steps, paths))
        logX = np.concatenate([np.zeros((1, paths)), np.cumsum(z, axis=0)])
        return x0*np.exp(logX)
