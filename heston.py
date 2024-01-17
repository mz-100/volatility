from typing import Union
from numpy import float_
from numpy.typing import NDArray
from numba.types import CPointer, float64, intc  # type: ignore
from dataclasses import dataclass
import math
import cmath
import numpy as np
import numba                     # type: ignore
import scipy.optimize as opt     # type: ignore
import scipy.integrate as intg   # type: ignore
from scipy import LowLevelCallable
from . import implied_vol
from scipy.special import lambertw
from scipy.misc import derivative


# The following function computes the characteristic function of the log-price change `ln(F_t/F_0)`
# in the Heston model. It is written in Numba for speed and is called by other Numba functions. As
@numba.njit(cache=True)
def _heston_cf(u, T, V0, kappa, theta, sigma, rho):
    """Characteristic function of the log-price change `ln(F_t/F_0)` in the Heston model.

    Args:
        u (complex): Argument of the characteristic function. The value of `u` can be complex, i.e.
            we allow a generalized Fourier transform.
        T (float): Maturity.
        V0 (float): Initial variance.
        kappa (float): Mean-reversion speed.
        theta (float): Long-term variance.
        sigma (float): Volatility of volatility.
        rho (float): Correlation coefficient of the Brownian motions.

    Returns:
        float: The value of the characteristic function at `u`.

    Notes:
        The characteristic function is computed by the formula of Albrecher et al. (2007).
    """
    d = cmath.sqrt((1j*rho*sigma*u - kappa)**2 + sigma**2*(1j*u + u**2))
    g = ((1j*rho*sigma*u - kappa + d) / (1j*rho*sigma*u - kappa - d))
    C = (kappa*theta/sigma**2 * (
        (kappa - 1j*rho*sigma*u - d)*T - 2*cmath.log((1 - g*cmath.exp(-d*T))/(1-g))))
    D = ((kappa - 1j*rho*sigma*u - d)/sigma**2 * ((1-cmath.exp(-d*T)) / (1-g*cmath.exp(-d*T))))
    return cmath.exp(C + D*V0)


@numba.cfunc(float64(intc, CPointer(float64)), cache=True)
def _heston_integrand(n, x):
    """Integrand in Heston's formula.

    This function is used by SciPy's integration routine and is written in Numba for speed.

    Args:
        n: Size of the array `x`. Ignored (but required by the signature of `cfunc`).
        x: Array of parameters: x = (u, t, omega, v0, kappa, theta, sigma, rho), where
            - u: integration variable;
            - t: maturity;
            - omega: log-moneyness;
            - v0, kappa, theta, sigma, rho: model parameters.

    Returns:
        The value of the integrand at `u`.
    """
    u = x[0]
    t = x[1]
    omega = x[2]
    v0 = x[3]
    kappa = x[4]
    theta = x[5]
    sigma = x[6]
    rho = x[7]
    return (cmath.exp(1j*omega*u)/(1j*u) *
            (_heston_cf(u-1j, t, v0, kappa, theta, sigma, rho) -
                math.exp(-omega)*_heston_cf(u, t, v0, kappa, theta, sigma, rho))).real

@dataclass
class Heston:
    """The Heston model.

    The forward price in the Heston model is defined by the SDEs
        ```
        d(F_t) = sqrt(V_t)*F_t*d(W^1_t),
        d(V_t) = kappa*(theta - V_t)*dt + sigma*sqrt(V_t)*d(W^2_t),
        ```
    where `W^1_t` and `W^2_t` are standard Brownian motions with correlation coefficient `rho`, and
    `kappa>0', 'theta>0', 'sigma>0', '-1 < rho < 1` are the model parameters.

    Attributes:
        v0: Initial variance.
        kappa: Mean-reversion speed.
        theta: Long-term variance.
        sigma: Volatility of volatility.
        rho: Correlation coefficient of the Brownian motions.

    Methods:
        call_price: Price of a call option.
        put_price: Price of a put option.
        implied_vol: Implied volatility produced by the model.
        calibrate: Calibration of the model parameters.
        simulate: Simulates random paths of the price and variance processes.
    """
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def call_price(self, forward_price, maturity, strike, discount_factor=1.0,
                   epsabs=1.49e-08, epsrel=1.49e-08):
        """Computes call option prices.

        Args:
            forward_price (float): Current forward price.
            maturity (float): Time to maturity.
            strike (float or array): A single strike or an array of strikes.
            discount_factor: Discount factor.
            epsabs (float): Absolute error tolerance passed to SciPy's integration routine.
            epsrel (float): Relative error tolerance passed to SciPy's integration routine.

        Returns:
            A single option price or an array of prices of the same length as the array of strikes.
        """
        if np.isscalar(strike):
            return discount_factor*self._undiscounted_call_price(forward_price, maturity, strike, epsabs, epsrel)
        else:
            return np.array([discount_factor*self._undiscounted_call_price(forward_price, maturity, k, epsabs, epsrel) for k in strike])

    def put_price(self, forward_price, maturity, strike, discount_factor=1.0,
                   epsabs=1.49e-08, epsrel=1.49e-08):
        """Computes put option prices.

        The computation is reduced to call options by the call-put parity.

        Args:
            forward_price (float): Current forward price.
            maturity (float): Time to maturity.
            strike (float or array): A single strike or an array of strikes.
            discount_factor: Discount factor.
            epsabs (float): Absolute error tolerance passed to SciPy's integration routine.
            epsrel (float): Relative error tolerance passed to SciPy's integration routine.

        Returns:
            A single option price or an array of prices of the same length as the array of strikes.
        """
        return self.call_price(forward_price, maturity, strike, discount_factor, epsabs, epsrel) - discount_factor*(forward_price - strike)

    def _undiscounted_call_price(self, f0, t, k, epsabs=1.49e-08, epsrel=1.49e-08):
        """Computes the undiscounted price of a single call option by Heston's semi-closed formula.

        Args:
            f0 (float): Current forward price.
            t (float): Time to maturity.
            k (float): Strike.
            epsabs (float): Absolute error tolerance passed to SciPy's integration routine.
            epsrel (float): Relative error tolerance passed to SciPy's integration routine.

        Returns:
            The price of the option.
        """
        return f0 * (
            (1 - k/f0)/2 +
            1/math.pi * intg.quad(
                LowLevelCallable(_heston_integrand.ctypes),
                0, math.inf,
                args=(t, math.log(f0/k), self.v0, self.kappa, self.theta, self.sigma, self.rho),
                epsabs=epsabs,
                epsrel=epsrel)[0]
            )

    def implied_vol(self, forward_price, maturity, strike, epsabs=1.49e-08, epsrel=1.49e-08):
        """Computes the Black implied volatility produced by the model.

        Args:
            forward_price (float): Current forward price.
            maturity (float): Time to maturity.
            strike (float or array): A single strike or an array of strikes.
            epsabs (float): Absolute error tolerance passed to SciPy's integration routine.
            epsrel (float): Relative error tolerance passed to SciPy's integration routine.

        Returns:
            A single implied volatility or an array of implied volatilities of the same length as
            the array of strikes. 
        """
        return implied_vol(forward_price, maturity, strike,
                           self.call_price(forward_price, maturity, strike, epsabs, epsrel),
                           call_or_put_flag=1)
    
    @classmethod
    def calibrate(cls, forward_price, maturity, strikes, implied_vol, initial_guess=None,
                  min_method="SLSQP", return_minimize_result=False):
        """Calibrates the parameters of the Heston model.

        Args:
            forward_price (float): Initial forward price.
            maturity (float): Maturity of options. Only fixed maturity is supported.
            strikes (array): Array of strikes.
            implied_vol (array): Array of market implied volatilities.
            initial_guess: Initial guess for the parameters. Must be an instance of `Heston` class.
                If `None`, the default guess is used.
            min_method: Minimization method to be passed to `scipy.optimize.minimize`. The method
                must support bounds.
            return_minimize_result: If True, return also the minimization result of
                `scipy.optimize.minimize`.

        Returns:
            If `return_minimize_result` is True, returns a tuple `(cls, res)`, where `cls` is an
            instance of the class with the calibrated parameters and `res` in the optimization
            result returned by `scipy.optimize.minimize` (useful for debugging). Otherwise returns
            only `cls`.

        Notes:
            If `initial_guess` is not provided, the default guess is computed as follows:
                - `v0` and `theta` are set equal to the ATM variance (or variance closest to ATM);
                - `kappa` and `sigma` are 1;
                - `rho` is -0.5.
        """
        if initial_guess is None:
            v0 = implied_vol[np.abs(strikes-forward_price).argmin()]**2  # ATM variance
            x0 = (v0, 1.0, v0, 1.0, -0.5),  # (V0, kappa, theta, sigma, rho)
        else:
            x0 = (initial_guess.v0, initial_guess.kappa, initial_guess.theta, initial_guess.sigma,
                  initial_guess.rho)
        res = opt.minimize(
            fun=lambda p: np.linalg.norm(Heston(*p).implied_vol(forward_price, maturity, strikes) - implied_vol),
            x0=x0,
            method=min_method,
            bounds=[(0, math.inf), (0, math.inf), (0, math.inf), (0, math.inf),
                    (-1, 1)])
        model = cls(v0=res.x[0], kappa=res.x[1], theta=res.x[2], sigma=res.x[3], rho=res.x[4])
        if return_minimize_result:
            return model, res
        else:
            return model

    def simulate_euler(self, x0, t, steps, paths, drift=0.0, return_variance=False, rng=None):
        """Simulates paths using Euler's scheme.

        Euler's scheme is the fastest but least precise simulation method. This function simulates
        random paths of the price process
        ```
            dX_t = mu_t X_t dt + sqrt(V_t) X_t dW^1_t,    X_0 = x0, 
        ```
        where `mu_t` is a drift process (e.g., 0 for the forward price, or the risk-free rate for
        the spot price), and `V_t` is the variance process in the Heston model.

        Args:
            x0 (float): Initial value of the process.
            t: Right end of the time interval.
            steps: Number of time steps, i.e. paths are sampled at `t_i = i*dt`, where
                `i = 0, ..., steps`, `dt = t/steps`.
            paths: Number of paths to simulate.
            drift (float or array or callable): Drift of the simulated process. If a scalar, the
                drift is assumed to be constant. If an array, then the drift at `t_i` is set equal
                to `drift[i]`; in this case the length of the array must be equal to `steps`. If a 
                callable the drift at `t_i` is computed as `drift(t_i)`.
            return_variance : If True, returns both price and variance processes.
            rng: A NumPy random number generator. If None, `numpy.random.default_rng` will be used.

        Returns:
            If `return_variance` is False, returns an array `X` of shape `(steps+1, paths)`, where
            `X[i, j]` is the value of `j`-th path of the simulated price process at point `t_i`.

            If `return_variance` is True, returns a tuple `(X, V)`, where `X` and `V` are arrays
            of shape `(steps+1, paths)` representing the price and variance processes.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        dt = t/steps

        if np.isscalar(drift):
            r = np.ones(steps) * drift
        elif isinstance(drift, np.ndarray):
            r = drift
        else: 
            r = np.array([drift(dt*i) for i in range(steps)])

        Z = rng.normal(size=(2, steps, paths))
        V = np.empty(shape=(steps+1, paths))
        X = np.empty_like(V)
        V[0] = self.v0
        X[0] = x0

        for i in range(steps):
            Vplus = np.maximum(V[i], 0)
            V[i+1] = (
                V[i] + self.kappa*(self.theta-Vplus)*dt +
                self.sigma*np.sqrt(Vplus) * Z[0, i]*math.sqrt(dt))
            X[i+1] = X[i]*np.exp(
                r[i]*dt - 0.5*Vplus*dt +
                np.sqrt(Vplus)*(
                    self.rho*Z[0, i] +
                    math.sqrt(1-self.rho**2)*Z[1, i])*math.sqrt(dt))
        if return_variance:
            return X, V
        else:
            return X

    def simulate_qe(self, x0, t, steps, paths, drift=0.0, return_variance=False, rng=None):
        """Simulates paths using Andersen's QE scheme.

        The QE scheme is the most effective simulation method in terms of the trade-off between
        simulation error and speed. This function simulates random paths of the price process
        ```
            dX_t = mu_t X_t dt + sqrt(V_t) X_t dW^1_t,    X_0 = x0, 
        ```
        where `mu_t` is a drift process (e.g., 0 for the forward price, or the risk-free rate for
        the spot price), and `V_t` is the variance process in the Heston model.

        Args:
            x0 (float): Initial value of the process.
            t: Right end of the time interval.
            steps: Number of time steps, i.e. paths are sampled at `t_i = i*dt`, where
                `i = 0, ..., steps`, `dt = t/steps`.
            paths: Number of paths to simulate.
            drift (float or array or callable): Drift of the simulated process. If a scalar, the
                drift is assumed to be constant. If an array, then the drift at `t_i` is set equal
                to `drift[i]`; in this case the length of the array must be equal to `steps`. If a 
                callable the drift at `t_i` is computed as `drift(t_i)`.
            return_variance : If True, returns both price and variance processes.
            rng: A NumPy random number generator. If None, `numpy.random.default_rng` will be used.

        Returns:
            If `return_variance` is False, returns an array `X` of shape `(steps+1, paths)`, where
            `X[i, j]` is the value of `j`-th path of the simulated price process at point `t_i`.

            If `return_variance` is True, returns a tuple `(X, V)`, where `X` and `V` are arrays
            of shape `(steps+1, paths)` representing the price and variance processes.
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = t/steps

        if np.isscalar(drift):
            r = np.ones(steps) * drift
        elif isinstance(drift, np.ndarray):
            r = drift
        else: 
            r = np.array([drift(dt*i) for i in range(steps)])

        K0 = -self.rho*self.kappa*self.theta*dt/self.sigma
        K1 = 0.5*(self.kappa*self.rho/self.sigma-0.5)*dt - self.rho/self.sigma
        K2 = 0.5*(self.kappa*self.rho/self.sigma-0.5)*dt + self.rho/self.sigma
        K3 = 0.5*(1-self.rho**2)*dt
        # K4 from Andersen's paper is equal to K3, since we use gamma_1 = gamma_2 = 1/2
        C1 = math.exp(-self.kappa*dt)
        C2 = self.sigma**2*C1*(1-C1)/self.kappa
        C3 = 0.5*self.theta*self.sigma**2*(1-C1)**2/self.kappa
        Z = rng.normal(size=(2, steps, paths))
        U = rng.uniform(size=(steps, paths))
        V = np.empty(shape=(steps+1, paths))
        X = np.empty_like(V)
        V[0] = self.v0
        X[0] = x0

        for i in range(steps):
            m = V[i]*C1 + self.theta*(1-C1)
            s_sq = V[i]*C2 + C3
            psi = s_sq/m**2
            b_sq = np.where(
                psi < 2,
                2/psi - 1 + np.sqrt(np.maximum(4/psi**2 - 2/psi, 0)), 0)
            a = m/(1+b_sq)
            p = np.where(psi > 1, (psi-1)/(psi+1), 0)
            beta = (1-p)/m
            V[i+1] = np.where(psi < 1.5,
                              a*(np.sqrt(b_sq)+Z[0, i])**2,
                              np.where(U[i] < p, 0, np.log((1-p)/(1-U[i]))/beta))
            X[i+1] = X[i]*np.exp(K0 + K1*V[i] + K2*V[i+1] + np.sqrt(K3*(V[i] + V[i+1]))*Z[1, i])
        if return_variance:
            return X, V
        else:
            return X


"""
REFERENCES

Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). The little Heston trap. Wilmott
    Magazine, 2007(1), 83-92.
Andersen, L. (2008). Efficient simulation of the Heston stochastic volatility model.
    https://ssrn.com/abstract=946405
Heston, S. (1993). A closed-form solution for options with stochastic volatility with applications
    to bond and currency options. The Review of Financial Studies, 6(2), 327-343.
"""
