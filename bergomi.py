from dataclasses import dataclass
import math
import numpy as np

# Auxiliary functions needed in the IV approximation formula for the Bergomi model
def _H(x):
    return np.exp(-x) / x

def _I(x):
    return (1 - np.exp(-x)) / x

def _J(x):
    return (x - 1 + np.exp(-x)) / x ** 2

def _K(x):
    return (1 - np.exp(-x) - x * np.exp(-x)) / x ** 2


@dataclass
class Bergomi:
    """The 2-factor Bergomi model.

    The forward price and the forward variance in the 2-factor Bergomi model are defined by the SDEs
    ```
        d(F_t) = sqrt(xi^t_t)*F_t*d(W_t),      S_0 = s,
        d(xi^T_t) = omega*xi^T_t*alpha 
            *((1-theta)*exp(-k1*(T-t))dW^1_t + theta*exp(-k2*(T-t))dW^2_t),     T > 0, 0 <= t <= T,
    ```
    where `W_t`, `W^1_t` and `W^2_t` are standard Brownian motions with correlation coefficients
    `rho1 = cor(W, W^1)`, `rho2 = cor(W, W^2)`, `rho12 = cor(W^1, W^2)`.

    The 1-factor Bergomi model can be obtained as a particular case by putting `theta = 0` or `theta=1`. 
    
    Attributes:
        omega (float): Instantaneous volatility of `xi^t_t`.
        theta (float): Weight of the second factor. Must be between 0 and 1.
        k1 (float): Mean-reversion speed of the first factor.
        k2 (float): Mean-reversion speed of the second factor.
        rho1 (float): Correlation coefficient of `W_t` and `W^1_t`.
        rho2 (float): Correlation coefficient of `W_t` and `W^2_t`.
        rho12 (float): Correlation coefficient of `W^1_t` and `W^2_t`.
        xi0 (float): Initial forward variance (flat).

    Methods:
        simulate: Simulates random paths of the price process.
        approx_implied_vol: Analytical approximation of the implied volatility surface.
        approx_atm_iv: Analytical approximation of ATMF implied volatility.
        approx_atm_skew: Analytical approximation of ATMF volatility skew.
        approx_atm_curvature: Analytical approximation of ATMF volatility curvature.

    Notes:
        This implementation assumes a flat initial forward variance curve.
    """
    omega: float
    theta: float
    k1: float
    k2: float
    rho1: float
    rho2: float
    rho12: float
    xi0: float

    def approx_implied_vol(self, forward_price, maturity, strike):
        """Approximates the implied volatility by the 2nd order Bergomi-Guyon formula.

        The Bergomi-Guyon approximate formula is
        ```
            IV = IV_ATM + Skew_ATM * log(K/F_0) + Curvature_ATM * log(K/F0)^2.
        ```

        Args:
            forward_price (float or array): Current forward price.
            maturity (float or array): Time to maturity.
            strike (float or array): Strike.

        Returns:
            float or array: Approximate implied volatility.

        Notes:
            If the arguments are arrays, they are broadcasted if needed.
        """
        L = np.log(strike/forward_price)
        return (self.approx_atm_iv(maturity) + self.approx_atm_skew(maturity) * L +
                self.approx_atm_curvature(maturity) * L**2)

    def approx_atm_iv(self, maturity):
        """Approximates the ATMF implied volatility by the Bergomi-Guyon formula.

        Args:
            maturity (float or array): Time to maturity.

        Returns:
            float or array: Approximate ATMF implied volatility.
        """
        T = maturity
        nu = self.xi0 * T
        C_Xxi = self._spot_var_cov(T)
        C_xixi = self._var_var_cov(T)
        C_mu = self._c_mu(T)

        return (np.sqrt(nu/T) + C_Xxi / (4*np.sqrt(nu*T))*self.omega
                + 1/(32*nu**2.5 * np.sqrt(T)) * (12*C_Xxi**2 - C_xixi*nu*(nu+4) + 4*C_mu*nu*(nu-4))*self.omega**2)

    def approx_atm_skew(self, maturity):
        """Approximates the ATMF volatility skew by the Bergomi-Guyon formula.

        Args:
            maturity (float or array): Time to maturity.

        Returns:
            float or array: Approximate ATMF volatility skew.            
        """
        T = maturity
        nu = self.xi0*T
        C_Xxi = self._spot_var_cov(T)
        C_mu = self._c_mu(T)

        return (C_Xxi/(2*nu**1.5 * np.sqrt(T)) * self.omega + 1/(8*nu**2.5 * np.sqrt(T)) 
                * (4*C_mu*nu - 3*C_Xxi**2) * self.omega**2)

    def approx_atm_curvature(self, maturity):
        """Approximates the ATMF volatility curvature by the Bergomi-Guyon formula.

        Args:
            maturity (float or array): Time to maturity.

        Returns:
            float or array: Approximate ATMF volatility curvature.            
        """
        T = maturity
        nu = self.xi0*T
        C_Xxi = self._spot_var_cov(T)
        C_xixi = self._var_var_cov(T)
        C_mu = self._c_mu(T)
        return (1/(8*nu**3.5 * np.sqrt(T)) * (4*C_mu*nu + C_xixi*nu - 6*C_Xxi**2) * self.omega**2)

    def _alpha(self):
        return 1/math.sqrt((1-self.theta)**2 + self.theta**2 + 2*self.rho12*self.theta*(1-self.theta))
    
    def _spot_var_cov(self, T):
        w1x = (1 - self.theta) * self.rho1
        w1y = self.theta * self.rho2
        return self._alpha() * self.xi0**1.5 * T**2 * (w1x * _J(self.k1*T) + w1y*_J(self.k2*T))

    def _var_var_cov(self, T):
        w1x = (1 - self.theta) * self.rho1
        w1y = self.theta * self.rho2
        w2x = (1 - self.theta) * math.sqrt(1 - self.rho1**2)
        chi = (self.rho12 - self.rho1 * self.rho2) / math.sqrt((1 - self.rho1**2) * (1 - self.rho2**2))
        w2y = self.theta * chi * math.sqrt(1 - self.rho2**2)
        w3x = 0
        w3y = self.theta * math.sqrt((1 - chi**2) * (1 - self.rho2**2))

        lst = [(w1x, w1y), (w2x, w2y), (w3x, w3y)]
        w0 = sum([(x[0] / self.k1 + x[1] / self.k2)**2 for x in lst])
        wx = -2 * sum([(x[0] / self.k1 + x[1] / self.k2) * x[0] / self.k1 for x in lst])
        wy = -2 * sum([(x[0] / self.k1 + x[1] / self.k2) * x[1] / self.k2 for x in lst])
        wxx = sum([(x[0] / self.k1) ** 2 for x in lst])
        wyy = sum([(x[1] / self.k2) ** 2 for x in lst])
        wxy = 2 * sum([x[0] * x[1] / (self.k1*self.k2) for x in lst])
        return (self._alpha()**2 * self.xi0**2 * T
                  * (w0 + wx*_I(self.k1*T) + wy*_I(self.k2 * T) + wxx*_I(2*self.k1*T)
                     + wyy*_I(2*self.k2*T) + wxy*_I((self.k1 + self.k2)*T)))

    def _c_mu(self, T):
        w1x = (1 - self.theta) * self.rho1
        w1y = self.theta * self.rho2
        wx_ = w1x**2 / self.k1 + w1x * w1y / self.k2
        wy_ = w1y**2 / self.k2 + w1x * w1y / self.k1
        wx__ = w1x**2 / self.k1 + w1x * w1y / self.k2
        wy__ = w1y**2 / self.k2 + w1x * w1y / self.k1
        wxx__ = -w1x**2 / self.k1
        wyy__ = -w1y**2 / self.k2
        wxy__ = -w1x * w1y / self.k1 - w1x * w1y / self.k2

        C1mu = (-w1x**2 / self.k1*_K(self.k1*T) - w1y**2 / self.k2*_K(self.k2*T)
                + wx_*_J(self.k1*T) + wy_*_J(self.k2*T) - w1x*w1y / (self.k1 + self.k2)
                * (_H(self.k1*T) * _I(self.k1*T) + _H(self.k2*T) * _I(self.k2*T)
                   - 2*_H(2*self.k1*T)*_I(self.k2*T)
                   - 2*_H(2*self.k2*T)*_I(self.k1*T))
                )
        C2mu = (wx__ * _J(self.k1*T) + wy__ * _J(self.k2*T) + wxx__ * _J(2*self.k1*T)
                + wyy__ * _J(2*self.k2*T) + wxy__ * _J((self.k1 + self.k2)*T)
                )
        return self._alpha()**2 * self.xi0**2 * T**2 * (0.5*C1mu + C2mu)
    
    def simulate(self, s0, t, paths, steps, drift=0.0, rng=None):
        """Simulates random paths of the price process.

        This function simulates random paths of the price process in the 2-factor Bergomi model:
        ```
            dS_t = mu_t S_t dt + sqrt(xi^t_t) S_t d(W_t),   S_0 = s0,
        ```
        where `mu_t` is a drift process (e.g., 0 for the forward price, or the risk-free rate for
        the spot price), and `xi_t^T`, where `T >= t`, is the forward variance curve at time `T`.
        
        Args:
            t (float): Right end of the time interval.
            steps (int): Number of time steps, i.e. paths are sampled at `t_i = i*dt`, where
                `i = 0, ..., steps`, `dt = t/steps`.
            paths (int): Number of paths to simulate.
            drift (float or array or callable): Drift of the simulated process. If a scalar, the
                drift is assumed to be constant. If an array, then the drift at `t_i` is set equal
                to `drift[i]`; in this case the length of the array must be equal to `steps`. If a 
                callable the drift at `t_i` is computed as `drift(t_i)`.
            rng: A NumPy random number generator. If None, `numpy.random.default_rng` will be used.

        Returns:
            An array `X` of shape `(steps+1, paths)`, where `X[i, j]` is the value of `j`-th path
            of the simulated price process at point `t_i`.
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = t / steps

        if np.isscalar(drift):
            r = np.ones(steps) * drift
        elif isinstance(drift, np.ndarray):
            r = drift
        else: 
            r = np.array([drift(dt*i) for i in range(steps)])

        S = np.empty((steps+1, paths))
        xi = np.empty_like(S)           # xi_t^t
        X1 = np.empty_like(S)           # First Ornstein-Uhlenbeck factor
        X2 = np.empty_like(S)           # Second Ornstein-Uhlenbeck factor

        # Generate correlated normal variables (used in increments of the driving Brownian motions)
        # The direct method below is turns out to be faster than numpy.random.multivariate_normal or
        # multiplication by the square root of the covariance matrix
        Z = np.random.randn(3, steps, paths)
        Z1 = Z[0]
        Z2 = Z[0]*self.rho12 + math.sqrt(1 - self.rho12**2)*Z[1]
        l = np.sqrt((self.rho1**2 + self.rho2**2 - 2*self.rho12*self.rho1*self.rho2) / (1 - self.rho12**2))
        Z3 = ((self.rho1 - self.rho12*self.rho2) / (1 - self.rho12**2) * Z[0]
              + (self.rho1 - self.rho12*self.rho2) / (1 - self.rho12**2) * Z[1] + math.sqrt(1 - l**2)*Z[2])
        
        S[0] = s0
        xi[0] = self.xi0
        X1[0] = 0
        X2[0] = 0

        # Auxiliary constants needed below
        alpha = self._alpha()
        c11 = math.exp(-self.k1*dt)
        c12 = math.sqrt((1 - math.exp(-2*self.k1*dt)) / (2*self.k1))
        c21 = math.exp(-self.k2*dt)
        c22 = math.sqrt((1 - math.exp(-2*self.k2*dt)) / (2*self.k2))
        c30 = -0.5 * alpha**2 * self.omega**2
        c31 = (1 - self.theta)**2 / (2*self.k1)
        c32 = self.theta**2 / (2 *self.k2)
        c33 = 2*self.theta*(1 - self.theta)*self.rho12 / (self.k1 + self.k2)

        for i in range(steps):
            X1[i+1] = c11*X1[i]  + c12*Z1[i]
            X2[i+1] = c21*X2[i] + c22*Z2[i]
            xi[i+1] = self.xi0*np.exp(
                c30 * (c31*(1 - math.exp(-2*self.k1*(i + 1)*dt)) + 
                       c32*(1 - math.exp(-2*self.k2*(i + 1)*dt)) + 
                       c33*(1 - np.exp(-(self.k1 + self.k2)*(i + 1)*dt))
                      ) + self.omega * alpha * ((1 - self.theta)*X1[i+1] + self.theta*X2[i+1]))
            S[i+1] = S[i] * np.exp((r[i]-0.5*xi[i])*dt + np.sqrt(dt*xi[i])*Z3[i])

        return S


"""
REFERENCES
Bergomi, Lorenzo and Guyon, Julien, The Smile in Stochastic Volatility Models (December 2, 2011). 
    https://ssrn.com/abstract=1967470
Lorenzo Bergomi. Stochastic volatility modeling (2016). Chapman and Hall/CRC Financial Mathematics Series, 2016.
"""
