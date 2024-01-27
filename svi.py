from dataclasses import dataclass
import numpy as np
from scipy import optimize  


@dataclass
class SVI:
    """The SVI (Stochastic Volatility Inspired) model.

    The model directly represents a volatility smile by the function (the so-called "raw"
    parametrization):
    ```
        w(x) = a + b*(rho*(x-m) + sqrt((x-m)**2 + sigma**2))
    ```
    where
        `x = log(s/k)` is the log-moneyness, 
        `w = t*(iv**2)` is the total implied variance.
    and `a`, `b >= 0`, `-1 < rho < 1`, `m`, `sigma > 0` are model parameters. The maturity is
    assumed to be fixed and is not explicitly specified (except in the jump-wing parametrization,
    see the notes below).

    Attributes:
      a (float): Vertical shift.
      b (float): Angle between the left and right asymptotes.
      rho (float): Rotation.
      m (float): Horizontal shift.
      sigma (float): Curvature.

    Methods:
      to_natural: Returns the natural parameters.
      from_natural: Constructs the model from natural parameters.
      to_jumpwing: Returns the jump-wing parameters.
      from_jumpwing: Constructs the model from jump-wing parameters.
      calibrate: Calibrates parameters of the model.
      durrleman_function: Computes Durrleman's function.
      durrleman_condition: Find the minimum of Durrleman's function.

    Notes:
        Two other parametrizations are the natural parametrization and the jump-wing parametrization.
        There is a one-to-one correspondence between the raw, natural and jump-wings parameters
        (for the latter - with a fixed time to maturity).

        The natural parametrization is
        ```
            w(x) = delta + omega/2 * (1 + zeta*rho*(x-mu) + sqrt((zeta*(x-mu) + rho)**2 + 1 - rho**2).
        ```

        The jump-wings parametrization is defined by the following parameters (Gatheral, 2004).
            v: ATMF variance.
            psi: ATMF skew.
            p: Slope of the left wing.
            c: Slope of the right wing.
            v_tilde: Minimum implied variance.
            t: Time to maturity.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_natural(self):
        """Returns the natural parameters.

        Returns:
            Tuple `(delta, mu, rho, omega, zeta)`.

        Notes:
            See the class description for the natural parametrization formula.
        """
        omega = 2*self.b*self.sigma / np.sqrt(1-self.rho**2)
        delta = self.a - 0.5*omega*(1-self.rho**2)
        mu = self.m + self.rho*self.sigma / np.sqrt(1-self.rho**2)
        zeta = np.sqrt(1-self.rho**2) / self.sigma
        return delta, mu, self.rho, omega, zeta

    @classmethod
    def from_natural(cls, delta, mu, rho, omega, zeta):
        """Constructs a class instance from natural parameters.

        Args:
            delta, mu, rho, omega, zeta (float): Model parameters.

        Returns:
            An instance of the class.

        Notes:
            See the class description for the natural parametrization formula.
        """
        return cls(a=delta+0.5*omega*(1-rho**2),
                   b=0.5*omega*zeta,
                   rho=rho,
                   m=mu-rho/zeta,
                   sigma=np.sqrt(1-rho**2)/zeta)

    def jumpwing(self, t=1):
        """Returns the jump-wings parameters.

        Args:
          t (float): Time to maturity.

        Returns:
          Tuple  `(v, psi, p, c, v_tilde)`.
        """
        w = (self.a +
             self.b*(-self.rho*self.m + np.sqrt(self.m**2+self.sigma**2)))
        v = w/t
        psi = self.b/np.sqrt(w)/2 * (
            -self.m/np.sqrt(self.m**2+self.sigma**2) + self.rho)
        p = self.b*(1-self.rho)/np.sqrt(w)
        c = self.b*(1+self.rho)/np.sqrt(w)
        v_tilde = (self.a + self.b*self.sigma*np.sqrt(1-self.rho**2)) / t
        return v, psi, p, c, v_tilde

    @classmethod
    def from_jumpwing(cls, v, psi, p, c, v_tilde, t=1):
        """Constructs a class instance from jump-wing parameters.
        
        Args:
            v, psi, p, c, v_tilde (float): Model parameters.
            t (float): Time to maturity.

        Returns:
            An instance of the class.
        """
        w = v*t
        b = 0.5*np.sqrt(w)*(c+p)
        rho = 1 - 2*p/(c+p)
        beta = rho - 2*psi*np.sqrt(w)/b
        if np.abs(beta) > 1:
            raise ValueError(
                f"Smile is not convex: beta={beta}, but must be in [-1, 1].")
        elif beta == 0:
            m = 0
            sigma = (v-v_tilde)*t / (b*(1-np.sqrt(1-rho**2)))
        else:
            alpha = np.sign(beta) * np.sqrt(1/beta**2 - 1)
            m = (v-v_tilde)*t / (b*(-rho+np.sign(alpha)*np.sqrt(1+alpha**2) -
                                    alpha*np.sqrt(1-rho**2)))
            sigma = alpha*m
        a = v_tilde*t - b*sigma*np.sqrt(1-rho**2)
        return cls(a, b, rho, m, sigma)

    def __call__(self, x):
        """Returns the total implied variance `w(x)`.

        Args:
            x (float or array): Log-moneyness.

        Returns:
            Total implied variance.       
        """
        return self.a + self.b*(self.rho*(x-self.m) +
                                np.sqrt((x-self.m)**2 + self.sigma**2))

    def durrleman_function(self, x):
        """Durrleman's function for verifying the convexity of the price surface.

        Args:
            x (float or array): Log-moneyness.

        Returns:
            The value of Durrleman's function.
        """
        # Total variance and its two derivatives
        w = self.__call__(x)
        wp = self.b*(self.rho + (x-self.m) / np.sqrt(
            (x-self.m)**2 + self.sigma**2))
        wpp = self.b*(1/np.sqrt((x-self.m)**2 + self.sigma**2) -
                      (x-self.m)**2/((x-self.m)**2 + self.sigma**2)**1.5)
        return (1-0.5*x*wp/w)**2 - 0.25*wp**2*(1/w+0.25) + 0.5*wpp

    def durrleman_condition(self, min_x=None, max_x=None):
        """Checks Durrleman's condition.

        This function numerically finds the global minimum of Durrleman's function. If the minimum
        is negative, then Durrleman's condition fails, so the model has static arbitrage).

        Args:
            min_x (float): Left end of the log-moneyness interval where Durrleman's function is
                minimized. `None` corresponds to the the minus infinity.
            max_x (float): Right end of the log-moneyness interval where Durrleman's function is
                minimized. `None` corresponds to the the plus infinity.

        Returns:
            Tuple `(min, x)` where `min` is the minimum of Durrleman's function, and `x` is the point
            where it is attained.
        """
        res = optimize.dual_annealing(
            lambda x: self.durrleman_function(x[0]), x0=[0],
            bounds=[(min_x, max_x)],
            minimizer_kwargs={
                "method": "BFGS",
                "jac": (lambda x:
                        self.b*(self.rho + (x[0]-self.m) /
                                np.sqrt((x[0]-self.m)**2+self.sigma**2)))
            })
        return res.fun, res.x[0]

    @staticmethod
    def _calibrate_adc(x, w, m, sigma):
        """Calibrates the raw parameters `a, d, c` given `m, sigma`.

        This is an auxiliary function used in the two-step calibration procedure. It finds optimal
        values of the parameters `a`, `d`, `c` for given `m` and `sigma`.

        Args:
          x (array): Log-moneyness.
          w (array): Total implied variances.
          m (float): Parameter `m` of the model.
          sigma (float): Parameter `sigma` of the model.

        Returns:
            Tuple `((a, d, c), f)` where `a, d, c` are the calibrated parameters and `f` is the
            minimum of the objective function.
        """
        # Objective function; p = (a, d, c)
        def f(p):
            return 0.5*np.linalg.norm(
                p[0] + p[1]*(x-m)/sigma + p[2]*np.sqrt(((x-m)/sigma)**2+1) -
                w)**2

        # Gradient of the objective function
        def fprime(p):
            v1 = (x-m)/sigma
            v2 = np.sqrt(((x-m)/sigma)**2+1)
            v = p[0] + p[1]*v1 + p[2]*v2 - w
            return (np.sum(v), np.dot(v1, v), np.dot(v2, v))

        res = optimize.minimize(
            f,
            x0=(np.max(w)/2, 0, 2*sigma),
            method="SLSQP",
            jac=fprime,
            bounds=[(None, np.max(w)), (None, None), (0, 4*sigma)],
            constraints=[
                {'type': 'ineq',
                 'fun': lambda p: p[2]-p[1],
                 'jac': lambda _: (0, -1, 1)},
                {'type': 'ineq',
                 'fun': lambda p: p[2]+p[1],
                 'jac': lambda _: (0, 1, 1)},
                {'type': 'ineq',
                 'fun': lambda p: 4*sigma - p[2]-p[1],
                 'jac': lambda _: (0, -1, -1)},
                {'type': 'ineq',
                 'fun': lambda p: p[1]+4*sigma-p[2],
                 'jac': lambda _: (0, 1, -1)}])
        return res.x, res.fun

    @classmethod
    def calibrate(cls, x, w, min_sigma=1e-4, max_sigma=10, return_minimize_result=False):
        """Calibrates the parameters of the model.

        This function finds the parameters which minimize the mean square error between the given
        total implied variance curve and the one produced by the model.

        Args:
            x (array): Array of log-moneynesses
            w (array): Array of total implied variances.
            min_sigma (float): Left bound for the value of `sigma`.
            max_sigma (float): Right bound for the value of `sigma`.
            return_minimize_result (bool): If True, returns also the minimization result of SciPy's
                dual annealing algorithm.

        Returns:
            If `return_minimize_result` is True, returns a tuple `(cls, res)`, where `cls` is an
            instance of the class and `res` in the optimization result returned by
            `scipy.optimize.dual_annealing`. Otherwise returns only `cls`.

        Notes:
            The algorithm used is the two-step minimization procedure by Zeliade systems.
        """
        res = optimize.dual_annealing(
            lambda q: cls._calibrate_adc(x, w, q[0], q[1])[1],  # q=(m, sigma)
            bounds=[(min(x), max(x)), (min_sigma, max_sigma)],
            minimizer_kwargs={"method": "nelder-mead"})
        m, sigma = res.x
        a, d, c = cls._calibrate_adc(x, w, m, sigma)[0]
        rho = d/c
        b = c/sigma
        ret = cls(a, b, rho, m, sigma)
        if return_minimize_result:
            return ret, res
        else:
            return ret
