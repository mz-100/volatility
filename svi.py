from dataclasses import dataclass
import math
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d


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

    @classmethod
    def _calibrate_adc(cls, x, w, m, sigma):
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
        # The parameters (a, d, c) are found by the least squares method
        X = np.stack([np.ones_like(x), (x-m)/sigma, np.sqrt(((x-m)/sigma)**2+1)], axis=1)
        # p is the minimizer, s[0] is the sum of squared residuals
        p, s = np.linalg.lstsq(X, w, rcond=None)[:2]
        return p, s[0]   

    @classmethod
    def calibrate(cls, x, w, min_sigma=1e-4, max_sigma=10, method="differential_evolution"):
        """Calibrates the parameters of the model.

        This function finds the parameters which minimize the mean square error between the given
        total implied variance curve and the one produced by the model. The calibration is performed
        by the two-step minimization procedure by Zeliade systems.

        Args:
            x (array): Array of log-moneynesses
            w (array): Array of total implied variances.
            min_sigma (float): Left bound for the value of `sigma`.
            max_sigma (float): Right bound for the value of `sigma`.
            method (str): Method used for minimization. Must be a name of a global optimization 
                method from `scipy.optimize` module.

        Returns:
            An instance of the class with the calibrated parameters.
        """
        bounds = [(min(x), max(x)), (min_sigma, max_sigma)]
        res = optimize.__dict__[method](
            lambda q: cls._calibrate_adc(x, w, q[0], q[1])[1],  # q=(m, sigma)
            bounds=bounds)
        m, sigma = res.x
        a, d, c = cls._calibrate_adc(x, w, m, sigma)[0]
        rho = d/c
        b = c/sigma
        return cls(a, b, rho, m, sigma)


class ESSVI:
    """The Extended Surface SVI model.

    This class implements the eSSVI model by Corbetta, et al. (2019). Volatility smiles in the model
    are modelled by the equation
    ```
        w(x) = 0.5*(theta + rho*psi*x + sqrt((psi*x + theta*rho)^2 + theta^2*(1-rho)^2)),
    ```
    where `w` is the total implied variance, `x` is the log-moneyness (`log(strike/forward)`), and
    `theta`, `psi`, `rho` are maturity-dependent model parameters. The parameters `theta`, `psi` and 
    `rho*psi` are interpolated linearly between the given maturities.

    The main advantage of the model is that it can be easily calibrated free of static arbitrage.
    
    Methods:
        total_var: Total variance.
        __call__: Total variance.
        implied_vol: Implied volatility.
        local_vol: Local volatility.
        calibrate: Model calibration.    
    """

    def __init__(self, maturities, theta, psi, rho):
        """Initializes the model and builds linear interpolation of the parameters.

        Args:
            maturities (array): Array of maturities.
            theta (array): Array of `theta` parameters for each maturity.
            psi (array): Array of `psi` parameters for each maturity.
            rho (array): Array of `rho` parameters for each maturity.
        """
        # Add t=0 and then interpolate linearly
        maturities_ = np.concatenate(([0], maturities))
        theta_ = np.concatenate(([0], theta))
        psi_ = np.concatenate(([0], psi))
        rho_ = np.concatenate(([rho[0]], rho))
        self._theta = interp1d(maturities_, theta_, kind="linear")
        self._psi = interp1d(maturities_, psi_, kind="linear")
        self._rhopsi = interp1d(maturities_, rho_*psi_, kind="linear")

    @staticmethod
    def _essvi(x, theta, psi, rho):
        """The eSSVI function."""
        return 0.5*(theta + rho*psi*x + np.sqrt((psi*x + rho*theta)**2 + theta**2*(1 - rho**2)))
    
    def total_var(self, maturity, log_moneyness):
        """Computes the total implied variance.

        Args:
            maturity (float or array): Maturity.
            log_moneyness (float or array): Log-moneyness.
        
        Returns:
            float or array: Total implied variance for the given maturity and log-moneyness.
        """
        return self._essvi(log_moneyness, self._theta(maturity), self._psi(maturity), self._rhopsi(maturity)/self._psi(maturity))
    
    def __call__(self, maturity, log_moneyness):
        """Computes the total implied variance.

        Args:
            maturity (float or array): Maturity.
            log_moneyness (float or array): Log-moneyness.

        Returns:
            float or array: Total implied variance for the given maturity and log-moneyness.
        """
        return self.total_var(maturity, log_moneyness)
    
    def implied_vol(self, forward_price, maturity, strike):
        """Computes the implied volatility.

        Args:
            forward_price (float or array): Forward price.
            maturity (float or array): Maturity.
            strike (float or array): Strike price.

        Returns:
            float or array: Implied volatility for the given forward price, maturity and strike.
        """
        x = np.log(strike/forward_price)
        return np.sqrt(self.total_var(maturity, x)/maturity)
    
    def local_vol(self, initial_price, time, spot_price, discount_factor=1):
        """Computes the local volatility function of the underlying's spot price.

        Args:
            initial_price (float or array): Initial price of the underlying.
            time (float or array): Time variable as an argument of the local volatility function.
            spot_price (float or array): Spot price as an argument of the local volatility function.
            discount_factor (float or array): Discount factor for maturity equal to `time`.

        Returns:
            float or array: Local volatility function.
        """
        theta = self._theta(time)
        psi = self._psi(time)
        rho = self._rhopsi(time)/psi
        forward_price = initial_price/discount_factor
        
        # We use the formula from Gatheral's "Volatility Surface", p. 13, eq. 1.10.
        x = np.log(spot_price/forward_price)
        A = np.sqrt((psi*x + rho*theta)**2 + theta**2*(1 - rho**2))
        w = 0.5*(theta + rho*psi*x + A)

        dw_dx = 0.5*(rho*psi + (psi*x+theta*rho)*psi/A)
        d2w_dx2 = 0.5*psi**2*(1/A - (psi*x+theta*rho)**2/A**3)
        dw_dtheta = 0.5*(1 + (psi*x*rho+theta)/A)
        dw_dpsi = 0.5*x*(rho + (psi*x + theta*rho)/A)
        dw_drho = 0.5*psi*x*(1 + theta/A)

        # Compute the time derivatives of the parameters by small bumps (they are linearly
        # interpolated, so this should work OK, but not optimal)
        dt = 1e-6
        dtheta_dt = (self._theta(time+dt) - theta)/dt
        dpsi_dt = (self._psi(time+dt) - psi)/dt
        drho_dt = (self._rhopsi(time+dt)/self._psi(time+dt) - rho)/dt
        dw_dt = dw_dtheta*dtheta_dt + dw_dpsi*dpsi_dt + dw_drho*drho_dt

        v = dw_dt / (1 - x/w*dw_dx + 0.25*(-0.25-1/w+x**2/w**2) * (dw_dx)**2 + 0.5*d2w_dx2)

        return np.sqrt(v)
    
    @staticmethod
    def _calibrate_slice(x, w, theta_prev=None, psi_prev=None, rho_prev=None, rho_sample_size=20, rho_refinements=4):
        """Calibrates one slice of the eSSVI surface in an arbitrage-free way."""
        # See the algorithm in the paper by Corbetta, et al. (2019)

        # First, choose x* and theta* - the closest point to the ATMF total implied variance.
        if theta_prev is not None:
            w_prev = ESSVI._essvi(x, theta_prev, psi_prev, rho_prev)
        else:
            w_prev = 0
        x_star_index = np.argmin(np.abs(np.where(w_prev < w, x, math.inf)))
        x_star = x[x_star_index]
        theta_star = w[x_star_index]
        if theta_star < w[x_star_index]:
            raise RuntimeError("The next SVI slice is completely below the calibrated slice")
        
        # We need to find the optimal psi and rho. For each rho, we find psi by a bounded
        # minimization method. Then we search for the optimal rho by a simple grid search with
        # several level of refinements.
        rhos = np.linspace(-0.999, 0.999, rho_sample_size)
        opt_psi = np.empty_like(rhos)
        min_value = np.empty_like(rhos)

        for n in range(rho_refinements):
            for i, rho in enumerate(rhos):
                upper_bound = min(
                    -2*rho*x_star/(1+abs(rho)) + math.sqrt(4*(rho*x_star)**2/(1+abs(rho))**2 + 4*theta_star/(1+abs(rho))),
                    4/(1+abs(rho)),
                    theta_star/(rho*x_star) if rho*x_star > 0 else math.inf)
                if theta_prev is None:
                    lower_bound = 0
                else:
                    lower_bound = max(
                        0, 
                        (psi_prev - rho_prev*psi_prev)/(1-rho), (psi_prev+rho_prev*psi_prev)/(1+rho))
                    if rho*x_star < 0:
                        lower_bound = max(lower_bound, (theta_star-theta_prev)/(rho*x_star))
                    elif rho*x_star > 0:
                        upper_bound = min(upper_bound, (theta_star-theta_prev)/(rho*x_star))
                if lower_bound > upper_bound:
                    opt_psi[i] = math.nan
                    min_value[i] = math.inf
                else:
                    res = optimize.minimize_scalar(
                        lambda psi : np.linalg.norm(ESSVI._essvi(x, theta_star-rho*psi*x_star, psi, rho) - w), 
                        bounds = (lower_bound, upper_bound))
                    if res.success:
                        opt_psi[i] = res.x
                        min_value[i] = res.fun
                    else:
                        opt_psi[i] = math.nan
                        min_value[i] = math.inf

            opt_i = np.argmin(min_value)
            
            if n == rho_refinements-1:
                break

            if opt_i == 0:
                rhos = np.linspace(rhos[0], rhos[1], rho_sample_size)
            elif opt_i == rho_sample_size-1:
                rhos = np.linspace(rhos[-2], rhos[-1], rho_sample_size)
            else:
                rhos = np.linspace(rhos[opt_i-1], rhos[opt_i+1], rho_sample_size)
            
        if math.isnan(opt_psi[opt_i]):
            return math.nan, math.nan, math.nan
        else:
            return theta_star-rhos[opt_i]*opt_psi[opt_i]*x_star, opt_psi[opt_i], rhos[opt_i]
        
    @classmethod
    def calibrate(cls, maturity, log_moneyness, total_var, rho_sample_size=20, rho_refinements=4):
        """Calibrates the eSSVI model.

        This function calibrates the eSSVI model by calibrating each slice of the total implied
        variance surface separately avoiding the static arbitrage. The method is described in
        the paper by Corbetta, et al. (2019).

        Args:
            maturity (list or array): Array of maturities.
            log_moneyness (list of arrays): List of arrays of available log-moneynesses for each maturity.
            total_var (list of array): Total implied variances corresponding to the log-moneynesses.
            rho_sample_size (int): Number of points `rho` in one step of the grid search.
            rho_refinements (int): Number of refinements of the grid search.

        Returns:
            ESSVI: An instance of the class with the calibrated parameters.
        """
        theta = np.empty(len(maturity))
        psi = np.empty_like(theta)
        rho = np.empty_like(theta)
        
        # First slice - without arbitrage constraints
        theta[0], psi[0], rho[0] = cls._calibrate_slice(
            log_moneyness[0], total_var[0],
            rho_sample_size=rho_sample_size, rho_refinements=rho_refinements)
        
        # For next slices we do impose constraints
        for i in range(1, len(maturity)):
            theta[i], psi[i], rho[i] = ESSVI._calibrate_slice(
                log_moneyness[i], total_var[i], theta_prev=theta[i-1], psi_prev=psi[i-1], rho_prev=rho[i-1],
                rho_sample_size=rho_sample_size, rho_refinements=rho_refinements)

        return cls(maturity, theta, psi, rho)
