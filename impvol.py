import math
import numpy as np
import numba
from numba import float64
from numba_stats.norm import ppf as _normppf


def implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag=1, discount_factor=1.0, eps=1.48e-08):
    """Computes Black implied volatility of European call or put options by Newton's method.

    Args:
        forward_price (float or array): Current forward price.
        maturity (float or array): Time to maturity.
        strike (float or array): Strike.
        price (float or array) : Option price.
        discount_factor (float or array): Discount factor.
        call_or_put_flag (float or array): 1 for a call option, -1 for a put option.
        eps (float): Absolute error of the root finding method.

    Returns:
        Implied volatility. If the method fails to converge, `NaN` is returned.
    """
    # The actual computation is done by a Numba function. We have to use this wrapper as Numba's
    # vectorization does not allow default values.
    return _implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag, discount_factor, eps)


@numba.njit(float64(float64))
def normpdf(x):
    """Standard normal density function."""
    return 1/math.sqrt(2*math.pi)*math.exp(-0.5*x*x)


@numba.njit(float64(float64))
def normcdf(x):
    """Standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


@numba.njit(float64(float64))
def normppf(x):
    """Inverse standard normal cumulative distribution function."""
    # The following is needed because norm.ppf of numba_stats accepts
    # only arrays
    x_ = np.array([x])
    return _normppf(x_, 0.0, 1.0)[0]


# The function which performs the actual computation
# We use Newton's method with selection of the initial guess by Jäckel (2006)
@numba.vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def _implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag, discount_factor, eps):
    # Normalization of parameters
    # With the new variables, we are looking for sigma = implied_vol*sqrt(T)
    x = math.log(forward_price/strike)
    p = option_price/(discount_factor*math.sqrt(forward_price*strike))

    # Explicit solution for ATM options
    if np.isclose(x, 0):
        return -2*normppf((1-p)/2)/math.sqrt(maturity)

    # Check price bounds
    theta = call_or_put_flag        # for brevity
    lower_bound = theta * (math.exp(x/2) - math.exp(-x/2)) if theta*x > 0 else 0.0
    upper_bound = math.exp(theta*x/2)
    if p <= lower_bound or p >= upper_bound:
        return math.nan

    # Initial guess for sigma - the infliction point of the Black function. See Jäckel (2006).
    sigma = math.sqrt(2*abs(x))

    for n in range(0, 100):  # Maximum 100 iterations in Newton's method
        f = theta*(math.exp(x/2)*normcdf(theta*(x/sigma+sigma/2)) -
                   math.exp(-x/2)*normcdf(theta*(x/sigma-sigma/2))) - p
        if abs(f) < eps:
            break
        fprime =  math.exp(x/2)*normpdf(x/sigma+sigma/2) 
        sigma = sigma - f/fprime
    
    if n < 100:
        return sigma/math.sqrt(maturity)
    else:
        return math.nan


"""
REFERENCES

Jäckel, P. (2006). By Implication. http://www.jaeckel.org/ByImplication.pdf
"""
