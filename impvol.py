import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton


def implied_vol(forward_price, maturity, strike, option_price, call_or_put_flag=1, discount_factor=1.0, check_price_bounds=True):
    """Computes Black implied volatility of European call or put options by Newton's method.

    This function works with a single maturity and a single strike or an array of strikes.

    Args:
        forward_price (float): Current forward price.
        maturity (float): Time to maturity of options.
        strike (float | ndarray): A single strike or an array of strikes.
        price (float | ndarray) : Option prices. Must be of the same shape as `K`.
        discount_factor (float): Discount factor.
        call_or_put_flag (float | ndarray): 1 for call options, -1 for put options. Must be scalar
            or of the same shape as `K`.
        check_price_bounds (bool): If `True`, checks that the price is within the bounds implied by
            the Black formula; when not, NaN is returned. If `False`, no checks are done.

    Returns:
        float | ndarray: A single implied volatility or an array of implied volatilities. If the
            method fails to converge, `NaN` is returned for the corresponding options.
    """
    theta = call_or_put_flag        # for brevity

    # Normalization of parameters. See Jäckel (2006).
    # With the new variables, we are looking for sigma = implied_vol*sqrt(T)
    x = np.log(forward_price/strike)
    p = option_price/(discount_factor*np.sqrt(forward_price*strike))

    # Check that the price is within the bounds implied by the Black formula
    if check_price_bounds:
        if np.isscalar(strike):
            lower_bound = np.heaviside(theta*x, 0.5) * theta * (math.exp(x/2) - math.exp(-x/2))
            upper_bound = math.exp(theta*x/2)
            if p <= lower_bound or p >= upper_bound:
                return np.NaN
        else:
            # For vector-valued arguments, we filter out invalid prices and call `implied_vol`
            # for valid prices only. This is much faster than calling `implied_vol` for each
            # option separately.
            lower_bound = np.heaviside(theta*x, 0.5) * theta * (np.exp(x/2) - np.exp(-x/2))
            upper_bound = np.exp(theta*x/2)
            invalid = (p <= lower_bound) | (p >= upper_bound)
            # If there are invalid prices, we call `implied_vol` for valid prices only
            # If all prices are valid, we'll proceed to Newton's method
            if np.any(invalid):
                sigma = np.empty_like(strike)
                sigma[~invalid] = implied_vol(
                    forward_price=forward_price,
                    maturity=maturity,
                    strike=strike[~invalid],
                    option_price=np.broadcast_to(option_price, strike.shape)[~invalid],
                    call_or_put_flag=np.broadcast_to(call_or_put_flag, strike.shape)[~invalid],
                    discount_factor=discount_factor,
                    check_price_bounds=False)
                sigma[invalid] = np.NaN
                return sigma

    # Initial guess for sigma. For ATM options, we use the explicit solution. For other options,
    # we use the infliction point of the Black function. See Jäckel (2006).
    sigma0 = np.where(np.isclose(x, 0), -2*norm.ppf((1-p)/2), np.sqrt(2*np.abs(x)))

    # Objective function: (normalized market price) - (normalized Black price)
    def f(sigma):
        return theta*(np.exp(x/2)*norm.cdf(theta*(x/sigma+sigma/2)) -
                      np.exp(-x/2)*norm.cdf(theta*(x/sigma-sigma/2))) - p

    # Objective function derivative (normalized vega)
    def fprime(sigma):
        return np.exp(x/2)*norm.pdf(x/sigma+sigma/2)

    res = newton(func=f, x0=sigma0, fprime=None, full_output=True, disp=False)

    if np.isscalar(strike):
        return res[0]/math.sqrt(maturity) if res[1].converged else np.NaN
    else:
        return np.where(res.converged, res.root/math.sqrt(maturity), np.NaN)


"""
REFERENCES

Jäckel, P. (2006). By Implication. http://www.jaeckel.org/ByImplication.pdf
"""
