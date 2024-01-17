from typing import Optional
from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import norm


@dataclass
class MCResult:
    """Results of Monte-Carlo simulation.

    Attributes:
        x: Simulation average (mean value) or an array of averages.
        error: MoE (margin of error) for each average.
        conf_prob: Confidence probability of the confidence interval. Same for all averages.
        success: Indicator of whether the desired accuracy has been achieved for each average.
        iterations: Number of random realizations simulated.
        control_coef: Control variate coefficient (if it was used) for each average.
    """
    x: float | np.ndarray
    error: float | np.ndarray
    success: bool | np.ndarray
    conf_prob: float
    iterations: int
    control_coef: Optional[float | np.ndarray] = None

    def summary(self):
        """Returns a string with a summary of results in a human-readable form."""
        res = f"Value: {self.x}\n"
        if isinstance(self.x, float):
            res += f"Margin of error: {self.error}\n"
            res += "Desired accuracy achieved: " + ("Yes" if self.success else "No") + "\n"
        else:
            res += f"Maximum margin of error: {np.max(self.error)}\n"
            res += "Desired accuracy achieved: " + (
                "Yes" if np.all(self.success) else
                f"No (achieved at {np.sum(self.success)} out of {len(self.success)} values)") + "\n"
        res += f"Number of iterations: {self.iterations}\n"
        return res


def monte_carlo(simulator, f, abs_err=1e-3, rel_err=1e-3, conf_prob=0.95, batch_size=10_000,
                max_iter=10_000_000, control_f=None, control_estimation_iter=5000, rng=None):
    """The Monte-Carlo method for random processes.

    This function computes the expected value `E(f(X))`, where `f` is the provided function and `X`
    is a random process which is simulated by calling `simulator`.

    Several functions can be provided in a list, e.g. `[f1, f2, ...]`. In this case, the expected
    values `E(f1(X)), E(f2(X)), ...` are computed simultaneously on the same set of random paths.

    Simulation is performed in batches of random paths to allow speedup by vectorization. One batch
    is obtained in one call of `simulator`. Simulation is stopped when the maximum allowed number of
    paths has been exceeded or the method has converged in the sense that
        `error < abs_err + x*rel_err`
    where `x` is the current estimated mean value, `error` is the margin of error for the given
    confidence probability, i.e. `error = z*s/sqrt(n)`, where `z` is the critical value, `s` is the
    standard error, `n` is the number of paths simulated so far. In the case of a list of functions
    `f`, the convergence criterion is applied to each function separately and the method stops when
    the criterion is satisfied for all functions.

    It is also possible to provide a control variate, so that the desired value will be estimated as
        `E(f(X) - theta*control_f(X))`,
    which reduces the variance. The optimal coefficient `theta` is estimated by running a separate
    Monte-Carlo method with a small number of iterations (`control_estimation_iter`). The random
    variable corresponding to `control_f` must have zero expectation. Otherwise, the result will be
    incorrect. If a list of functions `f` is provided, it is possible to provide a single control
    variate `control_f` or a list of control variates `[control_f1, control_f2, ...]` of the same
    length as `f`. In any case, the optimal coefficients `theta` are estimated separately for each
    function in `f`.

    Args:
        simulator (callable): A function which produces random paths. It must accept two arguments
            (n, rng), where
                - n is the number of realizations to simulate (will be called with `n=batch_size` or
                    `n=control_estimation_iter`)
                - rng is a NumPy's random generator from which this function should sample; this
                    argument may be ignored, but then reproducibility of results is not guaranteed.
            The function must return an array of shape `(n, d)` where `d` is the number of sampling
            points in one path.
        f (callable): A function to apply to the simulated realizations or a list of functions. Must
            accept a batch of simulated paths (an array of shape `(n, d)`) and return a 1-D array of
            length `n`.
        abs_err (float): Desired absolute error. Same for all functions in `f` if `f` is a list.
        rel_err (float): Desired relative error. Same for all functions in `f` if `f` is a list.
        conf_prob  (float): Desired confidence probability. Same for all functions in `f` if `f` is
            a list.
        batch_size (int): Number of random realizations returned in one call to `simulator`.
        max_iter (int): Maximum allowed number of simulated realizations. The desired errors may may
            be not reached if more than `max_iter` paths are required.
        control_f (callable): A control variate or a list of control variates. Must be a function
            which satisfies the same requirements as `f`. If `f` is a list, then `control_f` can be
            a single function or a list of functions of the same length as `f`.
        control_estimation_iter (int): Number of random realizations for estimating `theta`. Same
            for all functions in `f` if `f` is a list.
        rng: NumPy's random number generator to be used. If None, `numpy.random.default_rng` will be
            used.

    Returns:
        MCResult: A structure with simulation results. If `f` is a single function, the structure
        will contain scalar values. If `f` is a list of functions, the structure will contain arrays
        of the same length as `f`.
    """
    # Convert `f` and `control_f` to lists if they are not lists
    if isinstance(f, list):
        f_ = f
    else:
        f_ = [f]
    if control_f is None:
        # zero function if no control variate is provided
        control_f_ = [lambda _: 0]*len(f_)
    elif isinstance(control_f, list):
        control_f_ = control_f
    else:
        control_f_ = [control_f]*len(f_)

    if rng is None:
        rng = np.random.default_rng()

    # Estimation of control variate coefficient `theta`
    theta = np.zeros(len(f_))
    if control_f is not None:
        S = simulator(control_estimation_iter, rng)
        for i in range(len(f_)):
            c = np.cov(f_[i](S), control_f_[i](S))
            theta[i] = c[0, 1] / c[1, 1]
    else:
        theta = np.zeros(len(f_))

    # Initialize variables
    z = norm.ppf((1+conf_prob)/2)           # critical value
    y = np.empty((len(f_), batch_size))     # batch of values minus control variate
    x = np.zeros(len(f_))                   # current means
    x_sq = np.zeros(len(f_))                # current means of squares
    s = np.zeros(len(f_))                   # current standard errors
    n = 0                                   # batches counter
    error = np.zeros(len(f_))               # margins of error

    # Main loop
    while (n == 0 or
           np.any(error > abs_err + np.abs(x)*rel_err) and
           n*batch_size < max_iter):
        # Simulate paths
        S = simulator(batch_size, rng)
        # Update means and standard errors for each function in `f`
        for i in range(len(f_)):
            y[i] = f_[i](S) - theta[i]*control_f_[i](S)
        x = (x*n + np.mean(y, axis=1))/(n+1)
        x_sq = (x_sq*n + np.mean(y**2, axis=1))/(n+1)
        s = np.sqrt(x_sq - x**2)

        # Update number of batches and margins of error
        n += 1
        error = z*s/math.sqrt(n*batch_size)

    # If `f` is a list, return a list of results. Otherwise, return a single result.
    if isinstance(f, list):
        return MCResult(
            x=x,
            error=error,
            success=(error <= abs_err + np.abs(x)*rel_err),
            conf_prob=conf_prob,
            iterations=n*batch_size,
            control_coef=theta)
    else:
        return MCResult(
            x=x[0],
            error=error[0],
            success=(error[0] <= abs_err + np.abs(x[0])*rel_err),
            conf_prob=conf_prob,
            iterations=n*batch_size,
            control_coef=theta[0])
