import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
from scipy.stats import chi2
from scipy.special import gammaln, digamma


def compute_pointwise_coverage(true_curve, bands):
    # check if each point in the grid is covered, then average over grid points
    intervals = [(b["lower"], b["upper"]) for b in bands]
    for l, u in intervals:
        assert true_curve.shape == l.shape == u.shape

    if any(np.any(np.isnan(l)) or np.any(np.isnan(u)) for l, u in intervals):
        return np.nan

    is_covered = [(true_curve >= l) & (true_curve <= u) for (l, u) in intervals]
    return np.mean(np.asarray(is_covered))


def compute_simultaneous_coverage(true_curve, bands):
    # coverage of the entire curve
    intervals = [(b["lower"], b["upper"]) for b in bands]
    for l, u in intervals:
        assert true_curve.shape == l.shape == u.shape

    if any(np.any(np.isnan(l)) or np.any(np.isnan(u)) for l, u in intervals):
        return np.nan

    is_covered = [np.all((true_curve >= l) & (true_curve <= u)) for (l, u) in intervals]
    return np.mean(is_covered)


def build_pointwise_band(mean, cov, alpha: float = 0.05):
    se = np.sqrt(cov)
    z = norm.ppf(1 - alpha / 2)
    # lower = np.clip(mean - z * se, 0.0, 1.0)
    # upper = np.clip(mean + z * se, 0.0, 1.0)
    lower = mean - z * se
    upper = mean + z * se
    width = 2 * z * se
    return {"mean": mean, "lower": lower, "upper": upper, "se": se, "width": width}


def build_simultaneous_band(mean, cov, alpha: float = 0.05):
    # See Algorithm 1 of https://doi.org/10.1002/jae.2656
    se = np.sqrt(np.diag(cov))

    key = jr.key(501938)
    draws = jr.multivariate_normal(key, jnp.zeros_like(mean), cov, shape=(1000,))

    # Handle division by zero safely in JAX
    se_safe = jnp.where(se == 0, jnp.inf, se)

    Z = draws / se_safe[None, :]
    T = jnp.max(jnp.abs(Z), axis=1)
    c_alpha = jnp.quantile(T, 1 - alpha)
    lower = mean - c_alpha * se
    upper = mean + c_alpha * se
    width = jnp.mean(2 * c_alpha * se)

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "c_alpha": c_alpha,
        "se": se,
        "draws": draws,
        "width": width,
    }


def compute_ellipsoid_log_volume(cov, radius):
    # compute log of the volume of a high-dimensional ellipsoid defined by radius^2 > x^T cov^{-1} x
    d = cov.shape[0]
    log_unit_ball = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf  # Or raise error: Covariance must be positive definite

    return log_unit_ball + 0.5 * logdet + d * np.log(radius)


def build_ellipsoid_band(mean, cov, alpha: float = 0.05):
    # given a multivariate normal defined by mean and cov, compute the ellipsoid
    # such that the volume of the ellipsoid is 1-alpha

    d = mean.shape[0]
    # The squared radius corresponding to probability mass 1-alpha
    # is the quantile of the chi-squared distribution with d degrees of freedom.
    radius_sq = chi2.ppf(1 - alpha, df=d)
    radius = np.sqrt(radius_sq)

    log_vol = compute_ellipsoid_log_volume(cov, radius)

    # This is the projection of the ellipsoid to the coordinate axes
    se = np.sqrt(np.diag(cov))
    delta = se * radius
    lower = mean - delta
    upper = mean + delta

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "radius": radius,
        "log_volume": log_vol,
    }


def match_gaussian_beta_moments(mu, sigma2, eps=1e-7):
    """
    Moment matching for a Beta distribution to a Gaussian distribution. It
    returns the parameters (a, b) of the Beta distribution. If mu and sigma2 are
    arrays, their shapes must be the same and the function performs elementwise
    moment matching. 

    Parameters
    ----------
    mu: (p, m) array
        Mean of the Gaussian distribution.
    sigma2: (p, m) array
        Variance of the Gaussian distribution.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    (p, m) array
        Moment matching for Beta distribution(a, b).
    """
    assert mu.shape == sigma2.shape
    mu = np.clip(mu, eps, 1 - eps)
    max_var = mu * (1 - mu)
    sigma2 = np.minimum(sigma2, max_var - eps)
    sigma2 = np.maximum(sigma2, eps)

    T = (mu * (1 - mu)) / sigma2 - 1.0
    T = np.maximum(T, eps)

    a = mu * T
    b = (1 - mu) * T
    return a, b


def compute_beta_entropy(a, b):
    """
    Computes element-wise entropy of the Beta distribution E_{g~Beta(a,b)}[h(g)]
    in closed form:
    
        - a/(a+b) ψ(a+1) - b/(a+b) ψ(b+1) + ψ(a+b+1). 

    Parameters
    ----------
    a: (p, m) array
        Parameters of the Beta distribution.
    b: (p, m) array
        Parameters of the Beta distribution.

    Returns
    -------
    (p, m) array
        Entropy of the Beta distribution.
    """
    ab = a + b
    return -(a / ab) * digamma(a + 1) - (b / ab) * digamma(b + 1) + digamma(ab + 1)


def compute_aleatoric_entropy(gn, sigma2):
    """
    Computes element-wise aleatoric entropy

    Parameters
    ----------
    gn: (p, m) array
        Events of the PPD.
    sigma2: (p, m) array
        Variance of the PPD.

    Returns
    -------
    (p, m) array
        Aleatoric entropy.
    """
    a, b = match_gaussian_beta_moments(gn, sigma2)
    return compute_beta_entropy(a, b)


def compute_total_entropy(gn, eps=1e-7):
    """
    Computes element-wise total entropy, which is essentially the binary cross-entropy.

    Parameters
    ----------
    gn: (p, m) array
        Events of the PPD.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    (p, m) array
        Total entropy.
    """

    p = np.clip(gn, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)
