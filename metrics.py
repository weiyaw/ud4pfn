import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.stats import norm
from scipy.special import digamma, gammaln
from scipy.stats import chi2


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


@jax.jit
def build_pointwise_band(mean, cov, alpha: float = 0.05):
    assert cov.ndim == 1
    se = jnp.sqrt(cov)
    z = norm.ppf(1 - alpha / 2)
    # lower = np.clip(mean - z * se, 0.0, 1.0)
    # upper = np.clip(mean + z * se, 0.0, 1.0)
    lower = mean - z * se
    upper = mean + z * se
    width = 2 * z * se
    return {"mean": mean, "lower": lower, "upper": upper, "se": se, "width": width}


@jax.jit
def build_simultaneous_band(mean, cov, alpha: float = 0.05):
    # See Algorithm 1 of https://doi.org/10.1002/jae.2656
    assert cov.ndim == 2
    se = jnp.sqrt(jnp.diag(cov))
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


def build_bootstrap_pointwise_band(mean, bootstrap_samples, alpha: float = 0.05):
    """
    Percentile bootstrap pointwise interval.

    Parameters
    ----------
    mean : (m,) ndarray
        Base predictions computed on the original dataset.
    bootstrap_samples : (B, m) ndarray
        Bootstrap predictions across B bootstrap resamples.
    alpha : float
        Miscoverage level.
    """
    mean = np.asarray(mean)
    bootstrap_samples = np.asarray(bootstrap_samples)
    assert mean.ndim == 1
    assert bootstrap_samples.ndim == 2
    assert bootstrap_samples.shape[1] == mean.shape[0]

    lower = np.quantile(bootstrap_samples, alpha / 2, axis=0)
    upper = np.quantile(bootstrap_samples, 1 - alpha / 2, axis=0)
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    width = np.mean(upper - lower)

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "width": width,
    }


def build_bootstrap_simultaneous_band(
    mean,
    bootstrap_samples,
    alpha: float = 0.05,
    studentize: bool = False,
    eps: float = 1e-12,
):
    """
    Sup-norm bootstrap simultaneous band.

    If ``studentize`` is True, uses sup-t style studentization by dividing
    pointwise deviations with bootstrap standard errors.
    """
    mean = np.asarray(mean)
    bootstrap_samples = np.asarray(bootstrap_samples)
    assert mean.ndim == 1
    assert bootstrap_samples.ndim == 2
    assert bootstrap_samples.shape[1] == mean.shape[0]

    diff = bootstrap_samples - mean[None, :]

    if studentize:
        se = np.std(bootstrap_samples, axis=0, ddof=1)
        se = np.maximum(se, eps)
        max_dev = np.max(np.abs(diff) / se[None, :], axis=1)
        c_alpha = np.quantile(max_dev, 1 - alpha)
        delta = c_alpha * se
    else:
        max_dev = np.max(np.abs(diff), axis=1)
        c_alpha = np.quantile(max_dev, 1 - alpha)
        se = None
        delta = c_alpha

    lower = np.clip(mean - delta, 0.0, 1.0)
    upper = np.clip(mean + delta, 0.0, 1.0)
    width = np.mean(upper - lower)

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "c_alpha": c_alpha,
        "se": se,
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


def match_gaussian_beta_moments(mu, sigma2, eps=1e-12):
    """
    Moment matching for a Beta distribution to a Gaussian distribution. It
    returns the parameters (a, b) of the Beta distribution. If mu and sigma2 are
    arrays, their shapes must be the same and the function performs elementwise
    moment matching.

    Parameters
    ----------
    mu: ndarray
        Mean of the Gaussian distribution.
    sigma2: ndarray
        Variance of the Gaussian distribution.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    ndarray
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


def compute_aleatoric_entropy_binary(gn, sigma2):
    """
    Computes element-wise aleatoric entropy, which is the expected entropy
    E_{g~Beta(a,b)}[h(g)] in closed form:

        - a/(a+b) ψ(a+1) - b/(a+b) ψ(b+1) + ψ(a+b+1).

    Parameters
    ----------
    gn: (m, ) ndarray
        Events of the PPD.
    sigma2: (m, ) ndarray
        Variance of the PPD.

    Returns
    -------
    (m, ) ndarray
        Aleatoric entropy.
    """
    a, b = match_gaussian_beta_moments(gn, sigma2)
    ab = a + b
    return -(a / ab) * digamma(a + 1) - (b / ab) * digamma(b + 1) + digamma(ab + 1)


def compute_total_entropy_binary(gn, eps=1e-12):
    """
    Computes element-wise total entropy, which is essentially the binary cross-entropy.

    Parameters
    ----------
    gn: (m, ) ndarray
        Events of the PPD.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    (m, ) ndarray
        Total entropy.
    """

    p = np.clip(gn, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def match_gaussian_dirichlet_moments(mu, sigma2, eps=1e-12):
    """
    Moment matching for a Dirichlet distribution to a Gaussian distribution.

    Parameters
    ----------
    mu: (K, m) ndarray
        Mean of the Gaussian distribution (probabilities).
    sigma2: (K, m) ndarray
        Variance of the Gaussian distribution.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    (K, m) ndarray
        Parameters alpha of the Dirichlet distribution.
    """
    # Calculate Total Variance of the predictive mean vector
    # ||mu||^2 sum over classes (axis=0)
    numerator = 1.0 - np.sum(mu**2, axis=0)  # (m,)

    # Sum of the CLT variances across classes (axis=0)
    denominator = np.sum(sigma2, axis=0)  # (m,)

    # Sum of the predictive variances is clipped to be strictly less than 1 - ||g_n||^2
    denominator = np.minimum(denominator, numerator - eps)
    denominator = np.maximum(denominator, eps)

    # Strict moment matching
    alpha0 = numerator / denominator - 1.0  # (m,)
    alpha0 = np.maximum(alpha0, eps)  # Enforce positivity

    # alpha_k = g_k * alpha0
    alpha = mu * alpha0  # (K, m)

    return alpha


def compute_total_entropy_multiclass(gn, eps=1e-12):
    """
    Computes element-wise Total Uncertainty (Entropy).

    H(y* | x*, D) = - sum_k g_k log g_k

    Parameters
    ----------
    gn: (K, m) ndarray
        Predictive mean vector (probabilities).
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    (m,) ndarray
        Total entropy.
    """
    gn = np.clip(gn, eps, 1.0)  # (K, m)
    return -np.sum(gn * np.log(gn), axis=0)  # (m,)


def compute_aleatoric_entropy_multiclass(gn, sigma2, eps=1e-12):
    """
    Computes aleatoric uncertainty for K-class classification, which is the
    expected entropy of the categorical distribution under the Dirichlet
    distribution with K classes.

    E[H(p)] where p ~ Dir(a) = psi(alpha0 + 1) - sum_k (alpha_k / alpha0) *
    psi(alpha_k + 1) where alpha0 = sum_k alpha_k.

    Parameters
    ----------
    gn: (K, n) ndarray
        Predictive mean vector.
    sigma2: (K, n) ndarray
        Predictive variance (from CLT).
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    (m,) ndarray
        Aleatoric entropy.
    """
    alpha = match_gaussian_dirichlet_moments(gn, sigma2, eps=eps)  # (K, m)
    alpha_sum = np.sum(alpha, axis=0)  # alpha0, shape (m,)

    # g_k = alpha_k / alpha0
    g = alpha / alpha_sum  # (K, m)

    term1 = digamma(alpha_sum + 1.0)  # (m,)
    term2 = np.sum(g * digamma(alpha + 1.0), axis=0)  # (m,)

    return term1 - term2

