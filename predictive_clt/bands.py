"""Generic Gaussian credible bands for predictive-CLT approximations."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.stats import norm
from scipy.special import gammaln
from scipy.stats import chi2


@jax.jit
def build_pointwise_band(mean, cov, alpha: float = 0.05):
    assert cov.ndim == 1
    se = jnp.sqrt(cov)
    z = norm.ppf(1 - alpha / 2)
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

    # Handle division by zero safely in JAX.
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
    # Compute the log-volume of radius^2 > x^T cov^{-1} x.
    d = cov.shape[0]
    log_unit_ball = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf

    return log_unit_ball + 0.5 * logdet + d * np.log(radius)


def build_ellipsoid_band(mean, cov, alpha: float = 0.05):
    d = mean.shape[0]
    radius_sq = chi2.ppf(1 - alpha, df=d)
    radius = np.sqrt(radius_sq)
    log_vol = compute_ellipsoid_log_volume(cov, radius)

    # Projection of the ellipsoid onto the coordinate axes.
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


__all__ = [
    "build_pointwise_band",
    "build_simultaneous_band",
    "build_ellipsoid_band",
    "compute_ellipsoid_log_volume",
]
