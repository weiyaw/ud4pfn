"""Moment-matched entropy decompositions used by this experiment."""

import numpy as np
from scipy.special import digamma


def match_gaussian_beta_moments(mu, sigma2, eps=1e-12):
    assert mu.shape == sigma2.shape
    mu = np.clip(mu, eps, 1 - eps)
    max_var = mu * (1 - mu)
    sigma2 = np.minimum(sigma2, max_var - eps)
    sigma2 = np.maximum(sigma2, eps)
    concentration = np.maximum((mu * (1 - mu)) / sigma2 - 1.0, eps)
    return mu * concentration, (1 - mu) * concentration


def compute_aleatoric_entropy_binary(gn, sigma2):
    a, b = match_gaussian_beta_moments(gn, sigma2)
    total = a + b
    return (
        -(a / total) * digamma(a + 1)
        - (b / total) * digamma(b + 1)
        + digamma(total + 1)
    )


def compute_total_entropy_binary(gn, eps=1e-12):
    probability = np.clip(gn, eps, 1 - eps)
    return -probability * np.log(probability) - (1 - probability) * np.log(
        1 - probability
    )


def match_gaussian_dirichlet_moments(mu, sigma2, eps=1e-12):
    numerator = 1.0 - np.sum(mu**2, axis=0)
    denominator = np.sum(sigma2, axis=0)
    denominator = np.minimum(denominator, numerator - eps)
    denominator = np.maximum(denominator, eps)
    concentration = np.maximum(numerator / denominator - 1.0, eps)
    return mu * concentration


def compute_total_entropy_multiclass(gn, eps=1e-12):
    probability = np.clip(gn, eps, 1.0)
    return -np.sum(probability * np.log(probability), axis=0)


def compute_aleatoric_entropy_multiclass(gn, sigma2, eps=1e-12):
    alpha = match_gaussian_dirichlet_moments(gn, sigma2, eps=eps)
    alpha_sum = np.sum(alpha, axis=0)
    probability = alpha / alpha_sum
    return digamma(alpha_sum + 1.0) - np.sum(
        probability * digamma(alpha + 1.0), axis=0
    )


__all__ = [
    "match_gaussian_beta_moments",
    "compute_aleatoric_entropy_binary",
    "compute_total_entropy_binary",
    "match_gaussian_dirichlet_moments",
    "compute_total_entropy_multiclass",
    "compute_aleatoric_entropy_multiclass",
]
