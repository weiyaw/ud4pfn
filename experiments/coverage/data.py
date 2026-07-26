"""Ten-dimensional coverage data-generating processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.stats import norm, poisson, qmc


class CoverageData(ABC):
    def __init__(self, key, n: int, shuffle: bool, x_design: str = "sobol-10d"):
        if not x_design.startswith("sobol-"):
            raise ValueError("Coverage experiments require a Sobol design")
        self.key = key
        self.x_design = x_design
        self.dim = int(x_design.split("-")[-1].removesuffix("d"))
        key_data, key_shuffle = jr.split(key)
        self.X, self.y = self.get_xy(key_data, n)
        self.X, self.y = np.asarray(self.X), np.asarray(self.y)
        if shuffle:
            permutation = jr.permutation(key_shuffle, n)
            self.X = self.X[permutation]
            self.y = self.y[permutation]

    def get_x(self, key, n):
        seed = int(jr.randint(key, (), 0, 2_147_483_647))
        sampler = qmc.Sobol(
            d=self.dim, scramble=True, rng=np.random.default_rng(seed)
        )
        return jnp.asarray(2.0 * sampler.random(n=n) - 1.0, dtype=jnp.float32)

    def flatten_x(self, x):
        array = np.asarray(x, dtype=np.float32)
        if array.ndim < 2 or array.shape[-1] != self.dim:
            raise ValueError(f"Expected a final covariate dimension of {self.dim}")
        return array.reshape(-1, self.dim), array.shape[:-1]

    def weights(self, shift=0):
        index = np.arange(1, self.dim + 1, dtype=np.float32)
        weights = ((-1.0) ** index) * index
        if shift:
            weights = np.roll(weights, shift)
        return (weights / np.linalg.norm(weights)).astype(np.float32)

    @abstractmethod
    def get_xy(self, key, n):
        raise NotImplementedError

    @abstractmethod
    def get_true_event(self, x, t):
        raise NotImplementedError


class GaussianLinearMultivariate(CoverageData):
    def _params(self, x):
        flat, _ = self.flatten_x(x)
        return np.sqrt(1.5) * (flat @ self.weights()), float(np.sqrt(0.5))

    def get_xy(self, key, n):
        key_x, key_y = jr.split(key)
        x = self.get_x(key_x, n)
        mean, scale = self._params(x)
        y = mean + jr.normal(key_y, (n,)) * scale
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def get_true_event(self, x, t):
        flat, leading = self.flatten_x(x)
        mean, scale = self._params(flat)
        return norm.cdf(t, loc=mean, scale=scale).astype(np.float32).reshape(leading)


class GaussianLinearDependentErrorMultivariate(CoverageData):
    def _params(self, x):
        flat, _ = self.flatten_x(x)
        mean = np.sqrt(0.75) * (flat @ self.weights())
        scale = 0.75 + 0.25 * np.abs(flat @ self.weights(1))
        return mean.astype(np.float32), scale.astype(np.float32)

    def get_xy(self, key, n):
        key_x, key_y = jr.split(key)
        x = self.get_x(key_x, n)
        mean, scale = self._params(x)
        y = mean + jr.normal(key_y, (n,)) * scale
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def get_true_event(self, x, t):
        flat, leading = self.flatten_x(x)
        mean, scale = self._params(flat)
        return norm.cdf(t, loc=mean, scale=scale).astype(np.float32).reshape(leading)


class PoissonLinearMultivariate(CoverageData):
    def _rate(self, x):
        flat, _ = self.flatten_x(x)
        return np.clip(
            np.exp(0.5 * (flat @ self.weights())).astype(np.float32), 1e-3, None
        )

    def get_xy(self, key, n):
        key_x, key_y = jr.split(key)
        x = self.get_x(key_x, n)
        rate = self._rate(x)
        counts = jr.poisson(key_y, rate).astype(np.float32)
        y = (counts - rate) / np.sqrt(rate)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def get_true_event(self, x, t):
        flat, leading = self.flatten_x(x)
        rate = self._rate(flat)
        threshold = np.floor(t * np.sqrt(rate) + rate)
        return poisson.cdf(threshold, rate).astype(np.float32).reshape(leading)


class ProbitMixtureMultivariate(CoverageData):
    def _probability(self, x):
        flat, _ = self.flatten_x(x)
        first = norm.cdf(1.4 * (flat @ self.weights()))
        second = norm.cdf(1.4 * (flat @ self.weights(1)))
        return np.clip(0.5 * first + 0.5 * second, 1e-4, 1 - 1e-4).astype(
            np.float32
        )

    def get_xy(self, key, n):
        key_x, key_y = jr.split(key)
        x = self.get_x(key_x, n)
        probability = self._probability(x)
        y = np.where(jr.uniform(key_y, (n,)) < probability, 1, -1)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)

    def get_true_event(self, x, t):
        flat, leading = self.flatten_x(x)
        probability = self._probability(flat)
        if np.isclose(t, 1):
            result = probability
        elif np.isclose(t, -1) or np.isclose(t, 0):
            result = 1 - probability
        else:
            raise ValueError("Probit events must be one of {-1, 0, 1}")
        return result.astype(np.float32).reshape(leading)


class CategoricalLinearMultivariate(CoverageData):
    def _probabilities(self, x):
        flat, _ = self.flatten_x(x)
        first = flat @ self.weights()
        second = flat @ self.weights(1)
        logits = np.stack(
            [1.2 * first, -1.2 * first, 1.2 * second, -1.2 * second], axis=1
        )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return (probabilities / probabilities.sum(axis=1, keepdims=True)).astype(
            np.float32
        )

    def get_xy(self, key, n):
        key_x, key_y = jr.split(key)
        x = self.get_x(key_x, n)
        probabilities = self._probabilities(x)
        keys = jr.split(key_y, n)
        y = jax.vmap(lambda subkey, p: jr.choice(subkey, a=4, p=p))(
            keys, probabilities
        )
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)

    def get_true_event(self, x, t):
        flat, leading = self.flatten_x(x)
        if not float(t).is_integer() or not 0 <= int(t) <= 3:
            raise ValueError("Categorical events must be class indices in {0,1,2,3}")
        return self._probabilities(flat)[:, int(t)].reshape(leading)


__all__ = [
    "GaussianLinearMultivariate",
    "GaussianLinearDependentErrorMultivariate",
    "PoissonLinearMultivariate",
    "ProbitMixtureMultivariate",
    "CategoricalLinearMultivariate",
]
