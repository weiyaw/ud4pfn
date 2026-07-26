"""One-gap data-generating processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.stats import norm, poisson


class OneGapData(ABC):
    def __init__(self, key, n: int, shuffle: bool, x_design: str = "one-gap"):
        if x_design != "one-gap":
            raise ValueError("Gap experiments require x_design='one-gap'")
        self.key = key
        self.x_design = x_design
        key_data, key_shuffle = jr.split(key)
        key_left, key_right, key_y = jr.split(key_data, 3)
        left = jr.uniform(key_left, (n // 2,), minval=-8, maxval=-2)
        right = jr.uniform(key_right, (n - n // 2,), minval=2, maxval=8)
        self.X = np.asarray(jnp.concatenate([left, right])[:, None])
        self.y = np.asarray(self.get_y(key_y, self.X))
        if shuffle:
            permutation = jr.permutation(key_shuffle, n)
            self.X = self.X[permutation]
            self.y = self.y[permutation]

    @abstractmethod
    def get_y(self, key, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_true_event(self, x: np.ndarray, t: int | float) -> np.ndarray:
        raise NotImplementedError


class GaussianLinear(OneGapData):
    def _params(self, x):
        return (0.2 * x).squeeze(-1), 1.0

    def get_y(self, key, x):
        mean, scale = self._params(x)
        return np.asarray(mean + jr.normal(key, mean.shape) * scale, dtype=np.float32)

    def get_true_event(self, x, t):
        mean, scale = self._params(x)
        return norm.cdf(t, loc=mean, scale=scale).astype(np.float32)


class GaussianPolynomial(OneGapData):
    def _params(self, x):
        return (1.0 - 0.03 * x**2).squeeze(-1), 1.0

    def get_y(self, key, x):
        mean, scale = self._params(x)
        return np.asarray(mean + jr.normal(key, mean.shape) * scale, dtype=np.float32)

    def get_true_event(self, x, t):
        mean, scale = self._params(x)
        return norm.cdf(t, loc=mean, scale=scale).astype(np.float32)


class GaussianLinearDependentError(OneGapData):
    def _params(self, x):
        mean = (0.5 * x + 1.0).squeeze(-1)
        scale = (0.5 + 0.5 * np.abs(x)).squeeze(-1)
        return mean.astype(np.float32), scale

    def get_y(self, key, x):
        mean, scale = self._params(x)
        return np.asarray(mean + jr.normal(key, mean.shape) * scale, dtype=np.float32)

    def get_true_event(self, x, t):
        mean, scale = self._params(x)
        return norm.cdf(t, loc=mean, scale=scale).astype(np.float32)


class GaussianSine(OneGapData):
    def _params(self, x):
        return (0.5 * np.sin(x / 2)).squeeze(-1), 0.5

    def get_y(self, key, x):
        mean, scale = self._params(x)
        return np.asarray(mean + jr.normal(key, mean.shape) * scale, dtype=np.float32)

    def get_true_event(self, x, t):
        mean, scale = self._params(x)
        return norm.cdf(t, loc=mean, scale=scale).astype(np.float32)


class PoissonLinear(OneGapData):
    def _rate(self, x):
        return (0.05 * (x**2 - 80.0) + 5.0).squeeze(-1).astype(np.float32)

    def get_y(self, key, x):
        return np.asarray(jr.poisson(key, self._rate(x)), dtype=np.int32)

    def get_true_event(self, x, t):
        return poisson.cdf(t, self._rate(x)).astype(np.float32)


class ProbitMixture(OneGapData):
    def _probability(self, x):
        probability = (
            0.6 * norm.cdf((x - 8.0) / 4.0)
            + 0.4 * norm.cdf((x + 8.0) / 4.0)
        )
        return probability.squeeze(-1).astype(np.float32)

    def get_y(self, key, x):
        probability = self._probability(x)
        return np.asarray(jr.uniform(key, probability.shape) < probability, dtype=np.int32)

    def get_true_event(self, x, t):
        probability = self._probability(x)
        return (t * probability + (1 - t) * (1 - probability)).astype(np.float32)


class CategoricalLinear(OneGapData):
    def _probabilities(self, x):
        values = x.squeeze(-1)
        logits = np.zeros((values.size, 4), dtype=np.float32)
        logits[:, 0] = -(values + 5.0) ** 2 / 10.0
        logits[:, 1] = -(values**2) / 30.0
        logits[:, 2] = -(values - 7.0) ** 2 / 5.0
        logits[:, 3] = -(values - 4.0) ** 2 / 8.0
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def get_y(self, key, x):
        probabilities = self._probabilities(x)
        keys = jr.split(key, x.shape[0])
        return np.asarray(
            jax.vmap(lambda subkey, p: jr.choice(subkey, a=4, p=p))(
                keys, probabilities
            ),
            dtype=np.int32,
        )

    def get_true_event(self, x, t):
        return self._probabilities(x)[:, int(t)].astype(np.float32)


__all__ = [
    "GaussianLinear",
    "GaussianPolynomial",
    "GaussianLinearDependentError",
    "GaussianSine",
    "PoissonLinear",
    "ProbitMixture",
    "CategoricalLinear",
]
