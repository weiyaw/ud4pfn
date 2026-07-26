"""Data-generating processes for entropic uncertainty decomposition."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import jax
import jax.random as jr
import numpy as np


class EntropicData(ABC):
    def __init__(self, key, n: int, shuffle: bool, x_design: str | None = None):
        self.key = key
        self.x_design = x_design
        key_data, key_shuffle = jr.split(key)
        self.X, self.y = self.get_xy(key_data, n)
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        if shuffle:
            permutation = jr.permutation(key_shuffle, n)
            self.X = self.X[permutation]
            self.y = self.y[permutation]

    @abstractmethod
    def get_xy(self, key, n):
        raise NotImplementedError

    @abstractmethod
    def get_true_event(self, x, t):
        raise NotImplementedError


class LogisticLinear(EntropicData):
    def get_xy(self, key, n):
        if not str(self.x_design).startswith("gaussian:"):
            raise ValueError("LogisticLinear requires a gaussian:<mean>:<std> design")
        mean, scale = map(float, str(self.x_design).split(":")[1:])
        key_x, key_y = jr.split(key)
        x = mean + scale * jr.normal(key_x, (n, 1))
        probability = self._probability(np.asarray(x))
        y = jr.uniform(key_y, probability.shape) < probability
        return np.asarray(x), np.asarray(y, dtype=np.int32)

    @staticmethod
    def _probability(x):
        logits = 0.25 * x - 0.5
        return (1.0 / (1.0 + np.exp(-logits))).squeeze(-1).astype(np.float32)

    def get_true_event(self, x, t):
        probability = self._probability(x)
        return t * probability + (1 - t) * (1 - probability)


def make_moons(key, n: int, noise_std: float):
    n_outer = n // 2
    n_inner = n - n_outer
    outer_x = np.cos(np.linspace(0, np.pi, n_outer))
    outer_y = np.sin(np.linspace(0, np.pi, n_outer))
    inner_x = 1 - np.cos(np.linspace(0, np.pi, n_inner))
    inner_y = 1 - np.sin(np.linspace(0, np.pi, n_inner)) - 0.5
    x = np.vstack(
        [np.append(outer_x, inner_x), np.append(outer_y, inner_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_outer, dtype=np.intp), np.ones(n_inner, dtype=np.intp)]
    )
    x += noise_std * jr.normal(key, shape=x.shape)
    return x.astype(float), y.astype(int)


class TwoMoons1(EntropicData):
    def get_xy(self, key, n):
        return make_moons(key, n, noise_std=0.1)

    def get_true_event(self, x, t):
        return np.full(x.shape[0], np.nan)


class TwoMoons2(EntropicData):
    def get_xy(self, key, n):
        return make_moons(key, n, noise_std=0.4)

    def get_true_event(self, x, t):
        return np.full(x.shape[0], np.nan)


class Spiral(EntropicData):
    def get_xy(self, key, n):
        counts = [n // 3 + (1 if arm < n % 3 else 0) for arm in range(3)]
        x_parts, y_parts = [], []
        for arm, count in enumerate(counts):
            arm_key = jr.fold_in(key, arm)
            arm_key, subkey = jr.split(arm_key)
            position = jr.uniform(subkey, shape=(count,))
            radius = 4.0 * position
            angle = 4.0 * math.pi * position + 2.0 * math.pi * arm / 3
            arm_key, subkey = jr.split(arm_key)
            x1 = radius * np.cos(angle) + jr.normal(subkey, (count,)) * 0.1
            arm_key, subkey = jr.split(arm_key)
            x2 = radius * np.sin(angle) + jr.normal(subkey, (count,)) * 0.1
            x_parts.append(np.c_[x1, x2])
            y_parts.append(np.full(count, arm, dtype=int))
        return np.vstack(x_parts).astype(float), np.concatenate(y_parts).astype(int)

    def get_true_event(self, x, t):
        return np.full(x.shape[0], np.nan)


__all__ = ["LogisticLinear", "TwoMoons1", "TwoMoons2", "Spiral", "make_moons"]
