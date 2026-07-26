"""TabICL predictive-distribution adapters for the predictive CLT."""

import jax.random as jr
import numpy as np
import torch
from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._model.quantile_dist import QuantileDistribution

from .tabpfn_adapter import assert_ppd_args_shape


def _quantile_distribution(
    raw_quantiles: np.ndarray,
) -> QuantileDistribution:
    """Build TabICL's predictive distribution from raw quantile output."""
    raw_quantiles = np.asarray(raw_quantiles)
    assert raw_quantiles.ndim == 2, (
        "raw_quantiles must be 2D array (num_data, num_quantiles)"
    )
    return QuantileDistribution(torch.from_numpy(raw_quantiles).float())


class TabICLRegressorPPD(TabICLRegressor):
    """Extend TabICL regression with predictive-distribution operations."""

    def _fit_predict_distribution(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> tuple[np.ndarray, QuantileDistribution]:
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        raw_quantiles = np.asarray(
            self.predict(x_new, output_type="raw_quantiles")
        )
        return raw_quantiles, _quantile_distribution(raw_quantiles)

    def sample(
        self,
        key: jr.key,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        size: int = 1,
    ) -> tuple[np.ndarray, dict]:
        """Draw ``size`` samples from each query-point predictive distribution."""
        assert size >= 1, "size must be at least 1"
        raw_quantiles, distribution = self._fit_predict_distribution(
            x_new, x_prev, y_prev
        )
        uniforms = jr.uniform(
            key,
            shape=(x_new.shape[0], size),
            minval=1e-5,
            maxval=1 - 1e-5,
        )
        probabilities = torch.from_numpy(
            np.array(uniforms, dtype=np.float32, copy=True)
        )
        with torch.no_grad():
            samples = distribution.icdf(probabilities).detach().cpu().numpy().T
        return samples, {
            "raw_quantiles": raw_quantiles,
            "quantile_distribution": distribution,
        }

    def icdf(
        self,
        u: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Evaluate inverse CDFs, returning shape ``(len(u), len(x_new))``."""
        _, distribution = self._fit_predict_distribution(
            x_new, x_prev, y_prev
        )
        u = np.atleast_1d(u)
        assert u.ndim == 1, "u must be 1D array"
        assert np.all((u >= 0) & (u <= 1)), "u must contain values in [0, 1]"
        probabilities = torch.from_numpy(np.asarray(u, dtype=np.float32))
        with torch.no_grad():
            values = distribution.icdf(probabilities)
        return values.detach().cpu().numpy().T

    def cdf(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Evaluate predictive CDFs, returning shape ``(len(t), len(x_new))``."""
        _, distribution = self._fit_predict_distribution(
            x_new, x_prev, y_prev
        )
        t = np.atleast_1d(t)
        assert t.ndim == 1, "t must be 1D array"
        thresholds = torch.from_numpy(
            np.asarray(t, dtype=np.float32)
        ).unsqueeze(0)
        thresholds = thresholds.expand(x_new.shape[0], -1)
        with torch.no_grad():
            probabilities = distribution.cdf(thresholds)
        return probabilities.detach().cpu().numpy().T

    def predict_event(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return ``P(Y <= t | X=x_new, x_prev, y_prev)``."""
        return self.cdf(t, x_new, x_prev, y_prev)


class TabICLClassifierPPD(TabICLClassifier):
    """Extend TabICL classification with predictive-distribution operations."""

    def sample(
        self,
        key: jr.key,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        size: int = 1,
    ) -> tuple[np.ndarray, dict]:
        """Draw ``size`` class labels for every query point."""
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        assert size >= 1, "size must be at least 1"
        self.fit(x_prev, y_prev)
        probabilities = np.asarray(self.predict_proba(x_new))

        keys = jr.split(key, probabilities.shape[0])
        sampled_indices = np.stack(
            [
                np.asarray(
                    jr.choice(
                        subkey,
                        a=self.classes_.size,
                        shape=(size,),
                        p=probability,
                    )
                )
                for subkey, probability in zip(keys, probabilities)
            ],
            axis=1,
        )
        samples = np.asarray(self.classes_)[sampled_indices]
        return samples, {"probs": probabilities}

    def pmf(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return ``P(Y = t | X=x_new, x_prev, y_prev)``."""
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        probabilities = np.asarray(self.predict_proba(x_new), dtype=np.float64)

        t = np.atleast_1d(t)
        assert t.ndim == 1, "t must be 1D array"
        classes = np.asarray(self.classes_)
        return np.stack(
            [
                probabilities @ (classes == event).astype(np.float64)
                for event in t
            ]
        )

    def predict_event(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return ``P(Y = t | X=x_new, x_prev, y_prev)``."""
        return self.pmf(t, x_new, x_prev, y_prev)
