from tabpfn import TabPFNClassifier, TabPFNRegressor
import numpy as np
import torch
from typing import Callable

import warnings
import jax
import jax.numpy as jnp
import jax.random as jr


def assert_ppd_args_shape(x_new, x_prev, y_prev):
    assert x_new.ndim == 2, "x_new must be 2D array"
    assert x_prev.ndim == 2, "x_prev must be 2D array"
    assert y_prev.ndim == 1, "y_prev must be 1D array"
    assert (
        x_prev.shape[0] == y_prev.shape[0]
    ), "x_prev and y_prev must have same number of samples"
    assert (
        x_prev.shape[1] == x_new.shape[1]
    ), "x_prev and x_new must have same number of features"
    assert y_prev.ndim == 1, "y_prev must be 1D array"


class TabPFNRegressorPPD(TabPFNRegressor):

    def sample(
        self,
        key: jax.random.key,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        size: int = 1,
    ) -> tuple[np.ndarray, dict]:
        """
        Sample from predictive density

        Parameters
        ----------
        key : jax.random.key
            Random key.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.
        size : int, default=1
            Number of samples.

        Return:
        -------
        tuple[np.ndarray, dict]
            Sampled values and additional information.
            Shape: (size, m)
        """
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")
        bardist = pred_output["criterion"]
        logits = pred_output["logits"]
        assert logits.ndim == 2, "logits must be 2D array (num_data, num_of_bins)"

        y_new = []
        EPS = 1e-5
        for i in range(size):
            all_u = jr.uniform(
                jr.fold_in(key, i), shape=(logits.shape[0],), minval=EPS, maxval=1 - EPS
            ) # icdf doesn't like u that are too close to 0 and 1
            y_new.append(
                np.array(
                    [bardist.icdf(l, float(u)).cpu() for l, u in zip(logits, all_u)]
                )
            )

        y_new = np.stack(y_new, axis=0)  # (size, num_x_new)
        return y_new, {"bardist": bardist, "logits": logits.cpu().numpy()}

    def icdf(
        self, u: np.ndarray, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> np.ndarray:
        """
        Return inverse CDF of P(Y <= t | X = x_new, x_prev, y_prev) given a
        value u between [0, 1].

        Parameters
        ----------
        u : (p, ) array
            Values between [0, 1].
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            Inverse CDF values. Each row corresponds to a value of u, and each
            column corresponds to a value of x_new. Shape: (p, m)
        """
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")
        bardist = pred_output["criterion"]
        logits = pred_output["logits"]  # (m, num_of_bins)

        all_u = np.atleast_1d(u)
        assert all_u.ndim == 1, "u must be 1D array"

        # For each u, compute for all x_new
        results = [[bardist.icdf(l, float(u)).cpu() for l in logits] for u in all_u]
        return np.array(results)

    def predict_event(
        self, t: np.ndarray, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> np.ndarray:
        """
        Return P(Y <= t | X = x_new, x_prev, y_prev).

        Parameters
        ----------
        t : (p, ) array
            Events of the PPD.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            P(Y <= t | X = x_new, prev data). Each row corresponds to a value of t, and each column corresponds to a value of x_new.
            Shape: (p, m)
        """
        return self.cdf(t, x_new, x_prev, y_prev)

    def cdf(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """
        Return P(Y <= t | X = x_new, prev data).

        Parameters
        ----------
        t : (p, ) array
            Events of the PPD.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            P(Y <= t | X = x_new, prev data). Each row corresponds to a value of t, and each column corresponds to a value of x_new.
            Shape: (p, m)
        """
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")

        logits = pred_output["logits"]  # shape: (m, num_of_bins)
        bardist = pred_output["criterion"]

        # t must be a 1D float array
        t = np.atleast_1d(t)
        assert t.ndim == 1
        assert np.issubdtype(t.dtype, np.floating)

        bardist.borders = bardist.borders.cpu()
        results = []
        for single_t in t:
            # Evaluate the predictive CDF at a single t for each x_new query point.
            ys = torch.full(
                (logits.shape[0], 1),
                float(single_t),
                dtype=torch.float32,
            )
            # cdf returns (m, 1) or (m,), squeeze to ensure (m,)
            cdf_val = bardist.cdf(logits.cpu(), ys).squeeze(-1)
            results.append(cdf_val.numpy())

        # Stack to get (p, m)
        return np.stack(results)


# %%


class TabPFNClassifierPPD(TabPFNClassifier):

    def sample(
        self,
        key: jr.key,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        size: int = 1,
    ) -> tuple[np.ndarray, dict]:
        """
        Sample from predictive density.

        Parameters
        ----------
        key : jax.random.key
            Random key.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.
        size : int, default=1
            Number of samples.

        Return:
        -------
        tuple[np.ndarray, dict]
            Sampled values and additional information.
            Shape: (size, m)
        """
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        probs_new = self.predict_proba(x_new)  # shape: (m, num_classes)

        # Draw index for each x_new across size samples
        # JAX version of random choice for each probabilities row
        def sample_classes(subkey, p):
            idx = jr.choice(subkey, a=self.classes_.size, shape=(size,), p=p)
            return jnp.array(self.classes_)[idx]

        keys = jr.split(key, probs_new.shape[0])
        y_new = jax.vmap(sample_classes)(keys, probs_new).T  # (n, num_x_new)

        return y_new, {"probs": probs_new}

    def pmf(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return P(Y = t | X = x_new, prev data).

        Parameters
        ----------
        t: (p, ) array
            Event of the PPD.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            P(Y = t | X = x_new, prev data). Each row corresponds to a value of t, and each column corresponds to a value of x_new.
            Shape: (p, m)
        """

        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)

        # Use logits for higher precision
        logits = self.predict_logits(x_new)  # shape: (m, num_classes)

        # Convert to float64 for precision
        logits = logits.astype(np.float64)

        # Compute softmax in float64
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # t must be a 1D integer array
        t = np.atleast_1d(t)
        assert t.ndim == 1
        assert np.issubdtype(t.dtype, np.integer)

        def predict_event_single_t(single_t: int) -> np.ndarray:
            # Create a mask for the class: (num_classes,)
            matches = self.classes_ == single_t
            # Dot product selects the column or results in 0 if no match
            # probs: (m, num_classes), matches: (num_classes,) -> (m,)
            return np.dot(probs, matches.astype(np.float64))

        event_prob = np.array([predict_event_single_t(ti) for ti in t])
        return event_prob

    def predict_event(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return P(Y = t | X = x_new, prev data).

        Parameters
        ----------
        t: (p, ) array
            Event of the PPD.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            P(Y = t | X = x_new, prev data). Each row corresponds to a value of t, and each column corresponds to a value of x_new.
            Shape: (p, m)
        """
        return self.pmf(t, x_new, x_prev, y_prev)


class BayesianBootstrapPPD:
    def __init__(self):
        self.x = None
        self.y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def sample(self, key: jr.key, x_new: np.ndarray, size: int = 1) -> np.ndarray:
        pass

    def predict_event(self, t: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        pass


# %%
