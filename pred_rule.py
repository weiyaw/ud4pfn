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

    def __init__(self, *args, y_star: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_star = y_star

    def sample(
        self,
        key: jr.key,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        size: int = 1,
    ) -> tuple[np.ndarray, dict]:
        # Sample from predictive density
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
        y_new = [self.bardist_sample(jr.fold_in(key, i), bardist.icdf, logits) for i in range(size)]
        y_new = np.stack(y_new, axis=0)  # (n, num_x_new)

        return y_new, {"bardist": bardist, "logits": logits.cpu().numpy()}

    def bardist_sample(
        self, key: jr.key, bardist_icdf: Callable, logits: torch.Tensor
    ) -> np.ndarray:
        """Samples values from the bar distribution. A modified version of
        https://github.com/PriorLabs/TabPFN/blob/1b786570f5d5da3f3b9b6179c3fa43faf0c77894/src/tabpfn/architectures/base/bar_distribution.py#L581
        Temperature t.
        """
        assert logits.ndim == 2, "logits must be 2D array (num_data, num_of_bins)"
        p_cdf = jr.uniform(key, shape=(logits.shape[0],))
        return np.array(
            [bardist_icdf(logits[i, :], p).cpu() for i, p in enumerate(p_cdf.tolist())]
        )

    def predict_event(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        # y_star: float = 3.0,
    ) -> np.ndarray:
        """
        Return P(Y <= y_star | X = x_new, prev data).

        Parameters
        ----------
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.
        y_star : float, default=3.0
            Event threshold.
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

        logits = pred_output["logits"]
        bardist = pred_output["criterion"]

        # Evaluate the predictive CDF at y_star for each query point.
        ys = torch.full(
            (logits.shape[0], 1),
            float(self.y_star),
            dtype=torch.float32,
        )

        bardist.borders = bardist.borders.cpu()
        cdf_vals = bardist.cdf(logits.cpu(), ys).squeeze(-1)
        return cdf_vals.numpy()


# %%


class TabPFNClassifierPPD(TabPFNClassifier):

    def __init__(self, *args, y_star: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_star = y_star

    def sample(
        self,
        key: jr.key,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        size: int = 1,
    ) -> tuple[np.ndarray, dict]:
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        probs_new = self.predict_proba(x_new)

        # Draw index for each x_new across size samples
        # JAX version of random choice for each probabilities row
        def sample_classes(subkey, p):
              idx = jr.choice(subkey, a=self.classes_.size, shape=(size,), p=p)
              return jnp.array(self.classes_)[idx]

        keys = jr.split(key, probs_new.shape[0])
        y_new = jax.vmap(sample_classes)(keys, probs_new).T # (n, num_x_new)

        return y_new, {"probs": probs_new}

    def predict_event(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return P(Y = 1 | X = x_new, prev data)."""
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        probs = self.predict_proba(x_new)

        if self.y_star in self.classes_:
            # Identify the column corresponding to the positive class label.
            class_idx = np.where(self.classes_ == self.y_star)[0]
            event_prob = probs[:, class_idx[0]].squeeze()
        else:
            # If y_star is not supported, return zero probability
            event_prob = np.zeros(probs.shape[0], dtype=probs.dtype)
        return event_prob


# %%
