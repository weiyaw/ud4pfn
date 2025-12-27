# %%
from abc import abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import norm, poisson, gamma
from dataclasses import dataclass
from typing import Union



@dataclass
class Data:
    key: jr.key
    X: np.ndarray
    y: np.ndarray
    x_design: str

    def __init__(
        self,
        key: jr.key,
        n: int,
        shuffle: bool = False,
        x_design: str = "one-gap",
    ):
        self.key = key
        self.x_design = x_design
        key_x, key_y, key_shuffle = jr.split(key, 3)
        self.X = self.get_x(key_x, n, x_design)
        # get y from subclass and compute true_curve separately via get_true_curve
        self.y = self.get_y(key_y, self.X)

        if shuffle:
            perm = jr.permutation(key_shuffle, n)
            self.X = self.X[perm]
            self.y = self.y[perm]

    def get_x(self, key, n, x_design="one-gap") -> np.ndarray:
        # uniform between -10 and 10. no data in the gaps
        if x_design == "one-gap":
            key1, key2 = jr.split(key)
            xs1 = jr.uniform(key1, shape=(n // 2,), minval=-8, maxval=-2)
            xs2 = jr.uniform(key2, shape=(n - n // 2,), minval=2, maxval=8)
            xs = jnp.concatenate([xs1, xs2])
        elif x_design == "two-gap":
            k1, k2, k3 = jr.split(key, 3)
            xs1 = jr.uniform(k1, shape=(n // 3,), minval=-10, maxval=-4)
            xs2 = jr.uniform(k2, shape=(n // 3,), minval=0, maxval=2)
            xs3 = jr.uniform(k3, shape=(n - 2 * (n // 3)), minval=6, maxval=10)
            xs = jnp.concatenate([xs1, xs2, xs3])
        elif x_design == "uniform-1d":
            xs = jr.uniform(key, shape=(n,), minval=-10, maxval=10)
        elif x_design == "uniform-2d":
            xs = jr.uniform(key, shape=(n, 2), minval=-10, maxval=10)
        else:
            raise ValueError(f"Unknown design='{x_design}'")
        if xs.ndim == 1:
            xs = xs[:, None]
        return np.array(xs)

    @abstractmethod
    def get_y(self, key, x) -> np.ndarray:
        pass

    @abstractmethod
    def get_true_event(self, x: np.ndarray, y_star: float | int) -> np.ndarray:
        """Return the true probability of event evaluated at x (1-D array)."""
        pass

    def visualise(self):
        # visualise data and true curve
        import matplotlib.pyplot as plt

        assert self.X.ndim == 2 and self.X.shape[1] == 1

        plt.figure(figsize=(8, 5))
        x = self.X[:, 0]
        plt.scatter(x, self.y, alpha=0.5, label="data")
        x_grid = np.linspace(x.min(), x.max(), 200, dtype=np.float32)[:, None]
        plt.title(f"{self.__class__.__name__} ({self.x_design} design)")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    def visualise_true_event(self):
        import matplotlib.pyplot as plt
        import utils
        from constants import Y_STAR_MAP

        assert self.X.ndim == 2 and self.X.shape[1] == 1

        setup_name = utils.camel_to_kebab(self.__class__.__name__)
        y_star = Y_STAR_MAP[setup_name]
        plt.figure(figsize=(8, 5))
        x = self.X[:, 0]
        x_grid = np.linspace(x.min(), x.max(), 200, dtype=np.float32)[:, None]
        plt.plot(x_grid, self.get_true_event(x_grid, y_star), label="true event")
        plt.title(f"{setup_name} ({self.x_design} design)")
        plt.xlabel("X")
        plt.ylabel(f"P(Y <= {y_star} | X)")
        plt.ylim(-0.01, 1.01)
        plt.legend()
        plt.grid()
        plt.show()

class GaussianLinear(Data):

    def _param(self, x):
        a = 0.2
        b = 0.0
        mean = (a * x + b).astype(np.float32)
        noise_std = 1.0
        return mean, noise_std

    def get_y(self, key, x):
        # linear function plus constant Gaussian noise
        assert x.ndim == 2 and x.shape[1] == 1
        mean, noise_std = self._param(x.ravel())
        y = mean + jr.normal(key, shape=x.ravel().shape) * noise_std
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: float) -> np.ndarray:
        # return the P(Y <= y_star | x) of Gaussian at mean=true_curve(x), sd=0.5
        mean, noise_std = self._param(x)
        cdf = norm.cdf(y_star, loc=mean, scale=noise_std)
        return cdf.astype(np.float32)


class GaussianPolynomial(Data):

    def _param(self, x):
        # Polynomial passing through (-10, -2), (0, 1), (10, -2)
        mean = (1.0 - 0.03 * x**2).astype(np.float32)
        noise_std = 1.0
        return mean, noise_std

    def get_y(self, key, x):
        # linear function plus constant Gaussian noise
        assert x.ndim == 2 and x.shape[1] == 1
        mean, noise_std = self._param(x.ravel())
        y = mean + jr.normal(key, shape=x.ravel().shape) * noise_std
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: float) -> np.ndarray:
        # return the P(Y <= y_star | x) of Gaussian at mean=true_curve(x), sd=0.5
        mean, noise_std = self._param(x)
        cdf = norm.cdf(y_star, loc=mean, scale=noise_std)
        return cdf.astype(np.float32)


class GaussianLinearDependentError(Data):

    def _params(self, x: np.ndarray):
        a = 0.5
        b = 1.0
        mean = (a * x + b).astype(np.float32)
        # Higher |x| -> larger noise variance
        noise_std = 0.5 + 0.5 * np.abs(x)
        return mean, noise_std

    def get_y(self, key, x):
        mean, noise_std = self._params(x.ravel())

        y = mean + jr.normal(key, shape=x.ravel().shape) * noise_std
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: float) -> np.ndarray:
        # return the P(Y <= y_star | x) of Gaussian at mean=true_curve(x), sd=noise_std(x)
        mean, noise_std = self._params(x.ravel())
        cdf = norm.cdf(y_star, loc=mean, scale=noise_std)
        return cdf.astype(np.float32)


class GammaLinear(Data):
    # good y^*: between 4 and 5

    def _params(self, x: np.ndarray):
        """Compute Gamma distribution parameters with linear mean."""
        a = 0.12
        b = 4.0
        mean = a * x + b  # mean > 0 for x in [-10,10]
        # Shape increases with |x|
        shape = 8.0 + 2.0 * np.abs(x)
        scale = mean / shape
        return shape.astype(np.float32), scale.astype(np.float32)

    def get_y(self, key, x):
        # Sample directly from Gamma(shape(x), scale(x)) with linear mean.
        assert x.ndim == 2 and x.shape[1] == 1
        shape, scale = self._params(x.ravel())
        assert shape.shape == scale.shape
        y = jr.gamma(key, shape) * scale
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: float) -> np.ndarray:
        # return the P(Y <= y_star | x) of Gamma at shape(x), scale(x)
        assert x.ndim == 2 and x.shape[1] == 1
        shape, scale = self._params(x.ravel())
        cdf = gamma.cdf(y_star, a=shape, scale=scale)
        return cdf.astype(np.float32)


class GaussianSine(Data):

    def _params(self, x: np.ndarray):
        # sine function
        a = 0.5
        b = 0.0
        mean = a * np.sin(x / 2) + b
        noise_std = 0.5
        return mean, noise_std

    def get_y(self, key, x):
        # sine function plus constant Gaussian noise
        # true curve evaluated at observed X
        assert x.ndim == 2 and x.shape[1] == 1
        mean, noise_std = self._params(x.ravel())
        y = mean + jr.normal(key, shape=x.ravel().shape) * noise_std
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: float) -> np.ndarray:
        # return the P(Y <= y_star | x) of Gaussian at mean=true_curve(x), sd=0.1
        mean, noise_std = self._params(x)
        cdf = norm.cdf(y_star, loc=mean, scale=noise_std)
        return cdf.astype(np.float32)


class PoissonLinear(Data):
    # good y_star = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        # parabola: y = a(x^2 - 80) + 5
        # choose a > 0 for positive opening, scaled appropriately
        a = 0.05
        rate = a * (x**2 - 80.0) + 5.0  # shift up to keep positive
        return rate.astype(np.float32)

    def get_y(self, key, x):
        # linear function plus poisson noise
        assert x.ndim == 2 and x.shape[1] == 1
        rate = self._params(x.ravel())
        # Poisson expects rate parameter (non-negative)
        y = jr.poisson(key, rate)
        return np.array(y).astype(np.int32)

    def get_true_event(self, x: np.ndarray, y_star: int) -> np.ndarray:
        # return the P(y <= y_star | x) of Poisson at rate=true_curve(x)
        rate = self._params(x.ravel())
        cdf = poisson.cdf(y_star, rate)
        return cdf.astype(np.float32)


class ProbitMixture(Data):
    # good y_star = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        p = 0.6 * norm.cdf((x - 8.0) / 4.0) + 0.4 * norm.cdf((x + 8.0) / 4.0)
        return p.astype(np.float32)

    def get_y(self, key, x):
        # mixture probit: probability as true curve at observed X
        assert x.ndim == 2 and x.shape[1] == 1
        p = self._params(x.ravel())
        y = (jr.uniform(key, shape=x.ravel().shape) < p).astype(np.int32)
        return np.array(y)

    def get_true_event(self, x: np.ndarray, y_star: int) -> np.ndarray:
        # return the P(Y = y_star | x) of mixture probit at observed X
        p = self._params(x.ravel())
        if y_star == 1:
            return p.astype(np.float32)
        else:
            return (1.0 - p).astype(np.float32)


class CategoricalLinear(Data):
    # four categories: 0,1,2,3
    # good y_star = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        # x is shape (n,)
        # logits: (n, 4)
        n = x.shape[0]
        logits = np.zeros((n, 4), dtype=np.float32)

        # Class 0: mostly in [-10, 0] -> center -5
        logits[:, 0] = -1.0 * (x + 5.0) ** 2 / 10.0
        # Class 1: mostly in [-5, 5] -> center 0
        logits[:, 1] = -1.0 * (x) ** 2 / 30.0
        # Class 2: mostly in [4, 10] -> center 7
        logits[:, 2] = -1.0 * (x - 7.0) ** 2 / 5.0
        # Class 3: mostly in [0, 8] -> center 4
        logits[:, 3] = -1.0 * (x - 4.0) ** 2 / 8.0

        # Softmax
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def get_y(self, key, x):
        # function to get logits, then softmax to get probabilities
        assert x.ndim == 2 and x.shape[1] == 1
        probs = self._params(x.ravel())
        n = probs.shape[0]

        # Sample
        def sample_row(subkey, p):
            return jr.choice(subkey, a=probs.shape[1], p=p)

        keys = jr.split(key, n)
        y = jax.vmap(sample_row)(keys, probs)
        return np.array(y).astype(np.int32)

    def get_true_event(self, x: np.ndarray, y_star: int) -> np.ndarray:
        # return the P(y = y_star | x)
        probs = self._params(x.ravel())
        return probs[:, y_star]


class Gamma(Data):
    # good y_star = 2.5

    def get_y(self, key, x):
        # Gamma(shape=2, scale=2) rv independent of x
        y = jr.gamma(key, 2, shape=(x.shape[0],)) * 2
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: int) -> np.ndarray:
        # return the P(y <= y_star | x) of Gamma(shape=2, scale=2)
        cdf = gamma.cdf(y_star, a=2, scale=2)
        return np.full(x.shape[0], cdf, dtype=np.float32)


class LogisticLinear(Data):
    # good y_star = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        # P(Y=1|X) = sigmoid(a*x + b)
        a = 0.2
        b = -0.4
        logits = a * x + b
        p = 1.0 / (1.0 + np.exp(-logits))
        return p.astype(np.float32)

    def get_y(self, key, x):
        assert x.ndim == 2 and x.shape[1] == 1
        p = self._params(x.ravel())
        y = (jr.uniform(key, shape=x.ravel().shape) < p).astype(np.int32)
        return np.array(y)

    def get_true_event(self, x: np.ndarray, y_star: int) -> np.ndarray:
        p = self._params(x.ravel())
        if y_star == 1:
            return p
        else:
            return 1.0 - p


# %%


# def load_syn_linear_gaussian_regression(
#     n,
#     *,
#     a=0.5,
#     b=1.0,
#     sigma=0.30,
#     m=50,
#     y_star=3.0,
#     design="iid",  # "iid" | "gap" | "sparse_band"
#     gap=(4.0, 6.0),  # x-region to thin/remove
#     sparse_keep_prob=0.15,  # prob to keep a point inside gap
#     oversample_factor=3,
#     rng=np.random.default_rng(),
# ):
#     """
#     Synthetic linear–Gaussian regression:
#       Y = a X + b + eps,   eps ~ N(0, sigma^2)

#     design:
#       - "iid":         x ~ Uniform(0,10), as in original.
#       - "gap":         no training points with x in (gap[0], gap[1]).
#       - "sparse_band": training points in gap kept with prob << 1.

#     Returns:
#       X (n,1), y (n,), x_grid (m,1), true_curve (m,), title (str).
#     """
#     lo, hi = gap
#     assert hi > lo, "gap must satisfy hi > lo"

#     # Use seperate RNGs for x and y to ensure reproducibility
#     rng_x, rng_y = rng.spawn(2)

#     def _base_sample(rng, k):
#         return rng.uniform(0.0, 10.0, k).astype(np.float32)

#     if design == "iid":
#         x = _base_sample(rng_x, n)

#     elif design == "gap":
#         xs = []
#         need = n
#         while need > 0:
#             prop = _base_sample(rng_x, max(need * oversample_factor, need))
#             keep = (prop <= lo) | (prop >= hi)
#             kept = prop[keep]
#             xs.append(kept[:need])
#             need -= kept[:need].size
#         x = np.concatenate(xs).astype(np.float32)

#     elif design == "sparse_band":
#         xs = []
#         need = n
#         p = float(sparse_keep_prob)
#         while need > 0:
#             # oversample because we will discard a bunch in the band
#             prop = _base_sample(
#                 rng_x, max(int(need / max(p, 1e-3)) * oversample_factor, need)
#             )
#             in_gap = (prop > lo) & (prop < hi)
#             keep = (~in_gap) | (rng_x.rand(prop.size) < p)
#             kept = prop[keep]
#             xs.append(kept[:need])
#             need -= kept[:need].size
#         x = np.concatenate(xs).astype(np.float32)

#     else:
#         raise ValueError(f"Unknown design='{design}'")

#     # response variable
#     y = (a * x + b + rng_y.normal(0.0, sigma, x.size)).astype(np.float32)

#     # grid + true curve: P(Y <= y_star | X=x) = Phi((y* - (a x + b))/sigma)
#     x_grid = np.linspace(0.0, 10.0, m, dtype=np.float32)[:, None]
#     true_curve = norm.cdf((y_star - (a * x_grid.ravel() + b)) / sigma).astype(
#         np.float32
#     )

#     # label
#     title_flavor = {
#         "iid": "iid",
#         "gap": f"gap {gap}",
#         "sparse_band": f"sparse {gap}, p={sparse_keep_prob}",
#     }
#     title = f"Synthetic linear–Gaussian (y*={y_star}, {title_flavor[design]})"

#     return x[:, None], y, x_grid, true_curve, title


# def load_syn_mixture_probit(
#     n,
#     *,
#     m=50,
#     design="iid",  # "iid" | "gap" | "sparse_band"
#     gap=(5.0, 7.0),  # region to thin/remove (lo, hi)
#     sparse_keep_prob=0.15,  # used when design="sparse_band"
#     oversample_factor=3,
#     rng=np.random.default_rng(),
# ):  # controls efficiency of thinning
#     """
#     Synthetic mixture–probit (binary):
#       X ~ mixture of N(5,1) and N(9,1)  (base proposal)
#       P(Y=1|X=x) = 0.7 Phi((x-5)/1) + 0.3 Phi((x-9)/1)

#     design:
#       - "iid":         no modification (original behavior).
#       - "gap":         *no* training points with x in (gap[0], gap[1]).
#       - "sparse_band": training points in gap are *thinned* with keep prob p<<1.

#     Uses global numpy RNG state (seed it outside if you want reproducibility).
#     Returns: X (n,1), y (n,), x_grid (m,1), true_curve (m,), title (str)
#     """
#     lo, hi = gap
#     assert hi > lo, "gap must satisfy hi > lo"

#     # Use seperate RNGs for x and y to ensure reproducibility
#     rng_x, rng_y = rng.spawn(2)

#     def _base_sample(rng, k):
#         # proposal distribution for X: same as your original two-Gaussian mix
#         k1 = k // 2
#         k2 = k - k1
#         x = np.concatenate([rng.normal(5.0, 1.0, k1), rng.normal(9.0, 1.0, k2)])
#         # shuffle so stream isn't ordered by component
#         perm = rng.permutation(k)
#         return x[perm]

#     if design == "iid":
#         x = _base_sample(rng_x, n)

#     elif design == "gap":
#         # rejection sample: discard any x in (lo, hi) until we have n points
#         xs = []
#         need = n
#         while need > 0:
#             prop = _base_sample(rng_x, max(need * oversample_factor, need))
#             keep = (prop <= lo) | (prop >= hi)
#             kept = prop[keep]
#             xs.append(kept[:need])
#             need -= kept[:need].size
#         x = np.concatenate(xs).astype(np.float32)

#     elif design == "sparse_band":
#         # thinning: keep everything outside gap; inside gap keep with prob p
#         xs = []
#         need = n
#         p = float(sparse_keep_prob)
#         while need > 0:
#             prop = _base_sample(
#                 rng_x, max(int(need / max(p, 1e-3)) * oversample_factor, need)
#             )
#             in_gap = (prop > lo) & (prop < hi)
#             keep = (~in_gap) | (rng_x.rand(prop.size) < p)
#             kept = prop[keep]
#             xs.append(kept[:need])
#             need -= kept[:need].size
#         x = np.concatenate(xs).astype(np.float32)

#     else:
#         raise ValueError(
#             f"Unknown design='{design}'. Use 'iid', 'gap', or 'sparse_band'."
#         )

#     # labels from the same mixture–probit as your original
#     p = 0.7 * norm.cdf((x - 5.0) / 1.0) + 0.3 * norm.cdf((x - 9.0) / 1.0)
#     y = (rng_y.random(x.size) < p).astype(np.int32)

#     # grid + true curve (unchanged)
#     x_grid = np.linspace(0.0, 12.0, m, dtype=np.float32)[:, None]
#     true_curve = (
#         0.7 * norm.cdf((x_grid.ravel() - 5.0) / 1.0)
#         + 0.3 * norm.cdf((x_grid.ravel() - 9.0) / 1.0)
#     ).astype(np.float32)

#     # tidy shapes & title
#     title_flavor = {
#         "iid": "iid",
#         "gap": f"gap {gap}",
#         "sparse_band": f"sparse {gap}, p={sparse_keep_prob}",
#     }
#     title = f"Synthetic mixture-probit ({title_flavor[design]})"
#     return x[:, None], y, x_grid, true_curve, title
