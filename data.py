# %%
from abc import abstractmethod
import numpy as np
from scipy.stats import norm, poisson, gamma
from dataclasses import dataclass
from typing import Union


def load_syn_linear_gaussian_regression(
    n,
    *,
    a=0.5,
    b=1.0,
    sigma=0.30,
    m=50,
    y_star=3.0,
    design="iid",  # "iid" | "gap" | "sparse_band"
    gap=(4.0, 6.0),  # x-region to thin/remove
    sparse_keep_prob=0.15,  # prob to keep a point inside gap
    oversample_factor=3,
    rng=np.random.default_rng(),
):
    """
    Synthetic linear–Gaussian regression:
      Y = a X + b + eps,   eps ~ N(0, sigma^2)

    design:
      - "iid":         x ~ Uniform(0,10), as in original.
      - "gap":         no training points with x in (gap[0], gap[1]).
      - "sparse_band": training points in gap kept with prob << 1.

    Returns:
      X (n,1), y (n,), x_grid (m,1), true_curve (m,), title (str).
    """
    lo, hi = gap
    assert hi > lo, "gap must satisfy hi > lo"

    # Use seperate RNGs for x and y to ensure reproducibility
    rng_x, rng_y = rng.spawn(2)

    def _base_sample(rng, k):
        return rng.uniform(0.0, 10.0, k).astype(np.float32)

    if design == "iid":
        x = _base_sample(rng_x, n)

    elif design == "gap":
        xs = []
        need = n
        while need > 0:
            prop = _base_sample(rng_x, max(need * oversample_factor, need))
            keep = (prop <= lo) | (prop >= hi)
            kept = prop[keep]
            xs.append(kept[:need])
            need -= kept[:need].size
        x = np.concatenate(xs).astype(np.float32)

    elif design == "sparse_band":
        xs = []
        need = n
        p = float(sparse_keep_prob)
        while need > 0:
            # oversample because we will discard a bunch in the band
            prop = _base_sample(
                rng_x, max(int(need / max(p, 1e-3)) * oversample_factor, need)
            )
            in_gap = (prop > lo) & (prop < hi)
            keep = (~in_gap) | (rng_x.rand(prop.size) < p)
            kept = prop[keep]
            xs.append(kept[:need])
            need -= kept[:need].size
        x = np.concatenate(xs).astype(np.float32)

    else:
        raise ValueError(f"Unknown design='{design}'")

    # response variable
    y = (a * x + b + rng_y.normal(0.0, sigma, x.size)).astype(np.float32)

    # grid + true curve: P(Y <= y_star | X=x) = Phi((y* - (a x + b))/sigma)
    x_grid = np.linspace(0.0, 10.0, m, dtype=np.float32)[:, None]
    true_curve = norm.cdf((y_star - (a * x_grid.ravel() + b)) / sigma).astype(
        np.float32
    )

    # label
    title_flavor = {
        "iid": "iid",
        "gap": f"gap {gap}",
        "sparse_band": f"sparse {gap}, p={sparse_keep_prob}",
    }
    title = f"Synthetic linear–Gaussian (y*={y_star}, {title_flavor[design]})"

    return x[:, None], y, x_grid, true_curve, title


def load_syn_mixture_probit(
    n,
    *,
    m=50,
    design="iid",  # "iid" | "gap" | "sparse_band"
    gap=(5.0, 7.0),  # region to thin/remove (lo, hi)
    sparse_keep_prob=0.15,  # used when design="sparse_band"
    oversample_factor=3,
    rng=np.random.default_rng(),
):  # controls efficiency of thinning
    """
    Synthetic mixture–probit (binary):
      X ~ mixture of N(5,1) and N(9,1)  (base proposal)
      P(Y=1|X=x) = 0.7 Phi((x-5)/1) + 0.3 Phi((x-9)/1)

    design:
      - "iid":         no modification (original behavior).
      - "gap":         *no* training points with x in (gap[0], gap[1]).
      - "sparse_band": training points in gap are *thinned* with keep prob p<<1.

    Uses global numpy RNG state (seed it outside if you want reproducibility).
    Returns: X (n,1), y (n,), x_grid (m,1), true_curve (m,), title (str)
    """
    lo, hi = gap
    assert hi > lo, "gap must satisfy hi > lo"

    # Use seperate RNGs for x and y to ensure reproducibility
    rng_x, rng_y = rng.spawn(2)

    def _base_sample(rng, k):
        # proposal distribution for X: same as your original two-Gaussian mix
        k1 = k // 2
        k2 = k - k1
        x = np.concatenate([rng.normal(5.0, 1.0, k1), rng.normal(9.0, 1.0, k2)])
        # shuffle so stream isn't ordered by component
        perm = rng.permutation(k)
        return x[perm]

    if design == "iid":
        x = _base_sample(rng_x, n)

    elif design == "gap":
        # rejection sample: discard any x in (lo, hi) until we have n points
        xs = []
        need = n
        while need > 0:
            prop = _base_sample(rng_x, max(need * oversample_factor, need))
            keep = (prop <= lo) | (prop >= hi)
            kept = prop[keep]
            xs.append(kept[:need])
            need -= kept[:need].size
        x = np.concatenate(xs).astype(np.float32)

    elif design == "sparse_band":
        # thinning: keep everything outside gap; inside gap keep with prob p
        xs = []
        need = n
        p = float(sparse_keep_prob)
        while need > 0:
            prop = _base_sample(
                rng_x, max(int(need / max(p, 1e-3)) * oversample_factor, need)
            )
            in_gap = (prop > lo) & (prop < hi)
            keep = (~in_gap) | (rng_x.rand(prop.size) < p)
            kept = prop[keep]
            xs.append(kept[:need])
            need -= kept[:need].size
        x = np.concatenate(xs).astype(np.float32)

    else:
        raise ValueError(
            f"Unknown design='{design}'. Use 'iid', 'gap', or 'sparse_band'."
        )

    # labels from the same mixture–probit as your original
    p = 0.7 * norm.cdf((x - 5.0) / 1.0) + 0.3 * norm.cdf((x - 9.0) / 1.0)
    y = (rng_y.random(x.size) < p).astype(np.int32)

    # grid + true curve (unchanged)
    x_grid = np.linspace(0.0, 12.0, m, dtype=np.float32)[:, None]
    true_curve = (
        0.7 * norm.cdf((x_grid.ravel() - 5.0) / 1.0)
        + 0.3 * norm.cdf((x_grid.ravel() - 9.0) / 1.0)
    ).astype(np.float32)

    # tidy shapes & title
    title_flavor = {
        "iid": "iid",
        "gap": f"gap {gap}",
        "sparse_band": f"sparse {gap}, p={sparse_keep_prob}",
    }
    title = f"Synthetic mixture-probit ({title_flavor[design]})"
    return x[:, None], y, x_grid, true_curve, title


@dataclass
class Data:
    rng: np.random.Generator
    X: np.ndarray
    y: np.ndarray
    x_design: str

    def __init__(self, n: int, rng: np.random.Generator, design: str = "one_gap"):
        self.rng = rng
        self.x_design = design
        rng_x, rng_y = rng.spawn(2)
        self.X = self.get_x(rng_x, n, design)
        # get y from subclass and compute true_curve separately via get_true_curve
        self.y = self.get_y(rng_y, self.X)

    def get_x(self, rng, n, design="one-gap") -> np.ndarray:
        # uniform between -10 and 10. no data in the gaps
        if design == "one-gap":
            xs1 = rng.uniform(-8, -2, n // 2)
            xs2 = rng.uniform(2, 8, n - n // 2)
            xs = np.concatenate([xs1, xs2])
        elif design == "two-gap":
            xs1 = rng.uniform(-10, -4, n // 3)
            xs2 = rng.uniform(0, 2, n // 3)
            xs3 = rng.uniform(6, 10, n - 2 * (n // 3))
            xs = np.concatenate([xs1, xs2, xs3])
        else:
            raise ValueError(f"Unknown design='{design}'")
        return xs[:, None]

    @abstractmethod
    def get_y(self, rng, x) -> np.ndarray:
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
        plt.scatter(self.X.ravel(), self.y, alpha=0.5, label="data")
        x_grid = np.linspace(-10, 10, 200, dtype=np.float32)[:, None]
        plt.title(f"{self.__class__.__name__} ({self.x_design} design)")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()


class GaussianLinear(Data):

    def _param(self, x):
        a = 2.0
        b = 1.0
        mean = (a * x + b).astype(np.float32)
        noise_std = 1.0
        return mean, noise_std

    def get_y(self, rng, x):
        # linear function plus constant Gaussian noise
        assert x.ndim == 2 and x.shape[1] == 1
        mean, noise_std = self._param(x.ravel())
        y = mean + rng.normal(0.0, noise_std, size=x.shape[0])
        return y.astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: float) -> np.ndarray:
        # return the P(Y <= y_star | x) of Gaussian at mean=true_curve(x), sd=0.5
        mean, noise_std = self._param(x)
        cdf = norm.cdf(y_star, loc=mean, scale=noise_std)
        return cdf.astype(np.float32)


class GaussianPolynomial(Data):

    def _param(self, x):
        # Polynomial passing through (-10, -2), (0, 2), (10, -2)
        mean = (2.0 - 0.04 * x**2).astype(np.float32)
        noise_std = 1.0
        return mean, noise_std

    def get_y(self, rng, x):
        # linear function plus constant Gaussian noise
        assert x.ndim == 2 and x.shape[1] == 1
        mean, noise_std = self._param(x.ravel())
        y = mean + rng.normal(0.0, noise_std, size=x.shape[0])
        return y.astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: float) -> np.ndarray:
        # return the P(Y <= y_star | x) of Gaussian at mean=true_curve(x), sd=0.5
        mean, noise_std = self._param(x)
        cdf = norm.cdf(y_star, loc=mean, scale=noise_std)
        return cdf.astype(np.float32)


class GaussianLinearDependentError(Data):

    def _params(self, x: np.ndarray):
        a = 2.0
        b = 1.0
        mean = (a * x + b).astype(np.float32)
        # Higher |x| -> larger noise variance
        noise_std = 0.1 + 0.5 * np.abs(x)
        return mean, noise_std

    def get_y(self, rng, x):
        mean, noise_std = self._params(x.ravel())

        y = rng.normal(mean, noise_std, size=x.shape[0])
        return y.astype(np.float32)

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

    def get_y(self, rng, x):
        # Sample directly from Gamma(shape(x), scale(x)) with linear mean.
        assert x.ndim == 2 and x.shape[1] == 1
        shape, scale = self._params(x.ravel())
        assert shape.shape == scale.shape
        y = rng.gamma(shape, scale)
        return y.astype(np.float32)

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
        noise_std = 0.1
        return mean, noise_std

    def get_y(self, rng, x):
        # sine function plus constant Gaussian noise
        # true curve evaluated at observed X
        assert x.ndim == 2 and x.shape[1] == 1
        mean, noise_std = self._params(x.ravel())
        y = mean + rng.normal(0.0, noise_std, x.shape[0])
        return y.astype(np.float32)

    def get_true_event(self, x: np.ndarray, y_star: float) -> np.ndarray:
        # return the P(Y <= y_star | x) of Gaussian at mean=true_curve(x), sd=0.1
        mean, noise_std = self._params(x)
        cdf = norm.cdf(y_star, loc=mean, scale=noise_std)
        return cdf.astype(np.float32)


class ProbitMixture(Data):
    # good y_star = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        p = 0.6 * norm.cdf((x - 3.0) / 1.0) + 0.4 * norm.cdf((x + 3.0) / 1.0)
        return p.astype(np.float32)

    def get_y(self, rng, x):
        # mixture probit: probability as true curve at observed X
        assert x.ndim == 2 and x.shape[1] == 1
        p = self._params(x.ravel())
        y = (rng.random(x.shape[0]) < p).astype(np.int32)
        return y

    def get_true_event(self, x: np.ndarray, y_star: int) -> np.ndarray:
        # return the P(Y = y_star | x) of mixture probit at observed X
        p = self._params(x.ravel())
        if y_star == 1:
            return p.astype(np.float32)
        else:
            return (1.0 - p).astype(np.float32)


class PoissonLinear(Data):
    # good y_star = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        # parabola: y = a(x+10)(x-10) = a(x^2 - 100)
        # choose a > 0 for positive opening, scaled appropriately
        a = 0.05
        rate = a * (x**2 - 100.0) + 5.0  # shift up to keep positive
        return rate.astype(np.float32)

    def get_y(self, rng, x):
        # linear function plus poisson noise
        assert x.ndim == 2 and x.shape[1] == 1
        rate = self._params(x.ravel())
        # Poisson expects rate parameter (non-negative)
        y = rng.poisson(rate)
        return y.astype(np.int32)

    def get_true_event(self, x: np.ndarray, y_star: int) -> np.ndarray:
        # return the P(y = y_star | x) of Poisson at rate=true_curve(x)
        rate = self._params(x.ravel())
        pmf = poisson.pmf(y_star, rate)
        return pmf.astype(np.float32)


# %%


# %%
