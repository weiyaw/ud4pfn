# %%
from abc import abstractmethod
import math
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key, X=np.ndarray{self.X.shape}, y=np.ndarray{self.y.shape}, x_design='{self.x_design}')"

    @abstractmethod
    def get_true_event(self, x: np.ndarray, t: float | int) -> np.ndarray:
        """Return the true probability of event evaluated at x (1-D array)."""
        pass


class Data1D(Data):
    def __init__(self, key: jax.random.key, n: int, shuffle: bool, x_design: str):
        self.key = key
        self.x_design = x_design
        key_x, key_y, key_shuffle = jr.split(key, 3)
        self.X = np.asarray(self.get_x(key_x, n, x_design))
        # get y from subclass and compute true_curve separately via get_true_curve
        self.y = self.get_y(key_y, self.X)

        if shuffle:
            perm = jr.permutation(key_shuffle, n)
            self.X = self.X[perm]
            self.y = self.y[perm]

    def get_x(self, key: jax.random.key, n: int, x_design: str) -> jax.Array:
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
        elif x_design.startswith("uniform:"):
            minval, maxval = x_design.split(":")[1:]
            xs = jr.uniform(key, shape=(n,), minval=float(minval), maxval=float(maxval))
        else:
            raise ValueError(f"Unknown design='{x_design}'")
        if xs.ndim == 1:
            xs = xs[:, None]
        return xs

    @abstractmethod
    def get_y(self, key: jax.random.key, x: np.ndarray) -> np.ndarray:
        pass

    def visualise(self, figsize=(8, 5)) -> None:
        # visualise data
        import matplotlib.pyplot as plt

        assert self.X.shape[1] == 1
        x = self.X[:, 0]

        plt.figure(figsize=figsize)
        plt.scatter(x, self.y, alpha=0.5, label="data")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.title(f"{self.__class__.__name__} ({self.x_design} design)")
        plt.show()

    def visualise_true_event(self):
        # visualise true probability of event
        import matplotlib.pyplot as plt
        import utils
        from constants import T_MAP

        assert self.X.ndim == 2
        assert self.X.shape[1] == 1

        setup_name = utils.camel_to_kebab(self.__class__.__name__)
        t_values = T_MAP[setup_name]
        n_plots = len(t_values)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
        axes = axes.flatten()

        for ax, t in zip(axes, t_values):
            # if x is 1D
            x = self.X[:, 0]
            x_grid = np.linspace(x.min(), x.max(), 200, dtype=np.float32)[:, None]
            ax.plot(x_grid, self.get_true_event(x_grid, t), label="true event")
            ax.set_title(f"{setup_name} ({self.x_design} design)\nt={t}")
            ax.set_xlabel("X")
            ax.set_ylabel(f"P(Y <= {t} | X)")
            ax.set_ylim(-0.01, 1.01)
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()


class GaussianLinear(Data1D):

    def _param(self, x: np.ndarray):
        assert x.ndim == 2 and x.shape[1] == 1
        a = 0.2
        b = 0.0
        mean = (a * x + b).squeeze(-1)
        noise_std = 1.0
        assert mean.shape == (x.shape[0],)
        return mean, noise_std

    def get_y(self, key, x):
        # linear function plus constant Gaussian noise
        mean, noise_std = self._param(x)
        y = mean + jr.normal(key, shape=mean.shape) * noise_std
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, t: float) -> np.ndarray:
        # return the P(Y <= t | x) of Gaussian at mean=true_curve(x), sd=0.5
        mean, noise_std = self._param(x)
        cdf = norm.cdf(t, loc=mean, scale=noise_std)
        assert cdf.shape == (x.shape[0],)
        return cdf.astype(np.float32)


class GaussianPolynomial(Data1D):

    def _param(self, x):
        # Polynomial passing through (-10, -2), (0, 1), (10, -2)
        assert x.ndim == 2 and x.shape[1] == 1
        mean = (1.0 - 0.03 * x**2).squeeze(-1)
        noise_std = 1.0
        assert mean.shape == (x.shape[0],)
        return mean, noise_std

    def get_y(self, key, x):
        # linear function plus constant Gaussian noise
        mean, noise_std = self._param(x)
        y = mean + jr.normal(key, shape=mean.shape) * noise_std
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, t: float) -> np.ndarray:
        # return the P(Y <= t | x) of Gaussian at mean=true_curve(x), sd=0.5
        mean, noise_std = self._param(x)
        cdf = norm.cdf(t, loc=mean, scale=noise_std)
        assert cdf.shape == (x.shape[0],)
        return cdf.astype(np.float32)


class GaussianLinearDependentError(Data1D):

    def _params(self, x: np.ndarray):
        assert x.ndim == 2 and x.shape[1] == 1
        a = 0.5
        b = 1.0
        mean = (a * x + b).astype(np.float32).squeeze(-1)
        # Higher |x| -> larger noise variance
        noise_std = (0.5 + 0.5 * np.abs(x)).squeeze(-1)
        assert mean.shape == (x.shape[0],)
        assert noise_std.shape == (x.shape[0],)
        return mean, noise_std

    def get_y(self, key, x):
        mean, noise_std = self._params(x)
        # mean is (N,), noise_std is (N,)
        y = mean + jr.normal(key, shape=mean.shape) * noise_std
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, t: float) -> np.ndarray:
        # return the P(Y <= t | x) of Gaussian at mean=true_curve(x), sd=noise_std(x)
        mean, noise_std = self._params(x)
        cdf = norm.cdf(t, loc=mean, scale=noise_std)
        assert cdf.shape == (x.shape[0],)
        return cdf.astype(np.float32)


class GammaLinear(Data1D):
    # good y^*: between 4 and 5

    def _params(self, x: np.ndarray):
        """Compute Gamma distribution parameters with linear mean."""
        assert x.ndim == 2 and x.shape[1] == 1
        a = 0.12
        b = 4.0
        mean = a * x + b  # mean > 0 for x in [-10,10]
        # Shape increases with |x|
        shape = 8.0 + 2.0 * np.abs(x)
        scale = mean / shape

        shape = shape.astype(np.float32).squeeze(-1)
        scale = scale.astype(np.float32).squeeze(-1)

        assert shape.shape == (x.shape[0],)
        assert scale.shape == (x.shape[0],)
        return shape, scale

    def get_y(self, key, x):
        # Sample directly from Gamma(shape(x), scale(x)) with linear mean.
        shape, scale = self._params(x)
        assert shape.shape == scale.shape
        y = jr.gamma(key, shape) * scale
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, t: float) -> np.ndarray:
        # return the P(Y <= t | x) of Gamma at shape(x), scale(x)
        shape, scale = self._params(x)
        cdf = gamma.cdf(t, a=shape, scale=scale)
        assert cdf.shape == (x.shape[0],)
        return cdf.astype(np.float32)


class GaussianSine(Data1D):

    def _params(self, x: np.ndarray):
        # sine function
        assert x.ndim == 2 and x.shape[1] == 1
        a = 0.5
        b = 0.0
        mean = (a * np.sin(x / 2) + b).squeeze(-1)
        noise_std = 0.5
        assert mean.shape == (x.shape[0],)
        return mean, noise_std

    def get_y(self, key, x):
        # sine function plus constant Gaussian noise
        # true curve evaluated at observed X
        assert x.ndim == 2 and x.shape[1] == 1
        mean, noise_std = self._params(x)
        y = mean + jr.normal(key, shape=mean.shape) * noise_std
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, t: float) -> np.ndarray:
        # return the P(Y <= t | x) of Gaussian at mean=true_curve(x), sd=0.1
        mean, noise_std = self._params(x)
        cdf = norm.cdf(t, loc=mean, scale=noise_std)
        assert cdf.shape == (x.shape[0],)
        return cdf.astype(np.float32)


class PoissonLinear(Data1D):
    # good t = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        # parabola: y = a(x^2 - 80) + 5
        # choose a > 0 for positive opening, scaled appropriately
        assert x.ndim == 2 and x.shape[1] == 1
        a = 0.05
        rate = a * (x**2 - 80.0) + 5.0  # shift up to keep positive
        rate = rate.astype(np.float32).squeeze(-1)
        assert rate.shape == (x.shape[0],)
        return rate

    def get_y(self, key, x):
        # linear function plus poisson noise
        assert x.ndim == 2 and x.shape[1] == 1
        rate = self._params(x)
        # Poisson expects rate parameter (non-negative)
        y = jr.poisson(key, rate)
        return np.array(y).astype(np.int32)

    def get_true_event(self, x: np.ndarray, t: int) -> np.ndarray:
        # return the P(y <= t | x) of Poisson at rate=true_curve(x)
        rate = self._params(x)
        cdf = poisson.cdf(t, rate)
        assert cdf.shape == (x.shape[0],)
        return cdf.astype(np.float32)


class ProbitMixture(Data1D):
    # good t = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 and x.shape[1] == 1
        p = 0.6 * norm.cdf((x - 8.0) / 4.0) + 0.4 * norm.cdf((x + 8.0) / 4.0)
        p = p.astype(np.float32).squeeze(-1)
        assert p.shape == (x.shape[0],)
        return p

    def get_y(self, key, x):
        # mixture probit: probability as true curve at observed X
        assert x.ndim == 2 and x.shape[1] == 1
        p = self._params(x)
        y = (jr.uniform(key, shape=p.shape) < p).astype(np.int32)
        return np.array(y)

    def get_true_event(self, x: np.ndarray, t: int) -> np.ndarray:
        # return the P(Y = t | x) of mixture probit at observed X
        p = self._params(x)
        ret = t * p + (1 - t) * (1 - p)
        assert ret.shape == (x.shape[0],)
        return ret.astype(np.float32)


class CategoricalLinear(Data1D):
    # four categories: 0,1,2,3
    # good t = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        # x is shape (n, 1)
        # logits: (n, 4)
        assert x.ndim == 2 and x.shape[1] == 1
        n = x.shape[0]
        logits = np.zeros((n, 4), dtype=np.float32)

        # We need to squeeze only for assignment to logits[:, k] which expects (n,)
        x = x.squeeze(-1)

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
        assert probs.shape == (n, 4)
        return probs

    def get_y(self, key, x):
        # function to get logits, then softmax to get probabilities
        assert x.ndim == 2 and x.shape[1] == 1
        probs = self._params(x)
        n = probs.shape[0]

        # Sample
        def sample_row(subkey, p):
            return jr.choice(subkey, a=probs.shape[1], p=p)

        keys = jr.split(key, n)
        y = jax.vmap(sample_row)(keys, probs)
        return np.array(y).astype(np.int32)

    def get_true_event(self, x: np.ndarray, t: int) -> np.ndarray:
        # return the P(y = t | x)
        probs = self._params(x)
        ret = probs[:, t]
        assert ret.shape == (x.shape[0],)
        return ret


class Gamma(Data1D):
    # good t = 2.5

    def get_y(self, key, x):
        # Gamma(shape=2, scale=2) rv independent of x
        y = jr.gamma(key, 2, shape=(x.shape[0],)) * 2
        return np.array(y).astype(np.float32)

    def get_true_event(self, x: np.ndarray, t: int) -> np.ndarray:
        # return the P(y <= t | x) of Gamma(shape=2, scale=2)
        cdf = gamma.cdf(t, a=2, scale=2)
        ret = np.full(x.shape[0], cdf, dtype=np.float32)
        assert ret.shape == (x.shape[0],)
        return ret


class LogisticLinear(Data1D):
    # good t = 1

    def _params(self, x: np.ndarray) -> np.ndarray:
        # P(Y=1|X) = sigmoid(a*x + b)
        assert x.ndim == 2 and x.shape[1] == 1
        a = 0.2
        b = -0.4
        logits = a * x + b
        p = 1.0 / (1.0 + np.exp(-logits))
        p = p.astype(np.float32).squeeze(-1)
        assert p.shape == (x.shape[0],)
        return p

    def get_y(self, key, x):
        assert x.ndim == 2 and x.shape[1] == 1
        p = self._params(x)
        y = (jr.uniform(key, shape=p.shape) < p).astype(np.int32)
        return np.array(y)

    def get_true_event(self, x: np.ndarray, t: int) -> np.ndarray:
        p = self._params(x)
        ret = t * p + (1 - t) * (1 - p)
        assert ret.shape == (x.shape[0],)
        return ret


class Data2D(Data):

    def __init__(
        self, key: jax.random.key, n: int, shuffle: bool, x_design: str = None
    ):
        self.key = key
        self.x_design = x_design
        key_data, key_shuffle = jr.split(key)
        self.X, self.y = self.get_x_and_y(key_data, n)

        if shuffle:
            perm = jr.permutation(key_shuffle, n)
            self.X = self.X[perm]
            self.y = self.y[perm]

    @abstractmethod
    def get_x_and_y(self, key: jax.random.key, n: int) -> tuple[np.ndarray, np.ndarray]:
        pass

    def visualise(self, figsize=(8, 5)) -> None:
        # visualise data and true curve
        import matplotlib.pyplot as plt

        assert self.X.ndim == 2
        assert self.X.shape[1] == 2

        plt.figure(figsize=figsize)

        x1 = self.X[:, 0]
        x2 = self.X[:, 1]
        sc = plt.scatter(x1, x2, c=self.y, alpha=0.5, label="data")
        plt.colorbar(sc, label="y")
        plt.xlabel("X1")
        plt.ylabel("X2")

        plt.legend()
        plt.grid()
        plt.title(f"{self.__class__.__name__} ({self.x_design} design)")
        plt.show()

    def visualise_true_event(self) -> None:
        import matplotlib.pyplot as plt
        import utils
        from constants import T_MAP

        assert self.X.ndim == 2
        assert self.X.shape[1] == 2

        setup_name = utils.camel_to_kebab(self.__class__.__name__)
        t_values = T_MAP[setup_name]
        n_plots = len(t_values)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
        axes = axes.flatten()

        for ax, t in zip(axes, t_values):
            # if x is 2D, plot a heatmap of the probability
            x1 = self.X[:, 0]
            x2 = self.X[:, 1]
            x1_grid = np.linspace(x1.min(), x1.max(), 200, dtype=np.float32)
            x2_grid = np.linspace(x2.min(), x2.max(), 200, dtype=np.float32)
            x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
            x_grid = np.stack([x1_grid, x2_grid], axis=-1)
            ax.imshow(
                self.get_true_event(x_grid, t),
                extent=[x1.min(), x1.max(), x2.min(), x2.max()],
                aspect="auto",
                origin="lower",
            )
            ax.set_title(f"{setup_name} ({self.x_design} design)\nt={t}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()


class GaussianLinearSusan(Data2D):

    def _param(self, x: np.ndarray):
        assert x.ndim == 2 and x.shape[1] == 2
        alpha = 1.0
        beta = np.array([1.5, -0.8])
        mean = alpha + beta @ x.T
        noise_std = 0.7
        assert mean.shape == (x.shape[0],)
        return mean, noise_std

    def get_x_and_y(self, key, n):
        x = jr.uniform(key, shape=(n, 2), minval=0, maxval=1)
        # linear function plus constant Gaussian noise
        mean, noise_std = self._param(x)
        y = mean + jr.normal(key, shape=mean.shape) * noise_std
        return np.array(x), np.array(y).astype(float)

    def get_true_event(self, x: np.ndarray, t: float) -> np.ndarray:
        # return the P(Y <= t | x) of Gaussian at mean=true_curve(x), sd=0.5
        mean, noise_std = self._param(x)
        cdf = norm.cdf(t, loc=mean, scale=noise_std)
        assert cdf.shape == (x.shape[0],)
        return cdf.astype(float)


class TwoMoons(Data2D):
    def get_x_and_y(self, key, n):
        n_out = n // 2
        n_in = n - n_out

        outer_circ_x = np.cos(np.linspace(0, np.pi, n_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_out))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_in))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_in)) - 0.5

        X = np.vstack(
            [
                np.append(outer_circ_x, inner_circ_x),
                np.append(outer_circ_y, inner_circ_y),
            ]
        ).T
        y = np.hstack([np.zeros(n_out, dtype=np.intp), np.ones(n_in, dtype=np.intp)])

        noise_std = 0.1
        X += noise_std * jr.normal(key, shape=X.shape)

        return X.astype(float), y.astype(int)

    def get_true_event(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.full(x.shape[0], np.nan)


class Spiral(Data2D):
    def get_x_and_y(self, key, n):
        n_arms = 3
        turns = 2
        radius = 4.0
        noise = 0.1

        counts = [n // n_arms + (1 if i < n % n_arms else 0) for i in range(n_arms)]

        X_list, y_list = [], []
        for c, n_per_arms in enumerate(counts):
            key_arm = jr.fold_in(key, c)
            key_arm, subkey = jr.split(key_arm)
            t = jr.uniform(subkey, shape=(n_per_arms,))
            r = radius * t
            theta = 2.0 * math.pi * turns * t + (2.0 * math.pi * c / n_arms)

            key_arm, subkey = jr.split(key_arm)
            x1 = r * np.cos(theta) + jr.normal(subkey, shape=(n_per_arms,)) * noise
            key_arm, subkey = jr.split(key_arm)
            x2 = r * np.sin(theta) + jr.normal(subkey, shape=(n_per_arms,)) * noise

            X_list.append(np.c_[x1, x2])
            y_list.append(np.full(n_per_arms, c, dtype=int))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        return X.astype(float), y.astype(int)

    def get_true_event(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.full(x.shape[0], np.nan)


# %%
