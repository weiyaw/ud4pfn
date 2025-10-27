import numpy as np
from scipy.stats import norm


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
    y = (rng_y.rand(x.size) < p).astype(np.int32)

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
