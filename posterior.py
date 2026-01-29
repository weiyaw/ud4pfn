import jax.random as jr
import numpy as np
from tqdm import trange

from pred_rule import TabPFNClassifierPPD, TabPFNRegressorPPD, assert_ppd_args_shape


def compute_gn(clf, t, x_grid, x_prev, y_prev):
    # Compute g_n = P(Y <= t | X=x_grid, x_prev, y_prev) or P(Y = t | X=x_grid, x_prev, y_prev)
    # t: (p,) array
    # x_grid: (m, d) array
    # x_prev: (n, d) array
    # y_prev: (n,) array
    # Return a 2d array of shape (p, m)
    assert_ppd_args_shape(x_grid, x_prev, y_prev)

    t = np.atleast_1d(t)
    m = x_grid.shape[0]

    # Guard against degenerate or low data
    if isinstance(clf, TabPFNClassifierPPD):
        if np.min(y_prev) == np.max(y_prev):
            # if all y_prev are the same, then g_n = 1 if t == y_prev[0], 0 otherwise
            probs = (t == y_prev[0]).astype(np.float32)
            return np.broadcast_to(probs[:, None], (t.shape[0], m))

    # Guard against degenerate or low data
    if isinstance(clf, TabPFNRegressorPPD):
        if y_prev.shape[0] < 2 or np.unique(y_prev).size < 2:
            # if all y_prev are the same, then g_n = 1 if t >= y_prev[0], 0 otherwise
            probs = (t >= y_prev[0]).astype(np.float32)
            return np.broadcast_to(probs[:, None], (t.shape[0], m))

    return clf.predict_event(t=t, x_new=x_grid, x_prev=x_prev, y_prev=y_prev)


def sample_gn_plus_1(key, clf, t, x_grid, x_prev, y_prev, size=100):
    # Draw P(Y_{n+2} <= t | X=x_grid, X_{n+1}, Y_{n+1}, x_prev, y_prev) for computing g_{n+1}. X_{n+1} is Bayesian bootstrap. Y_{n+1} is sampled from the PPD.
    # Return a 3d array of shape (mc_samples, p, m)
    assert_ppd_args_shape(x_grid, x_prev, y_prev)
    key_choice, key_sample = jr.split(key)
    idx = jr.choice(key_choice, x_prev.shape[0], shape=(size,), replace=True)
    x_plus_1 = x_prev[idx]
    y_plus_1, _ = clf.sample(
        key=key_sample, x_new=x_plus_1, x_prev=x_prev, y_prev=y_prev, size=1
    )
    y_plus_1 = np.array(y_plus_1).squeeze()

    assert x_plus_1.shape[0] == y_plus_1.shape[0]
    assert x_plus_1.shape[0] == size
    assert y_plus_1.ndim == 1
    prob_event = [
        clf.predict_event(
            t=t,
            x_new=x_grid,
            x_prev=np.vstack([x_prev, x_plus_1[i : i + 1]]),
            y_prev=np.hstack([y_prev, y_plus_1[i : i + 1]]),
        )
        for i in trange(size)
    ]
    return np.stack(prob_event, axis=0)


def compute_g0_to_gn(clf, t, x_grid, x_prev, y_prev):
    """
    Construct the sequence k ↦ v_k(x) mirroring build_g_hat_logreg,
    but leveraging compute_gn for probability evaluation.

    Parameters
    ----------
    clf : TabPFNClassifierPPD or TabPFNRegressorPPD
        Must expose `fit` and `predict_event`.
    t : (p,) array
        Events of the PPD.
    x_grid : (m, d) array
        Grid of covariate values.
    x_prev : (n, d) array
        Historical covariates.
    y_prev : (n,) array
        Historical binary labels.

    Returns
    -------
    g0_to_gn : (n+1, p, m) array
        If clf is TabPFNClassifierPPD, g0_to_gn[i, j, k] = P(Y = t[j] | X=x_grid[k], z_{1:i}).
        If clf is TabPFNRegressorPPD, g0_to_gn[i, j, k] = P(Y <= t[j] | X=x_grid[k], z_{1:i}).
        g0_to_gn[0, :, :] = NaN.
    """
    assert_ppd_args_shape(x_grid, x_prev, y_prev)

    n = x_prev.shape[0]
    m = x_grid.shape[0]
    g0_to_gn = np.empty((n + 1, t.shape[0], m), dtype=np.float32)
    g0_to_gn[0, :, :] = np.nan

    for i in trange(1, n + 1):
        g0_to_gn[i, :, :] = compute_gn(
            clf=clf, t=t, x_grid=x_grid, x_prev=x_prev[:i], y_prev=y_prev[:i]
        )

    return g0_to_gn


def compute_un(gn, gn_plus_1, n, type="simultaneous"):
    # Asymptotic CLT, Sec 4.3
    assert gn_plus_1.ndim == 2, "gn_plus_1 must be 2D array (mc_samples, m)"
    assert gn.ndim == 1, "gn must be 1D array (m,)"
    assert gn_plus_1.shape[1] == gn.shape[0], "gn_plus_1 and gn shape mismatch"

    ndiff = n * (gn_plus_1 - gn)  # (mc_samples, m)
    if type == "pointwise":
        return np.mean(ndiff**2, axis=0)  # (m,)
    elif type == "simultaneous":
        outer = np.einsum("ij,ik->ijk", ndiff, ndiff)
        return np.mean(outer, axis=0)  # (m, m)


def compute_vn(g0_to_gn, type="simultaneous"):
    # Asymptotic CLT, Sec 4.3, but starting from k = 2
    assert g0_to_gn.ndim == 2, "g0_to_gn must be 2D array (n+1, m)"

    n = g0_to_gn.shape[0] - 1
    diff = g0_to_gn[2:, :] - g0_to_gn[1:-1, :]  # (n-1, m)
    k = np.arange(2, n + 1)  # (n-1,)
    kdiff = k[:, None] * diff  # (n-1, m)

    if type == "pointwise":
        # v_n(x_j) = (1/(n-1)) * sum k^2 * Δ_k(x_j)^2
        return np.mean(kdiff**2, axis=0) # (m, )
    elif type == "simultaneous":
        # v_n(x) = (1/(n-1)) * sum k^2 * Δ_k(x) Δ_k(x)^T
        outer = np.einsum("ij,ik->ijk", kdiff, kdiff)  # (n-1, m, m)
        return np.mean(outer, axis=0)  # (m, m)


