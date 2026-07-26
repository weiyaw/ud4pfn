import numpy as np

from experiments.entropic_ud.entropy import (
    compute_aleatoric_entropy_binary,
    compute_aleatoric_entropy_multiclass,
    compute_total_entropy_binary,
    compute_total_entropy_multiclass,
    match_gaussian_beta_moments,
    match_gaussian_dirichlet_moments,
)


def test_binary_entropy_shapes_and_ordering():
    mean = np.array([0.2, 0.5, 0.8])
    variance = np.array([0.01, 0.02, 0.01])
    a, b = match_gaussian_beta_moments(mean, variance)
    total = compute_total_entropy_binary(mean)
    aleatoric = compute_aleatoric_entropy_binary(mean, variance)

    assert a.shape == b.shape == total.shape == aleatoric.shape == mean.shape
    assert np.all(a > 0) and np.all(b > 0)
    assert np.all(aleatoric <= total + 1e-12)


def test_multiclass_entropy_shapes_and_ordering():
    mean = np.array([[0.6, 0.2], [0.3, 0.3], [0.1, 0.5]])
    variance = np.full_like(mean, 0.01)
    alpha = match_gaussian_dirichlet_moments(mean, variance)
    total = compute_total_entropy_multiclass(mean)
    aleatoric = compute_aleatoric_entropy_multiclass(mean, variance)

    assert alpha.shape == mean.shape
    assert total.shape == aleatoric.shape == (2,)
    assert np.all(alpha > 0)
    assert np.all(aleatoric <= total + 1e-12)
