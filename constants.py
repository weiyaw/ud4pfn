REGRESSION = [
    "gaussian-linear",
    "gaussian-polynomial",
    "gaussian-linear-dependent-error",
    "gamma-linear",
    "gaussian-sine",
    "poisson-linear",
    "gamma",
    "gaussian-linear-susan",
    "gaussian-linear-multivariate",
    "gaussian-linear-dependent-error-multivariate",
    "poisson-linear-multivariate",
]

CLASSIFICATION = [
    "probit-mixture",
    "logistic-linear",
    "categorical-linear",
    "two-moons-1",
    "two-moons-2",
    "spiral",
    "fibre-strength",
    "labour-force",
    "probit-mixture-multivariate",
    "categorical-linear-multivariate",
]



T_MAP = {
    **{k: [-2.0, -1.0, 0.0, 1.0, 2.0] for k in REGRESSION},
    **{k: [0, 1] for k in CLASSIFICATION},
    "spiral": [0, 1, 2],
    "poisson-linear": [1.0, 2.0, 3.0],
    "poisson-linear-multivariate": [1.0, 2.0, 3.0],
    "gamma-linear": [1.0, 2.0, 3.0],
    "gamma": [1.0, 2.0, 3.0],
    "categorical-linear-multivariate": [0, 1, 2, 3],
}

# The corresponding index of T_MAP for each setup
DEFAULT_T_IDX = {
    **{k: 2 for k in REGRESSION},
    **{k: 1 for k in CLASSIFICATION},
    "spiral": 1,
    "poisson-linear": 1,
    "poisson-linear-multivariate": 1,
    "gamma-linear": 1,
    "gamma": 1,
}