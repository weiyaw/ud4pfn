REGRESSION = [
    "gaussian-linear",
    "gaussian-polynomial",
    "gaussian-linear-dependent-error",
    "gamma-linear",
    "gaussian-sine",
    "poisson-linear",
    "gamma",
    "gaussian-linear-susan",
]

CLASSIFICATION = [
    "probit-mixture",
    "logistic-linear",
    "categorical-linear",
    "two-moons",
    "spiral",
]

Y_STAR_MAP = {
    **{k: 0.0 for k in REGRESSION},
    **{k: 1 for k in CLASSIFICATION},
    "gamma-linear": 4.0,
    "gamma": 2.5,
    "poisson-linear": 1,
}

T_MAP = {
    **{k: [-2.0, -1.0, 0.0, 1.0, 2.0] for k in REGRESSION},
    **{k: [0, 1] for k in CLASSIFICATION},
    "gamma-linear": [1.0, 2.0, 3.0],
    "poisson-linear": [1.0, 2.0, 3.0],
    "gamma": [1.0, 2.0, 3.0],
}
