REGRESSION = [
    "gaussian-linear",
    "gaussian-polynomial",
    "gaussian-linear-dependent-error",
    "gamma-linear",
    "gaussian-sine",
    "poisson-linear",
    "gamma"
]

CLASSIFICATION = [
    "probit-mixture",
    "logistic-linear",
    "categorical-linear",
]

Y_STAR_MAP = {
    **{k: 0.0 for k in REGRESSION},
    **{k: 1 for k in CLASSIFICATION},
    "gamma-linear": 4.0,
    "gamma": 2.5,
    "poisson-linear": 1,
}
