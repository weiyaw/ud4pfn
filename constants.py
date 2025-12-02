REGRESSION = [
    "gaussian-linear",
    "gaussian-polynomial",
    "gaussian-linear-dependent-error",
    "gamma-linear",
    "gaussian-sine",
]

CLASSIFICATION = [
    "probit-mixture",
    "poisson-linear",
    "logistic-linear",
]

Y_STAR_MAP = {
    **{k: 0.0 for k in REGRESSION},
    **{k: 1 for k in CLASSIFICATION},
    "gamma-linear": 4.0,
}