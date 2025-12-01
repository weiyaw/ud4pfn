import data
import argparse
from loguru import logger
import sys

import numpy as np
import torch
from timeit import default_timer as timer

import utils
import credible_set
from credible_set import TabPFNClassifierPPD, TabPFNRegressorPPD

parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default="linreg")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_est", type=int, default=8)
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--mc_samples", type=int, default=1000)
parser.add_argument("--x_design", type=str, default="one_gap")
args = parser.parse_args()


n = args.n
m = 100  # size of x_grid
x_grid = np.linspace(-10, 10, m).reshape(-1, 1)
mc_samples = args.mc_samples
setup = args.setup  # linreg or probit
seed = args.seed
tabpfn_n_estimators = args.n_est
tabpfn_average_before_softmax = True

torch.manual_seed(8655 + seed)
rng = np.random.default_rng(1907 + seed)
rng_others, rng_setup = rng.spawn(2)


# convert setup (kebab-case) to Camalcase and ensure PPD suffix
setup_name = utils.kebab_to_camel(args.setup)
try:
    Setup = getattr(data, setup_name)
    setup = Setup(args.n, rng_setup, args.x_design)
except AttributeError:
    raise ValueError(f"Data {setup_name} not found in data module")


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
]

if args.setup == "gamma-linear":
    y_star = 4.0
elif args.setup in CLASSIFICATION:
    y_star = 1
else:
    y_star = 0.0

if args.setup in REGRESSION:
    savedir = f"outputs/coverage/setup={args.setup} y_star={y_star} x_design={args.x_design} n={n} m={m} n_est={tabpfn_n_estimators} seed={seed}"
    clf = TabPFNRegressorPPD(
        y_star=y_star,
        n_estimators=tabpfn_n_estimators,
        average_before_softmax=tabpfn_average_before_softmax,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2-regressor.ckpt",
    )
elif args.setup in CLASSIFICATION:
    savedir = f"outputs/coverage/setup={args.setup} y_star={y_star} x_design={args.x_design} n={n} m={m} n_est={tabpfn_n_estimators} seed={seed}"
    clf = TabPFNClassifierPPD(
        y_star=y_star,
        n_estimators=tabpfn_n_estimators,
        average_before_softmax=tabpfn_average_before_softmax,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2-classifier.ckpt",
    )
else:
    raise ValueError(f"Unknown data {data}")

X = setup.X
y = setup.y

logger.remove()  # remove default logger
log_format = "{time} - {level} - {message}"
logger.add(f"{savedir}/coverage.log", level="INFO", format=log_format)
logger.add(sys.stderr, level="INFO", format=log_format)
logger.info(f"Saving outputs to {savedir}")
logger.info(f"Git hash: {utils.githash()}")


utils.write_to_local(
    f"{savedir}/data.pickle",
    {"setup": setup, "x_grid": x_grid},
)

start = timer()
g0_to_gn = credible_set.compute_g0_to_gn(clf, x_grid, X, y)
utils.write_to_local(f"{savedir}/g0_to_gn.pickle", g0_to_gn)
logger.info(f"Built g0_to_gn in {timer() - start:.2f} seconds")

start = timer()
gn = credible_set.compute_gn(clf, x_grid, X, y)
utils.write_to_local(f"{savedir}/gn.pickle", gn)
logger.info(f"Built gn in {timer() - start:.2f} seconds")

start = timer()
gn_plus_1 = credible_set.sample_gn_plus_1(rng_others, clf, x_grid, X, y, size=mc_samples)
utils.write_to_local(f"{savedir}/gn_plus_1.pickle", gn_plus_1)
logger.info(f"Built gn_plus_1 in {timer() - start:.2f} seconds")
