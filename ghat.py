import data
import argparse
from loguru import logger
import sys

import numpy as np
import torch
from timeit import default_timer as timer

# from tabpfn import TabPFNClassifier
import utils
import forward
from forward import TabPFNClassifierPPD, TabPFNRegressorPPD

parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default="linreg")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_est", type=int, default=8)
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--mc_samples", type=int, default=1000)
args = parser.parse_args()


n = args.n
m = 100                         # size of x_grid
mc_samples = args.mc_samples
setup = args.setup              # linreg or probit
seed = args.seed
tabpfn_n_estimators = args.n_est
tabpfn_average_before_softmax = True

torch.manual_seed(8655 + seed)
rng = np.random.default_rng(1907 + seed)
rng_others, rng_data = rng.spawn(2)

if setup == "linreg":
    y_star = 3.0
    savedir = f"outputs/coverage/syn-linear-gaussian-regression y_star={y_star} n={n} m={m} n_est={tabpfn_n_estimators} seed={seed}"
    X, y, x_grid, true_curve, title = data.load_syn_linear_gaussian_regression(
        n=n, m=m, y_star=y_star, design="iid", rng=rng_data
    )
    clf = TabPFNRegressorPPD(
        n_estimators=tabpfn_n_estimators,
        average_before_softmax=tabpfn_average_before_softmax,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2-regressor.ckpt",
    )
elif setup == "probit":
    savedir = f"outputs/coverage/syn-mixture-probit n={n} m={m} n_est={tabpfn_n_estimators} seed={seed}"
    X, y, x_grid, true_curve, title = data.load_syn_mixture_probit(
        n=n, m=m, design="iid", rng=rng_data
    )
    clf = TabPFNClassifierPPD(
        n_estimators=tabpfn_n_estimators,
        average_before_softmax=True,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2-classifier.ckpt",
    )
else:
    raise ValueError(f"Unknown data {data}")


logger.remove()  # remove default logger
log_format = "{time} - {level} - {message}"
logger.add(f"{savedir}/coverage.log", level="INFO", format=log_format)
logger.add(sys.stderr, level="INFO", format=log_format)
logger.info(f"Saving outputs to {savedir}")
logger.info(f"Git hash: {utils.githash()}")


utils.write_to_local(
    f"{savedir}/data.pickle",
    {"X": X, "y": y, "x_grid": x_grid, "true_curve": true_curve, "title": title},
)

start = timer()
g0_to_gn = forward.compute_g0_to_gn(clf, x_grid, X, y)
utils.write_to_local(f"{savedir}/g0_to_gn.pickle", g0_to_gn)
logger.info(f"Built g0_to_gn in {timer() - start:.2f} seconds")

start = timer()
gn = forward.compute_gn(clf, x_grid, X, y)
utils.write_to_local(f"{savedir}/gn.pickle", gn)
logger.info(f"Built gn in {timer() - start:.2f} seconds")

start = timer()
gn_plus_1 = forward.sample_gn_plus_1(rng_others, clf, x_grid, X, y, size=mc_samples)
utils.write_to_local(f"{savedir}/gn_plus_1.pickle", gn_plus_1)
logger.info(f"Built gn_plus_1 in {timer() - start:.2f} seconds")
