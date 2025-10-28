import data
import argparse
from loguru import logger
import sys

import numpy as np
import torch
from timeit import default_timer as timer
from tabpfn import TabPFNClassifier
import utils
import forward

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_est", type=int, default=8)
args = parser.parse_args()


n = 100
m = 100
seed = args.seed
tabpfn_n_estimators = args.n_est
tabpfn_average_before_softmax = True
savedir = f"outputs/coverage/syn-mixture-probit n={n} m={m} seed={seed} n_est={tabpfn_n_estimators}"

logger.remove()  # remove default logger
log_format = "{time} - {level} - {message}"
logger.add(f"{savedir}/ghat.log", level="INFO", format=log_format)
logger.add(sys.stderr, level="INFO", format=log_format)
logger.info(f"Saving outputs to {savedir}")
logger.info(f"Git hash: {utils.githash()}")


torch.manual_seed(8655 + args.seed)
rng = np.random.default_rng(1907 + args.seed)
rng_others, rng_data = rng.spawn(2)

X, y, x_grid, true_curve, title = data.load_syn_mixture_probit(
    n=n, m=m, design="iid", rng=rng_data
)

utils.write_to_local(
    f"{savedir}/data-{args.seed}.pickle",
    {"X": X, "y": y, "x_grid": x_grid, "true_curve": true_curve, "title": title},
)

start = timer()
clf = TabPFNClassifier(
    n_estimators=1,
    average_before_softmax=True,
    softmax_temperature=1.0,
    fit_mode="low_memory",
)
g_hat = forward.build_g_hat_logreg(clf, X, y, x_grid)
logger.info(f"Built g_hat in {timer() - start:.2f} seconds")

utils.write_to_local(f"{savedir}/ghat-{args.seed}.pickle", g_hat)
