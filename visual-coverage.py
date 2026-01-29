# %%
import argparse
import re

import sys

import numpy as np
from numpy.random import Generator

from scipy.stats import norm
import torch
from timeit import default_timer as timer
import utils
from metrics import (
    build_pointwise_band,
    build_simultaneous_band,
    compute_pointwise_coverage,
    compute_simultaneous_coverage,
)
import metrics
import posterior
from posterior import compute_vn, compute_un
import os

import pandas as pd
import matplotlib.pyplot as plt

from constants import Y_STAR_MAP

# %load_ext autoreload
# %autoreload 2

# %%

np.set_printoptions(formatter={"float_kind": "{:.2e}".format}, linewidth=np.inf)
pd.set_option("display.max_rows", None)
id_dir = "../outputs/2026-01-22/"
image_dir = "../paper/images"



## COVERAGE
# Group by setup
def group_by_setup(all_dirs: list[str]) -> dict[str, list[str]]:
    """Group directories by setup name."""
    setup_to_dirs = {}
    for d in all_dirs:
        match = re.search(r"setup=([^\s]+)", d)
        if match:
            setup = match.group(1)
        if setup not in setup_to_dirs:
            setup_to_dirs[setup] = []
        setup_to_dirs[setup].append(d)
    return setup_to_dirs


def get_metrics(rep_dirs: list[str], data, t_idx):
    # given a setup, i.e. all the repetition directories of a setup, its
    # auxilary data and t_idx, produce a row of all coverage metrics
    gn_all_rep_t = [utils.read_from(f"{outdir}/gn.pickle") for outdir in rep_dirs]
    gn_plus_1_all_rep_t = [
        utils.read_from(f"{outdir}/gn_plus_1.pickle") for outdir in rep_dirs
    ]
    g0_to_gn_all_rep_t = [
        utils.read_from(f"{outdir}/g0_to_gn.pickle") for outdir in rep_dirs
    ]
    assert len(gn_all_rep_t) == len(gn_plus_1_all_rep_t) == len(g0_to_gn_all_rep_t)

    true_prob_all_t = data["true_prob"]
    n = data["y_prev"].size

    true_prob = true_prob_all_t[t_idx]
    gn_all_rep = [g[t_idx] for g in gn_all_rep_t]
    g0_to_gn_all_rep = [g[:, t_idx] for g in g0_to_gn_all_rep_t]
    gn_plus_1_all_rep = [g[:, t_idx] for g in gn_plus_1_all_rep_t]

    # Construct the bands
    vn_simul_bands = []
    vn_point_bands = []
    un_simul_bands = []
    un_point_bands = []
    for g0_to_gn, gn, gn_plus_1 in zip(g0_to_gn_all_rep, gn_all_rep, gn_plus_1_all_rep):
        clt_cov = compute_vn(g0_to_gn, type="simultaneous") / n
        vn_simul_bands.append(build_simultaneous_band(gn, clt_cov))

        clt_cov = compute_vn(g0_to_gn, type="pointwise") / n
        vn_point_bands.append(build_pointwise_band(gn, clt_cov))

        clt_cov = compute_un(gn, gn_plus_1, n, type="simultaneous") / n
        un_simul_bands.append(build_simultaneous_band(gn, clt_cov))

        clt_cov = compute_un(gn, gn_plus_1, n, type="pointwise") / n
        un_point_bands.append(build_pointwise_band(gn, clt_cov))

    # Compute the coverage
    return {
        "vn_simul": compute_simultaneous_coverage(true_prob, vn_simul_bands),
        "vn_simul_width": np.mean([b["width"] for b in vn_simul_bands]),
        "vn_point": compute_pointwise_coverage(true_prob, vn_point_bands),
        "vn_point_width": np.mean([b["width"] for b in vn_point_bands]),
        "un_simul": compute_simultaneous_coverage(true_prob, un_simul_bands),
        "un_simul_width": np.mean([b["width"] for b in un_simul_bands]),
        "un_point": compute_pointwise_coverage(true_prob, un_point_bands),
        "un_point_width": np.mean([b["width"] for b in un_point_bands]),
    }


def get_df(all_dirs: list[str]) -> pd.DataFrame:
    df = []
    for setup_name, dirs in group_by_setup(all_dirs).items():
        data = utils.read_from(f"{dirs[0]}/data.pickle")
        df.append(
            {
                "setup": setup_name,
                "n": data["y_prev"].size,
                **get_metrics(dirs, data, t_idx=1),
            }
        )
    return pd.DataFrame(df)


# %%

# all_dirs = utils.get_matching_dirs(id_dir, r".+n=100 .+")
# all_dirs = utils.get_matching_dirs(id_dir, r".+n=200 .+")
# all_dirs = utils.get_matching_dirs(id_dir, r".+n=500 .+")
all_dirs = utils.get_matching_dirs(id_dir, r".+n=1000 .+")
t_idx = 1

df = []
for setup_name, dirs in group_by_setup(all_dirs).items():
    if setup_name != "probit-mixture": # DEBUG
        continue
    data = utils.read_from(f"{dirs[0]}/data.pickle")
    gn_all_rep_t = [utils.read_from(f"{outdir}/gn.pickle") for outdir in dirs]
    gn_plus_1_all_rep_t = [
        utils.read_from(f"{outdir}/gn_plus_1.pickle") for outdir in dirs
    ]
    g0_to_gn_all_rep_t = [
        utils.read_from(f"{outdir}/g0_to_gn.pickle") for outdir in dirs
    ]
    assert len(gn_all_rep_t) == len(gn_plus_1_all_rep_t) == len(g0_to_gn_all_rep_t)

    true_prob_all_t = data["true_prob"]
    n = data["y_prev"].size

    true_prob = true_prob_all_t[t_idx]
    gn_all_rep = [g[t_idx] for g in gn_all_rep_t]
    g0_to_gn_all_rep = [g[:, t_idx] for g in g0_to_gn_all_rep_t]
    gn_plus_1_all_rep = [g[:, t_idx] for g in gn_plus_1_all_rep_t]

    # Construct the bands
    vn_simul_bands = []
    vn_point_bands = []
    un_simul_bands = []
    un_point_bands = []
    for g0_to_gn, gn, gn_plus_1 in zip(g0_to_gn_all_rep, gn_all_rep, gn_plus_1_all_rep):
        # clt_cov = compute_vn(g0_to_gn, type="simultaneous") / n
        # vn_simul_bands.append(build_simultaneous_band(gn, clt_cov))

        # clt_cov = compute_vn(g0_to_gn, type="pointwise") / n
        # vn_point_bands.append(build_pointwise_band(gn, clt_cov))

        # clt_cov = compute_un(gn, gn_plus_1, n, type="simultaneous") / n
        # un_simul_bands.append(build_simultaneous_band(gn, clt_cov))

        # clt_cov = compute_un(gn, gn_plus_1, n, type="pointwise") / n
        # un_point_bands.append(build_pointwise_band(gn, clt_cov))

        print(np.mean(np.abs(gn_plus_1 - gn), axis=0)) # DEBUG

    # Compute the coverage
    df.append(
        {
            "setup": setup_name,
            "n": data["y_prev"].size,
            "vn_simul": compute_simultaneous_coverage(true_prob, vn_simul_bands),
            "vn_simul_width": np.mean([b["width"] for b in vn_simul_bands]),
            "vn_point": compute_pointwise_coverage(true_prob, vn_point_bands),
            "vn_point_width": np.mean([b["width"] for b in vn_point_bands]),
            "un_simul": compute_simultaneous_coverage(true_prob, un_simul_bands),
            "un_simul_width": np.mean([b["width"] for b in un_simul_bands]),
            "un_point": compute_pointwise_coverage(true_prob, un_point_bands),
            "un_point_width": np.mean([b["width"] for b in un_point_bands]),
        }
    )
df = pd.DataFrame(df)
# %%

df
