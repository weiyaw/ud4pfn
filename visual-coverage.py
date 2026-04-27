# %%
import re
import os

import numpy as np
import pandas as pd
import tabulate
from tqdm.auto import tqdm

import utils
from constants import DEFAULT_T_IDX
from metrics import (
    build_bootstrap_pointwise_band,
    build_bootstrap_simultaneous_band,
    build_pointwise_band,
    build_simultaneous_band,
    compute_pointwise_coverage,
    compute_simultaneous_coverage,
)
from posterior import compute_un, compute_vn

# %load_ext autoreload
# %autoreload 2

# %%

np.set_printoptions(formatter={"float_kind": "{:.2e}".format}, linewidth=np.inf)
pd.set_option("display.max_rows", None)


# %%
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


def get_metrics(rep_dirs: list[str], data, t_idx, alpha=0.05):
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
    boot_simul_bands = []
    boot_point_bands = []
    copula_simul_bands = []
    copula_point_bands = []
    for g0_to_gn, gn, gn_plus_1 in zip(g0_to_gn_all_rep, gn_all_rep, gn_plus_1_all_rep):
        clt_cov = compute_vn(g0_to_gn, type="simultaneous") / n
        vn_simul_bands.append(build_simultaneous_band(gn, clt_cov, alpha=alpha))

        clt_cov = compute_vn(g0_to_gn, type="pointwise") / n
        vn_point_bands.append(build_pointwise_band(gn, clt_cov, alpha=alpha))

        clt_cov = compute_un(gn, gn_plus_1, n, type="simultaneous") / n
        un_simul_bands.append(build_simultaneous_band(gn, clt_cov, alpha=alpha))

        clt_cov = compute_un(gn, gn_plus_1, n, type="pointwise") / n
        un_point_bands.append(build_pointwise_band(gn, clt_cov, alpha=alpha))

    for outdir, gn in zip(rep_dirs, gn_all_rep):
        path = f"{outdir}/bootstrap-200.pickle"
        if not os.path.exists(path):
            break
            # raise RuntimeError(f"Missing bootstrap artifact: {path}")
        artifact = utils.read_from(path)
        bootstrap_preds_all_t = artifact["predictions"]
        bootstrap_preds = bootstrap_preds_all_t[:, t_idx, :]

        boot_point_bands.append(
            build_bootstrap_pointwise_band(gn, bootstrap_preds, alpha=alpha)
        )
        boot_simul_bands.append(
            build_bootstrap_simultaneous_band(gn, bootstrap_preds, alpha=alpha)
        )

    for outdir in rep_dirs:
        path = f"{outdir}/copula-200-1000.pickle"
        if not os.path.exists(path):
            break
        artifact = utils.read_from(path)
        cdf_samples = np.exp(artifact["logcdf"][:, t_idx, :])  # (B, x_grid)
        mean = cdf_samples.mean(axis=0)
        copula_point_bands.append(build_bootstrap_pointwise_band(mean, cdf_samples, alpha=alpha))
        copula_simul_bands.append(build_bootstrap_simultaneous_band(mean, cdf_samples, alpha=alpha))

    # Compute the coverage
    out = {
        "vn_simul": compute_simultaneous_coverage(true_prob, vn_simul_bands),
        "vn_simul_width": np.mean([b["width"] for b in vn_simul_bands]),
        "vn_point": compute_pointwise_coverage(true_prob, vn_point_bands),
        "vn_point_width": np.mean([b["width"] for b in vn_point_bands]),
        "un_simul": compute_simultaneous_coverage(true_prob, un_simul_bands),
        "un_simul_width": np.mean([b["width"] for b in un_simul_bands]),
        "un_point": compute_pointwise_coverage(true_prob, un_point_bands),
        "un_point_width": np.mean([b["width"] for b in un_point_bands]),
    }

    out |= {
        "boot_simul": compute_simultaneous_coverage(true_prob, boot_simul_bands),
        "boot_simul_width": np.mean([b["width"] for b in boot_simul_bands]),
        "boot_point": compute_pointwise_coverage(true_prob, boot_point_bands),
        "boot_point_width": np.mean([b["width"] for b in boot_point_bands]),
    }

    out |= {
        "copula_simul": compute_simultaneous_coverage(true_prob, copula_simul_bands),
        "copula_simul_width": np.mean([b["width"] for b in copula_simul_bands]) if copula_simul_bands else np.nan,
        "copula_point": compute_pointwise_coverage(true_prob, copula_point_bands),
        "copula_point_width": np.mean([b["width"] for b in copula_point_bands]) if copula_point_bands else np.nan,
    }

    return out


def get_df(all_dirs: list[str], alpha: float = 0.05) -> pd.DataFrame:
    df = []
    for setup_name, dirs in tqdm(group_by_setup(all_dirs).items()):
        data = utils.read_from(f"{dirs[0]}/data.pickle")
        df.append(
            {
                "setup": setup_name,
                "n": data["y_prev"].size,
                **get_metrics(
                    dirs,
                    data,
                    t_idx=DEFAULT_T_IDX[setup_name],
                    alpha=alpha,
                ),
            }
        )
    return pd.DataFrame(df)


# %%
# This is n_est=64
# Scalar x
id_dir = "../outputs/2026-01-22/" # (coverage for scalar x)

dfs05 = []
dfs20 = []
for n in [200, 500, 1000]:
    all_dirs = utils.get_matching_dirs(id_dir, rf".+n={n} .+")
    dfs05.append(get_df(all_dirs, alpha=0.05))
    dfs20.append(get_df(all_dirs, alpha=0.20))

dfs05 = pd.concat(dfs05).sort_values(by=["setup", "n"]).reset_index(drop=True)
print(dfs05.to_markdown(index=False))
dfs20 = pd.concat(dfs20).sort_values(by=["setup", "n"]).reset_index(drop=True)
print(dfs20.to_markdown(index=False))



# %%
# This is n_est=16
id_dir = "../outputs/2026-01-23/" # (coverage for multivariate x)

dfs05 = []
dfs20 = []
for n in [200, 500, 1000]:
    all_dirs = utils.get_matching_dirs(id_dir, rf".+n={n} .+")
    dfs05.append(get_df(all_dirs, alpha=0.05))
    dfs20.append(get_df(all_dirs, alpha=0.20))

dfs05 = pd.concat(dfs05).sort_values(by=["setup", "n"]).reset_index(drop=True)
dfs20 = pd.concat(dfs20).sort_values(by=["setup", "n"]).reset_index(drop=True)
# %%
print(dfs05.to_markdown(index=False))
print(dfs20.to_markdown(index=False))


# %%

print(dfs05[dfs05["setup"].str.contains("multivariate", na=False)].reset_index(drop=True).to_markdown(index=False, floatfmt=".2f"))
# %%

print(dfs20[dfs20["setup"].str.contains("multivariate", na=False)].reset_index(drop=True).to_markdown(index=False, floatfmt=".2f"))
# %%
