"""Per-method wall-clock timing for the illustration settings.

Times, on one machine, every pipeline the paper's real-data and
moons/spiral/logistic illustrations require:

  ours      gn (single plug-in predictive)  +  prefix trajectory g_0..g_n
            +  v_n (pointwise)  +  entropy decomposition (total/aleatoric/
            epistemic).  The entropy/v_n stage is pure numpy on cached
            arrays and is timed to show it is negligible.
  mc-un     the Monte-Carlo one-step-ahead arm (sample_gn_plus_1,
            mc_samples draws); the real-data figures use it, the
            moons/spiral/logistic figures do not.
  bootstrap B refits of TabPFN on resampled data (run-bootstrap.py logic).
            Cost is exactly B independent fit+predict evaluations, so it
            scales linearly in B.

The copula / nonparametric-resampling baseline (run-copula.py) wraps a
conditional-REGRESSION copula and has no classification analogue; every
setting here is classification, so it cannot be run on these settings.

Standalone: no hydra. Reads the frozen data.pickle of each existing run
directory (same data, same t, same x_grid as the paper's figures).

Usage:
    python time-methods.py --outputs-root outputs --out results_timing \
        --settings all --bootstrap-b 200 --device cuda
    python time-methods.py --collect results_timing   # emit markdown
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

# (label, relative rep dir, is_multiclass)
SETTINGS: dict[str, tuple[str, bool]] = {
    "logistic-n15": (
        "entropic-ud/setup=logistic-linear x_design=gaussian:1.5:3.0 shuffle=True n_est=64 n=15 m=100 seed=1000",
        False,
    ),
    "logistic-n50": (
        "entropic-ud/setup=logistic-linear x_design=gaussian:1.5:3.0 shuffle=True n_est=64 n=50 m=100 seed=1000",
        False,
    ),
    "logistic-n75": (
        "entropic-ud/setup=logistic-linear x_design=gaussian:1.5:3.0 shuffle=True n_est=64 n=75 m=100 seed=1000",
        False,
    ),
    "logistic-n150": (
        "entropic-ud/setup=logistic-linear x_design=gaussian:1.5:3.0 shuffle=True n_est=64 n=150 m=100 seed=1000",
        False,
    ),
    "moons1-n30": (
        "entropic-ud/setup=two-moons-1 x_design=None shuffle=True n_est=64 n=30 m=100 seed=1000",
        False,
    ),
    "moons1-n100": (
        "entropic-ud/setup=two-moons-1 x_design=None shuffle=True n_est=64 n=100 m=100 seed=1000",
        False,
    ),
    "moons2-n30": (
        "entropic-ud/setup=two-moons-2 x_design=None shuffle=True n_est=64 n=30 m=100 seed=1000",
        False,
    ),
    "moons2-n100": (
        "entropic-ud/setup=two-moons-2 x_design=None shuffle=True n_est=64 n=100 m=100 seed=1000",
        False,
    ),
    "spiral-n200": (
        "entropic-ud/setup=spiral x_design=None shuffle=True n_est=64 n=200 m=100 seed=1000",
        True,
    ),
    "labour-force": (
        "2026-01-51/setup=labour-force shuffle=True n_est=64 m=100 seed=1000",
        False,
    ),
    "fibre-strength": (
        "2026-01-51/setup=fibre-strength shuffle=True n_est=64 m=100 seed=1000",
        False,
    ),
}
REAL_DATA = {"labour-force", "fibre-strength"}


def time_setting(
    label: str,
    rep_dir: Path,
    multiclass: bool,
    n_estimators: int,
    bootstrap_b: int,
    mc_samples: int,
    device: str,
    seed: int = 1000,
) -> dict:
    import jax.random as jr

    import posterior
    import utils
    from metrics import (
        compute_aleatoric_entropy_binary,
        compute_aleatoric_entropy_multiclass,
        compute_total_entropy_binary,
        compute_total_entropy_multiclass,
    )
    from pred_rule import TabPFNClassifierPPD

    data = utils.read_from(str(rep_dir / "data.pickle"))
    x_prev, y_prev = data["x_prev"], data["y_prev"]
    t, x_grid = data["t"], data["x_grid"]
    n, m = x_prev.shape[0], x_grid.shape[0]

    torch.manual_seed(8655 + seed)
    clf = TabPFNClassifierPPD(
        n_estimators=n_estimators,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt",
        device=device,
    )

    res: dict = {
        "label": label,
        "n": int(n),
        "grid_points": int(m),
        "num_t": int(np.atleast_1d(t).shape[0]),
        "n_estimators": n_estimators,
        "bootstrap_b": bootstrap_b,
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        res["gpu"] = torch.cuda.get_device_name(0)

    # --- ours: single plug-in predictive g_n
    t0 = time.perf_counter()
    gn = posterior.compute_gn(clf, t, x_grid, x_prev, y_prev)
    res["t_gn"] = time.perf_counter() - t0
    print(f"[{label}] gn: {res['t_gn']:.2f}s", flush=True)

    # --- ours: prefix trajectory g_0..g_n
    t0 = time.perf_counter()
    g0_to_gn = posterior.compute_g0_to_gn(clf, t, x_grid, x_prev, y_prev)
    res["t_prefix"] = time.perf_counter() - t0
    print(f"[{label}] prefix (n={n}): {res['t_prefix']:.2f}s", flush=True)

    # --- ours: v_n + entropy decomposition (numpy post-processing)
    t0 = time.perf_counter()
    if multiclass:
        K = np.atleast_1d(t).shape[0]
        clt_var = np.array(
            [posterior.compute_vn(g0_to_gn[:, k], type="pointwise") / n for k in range(K)]
        )
        total = compute_total_entropy_multiclass(gn)
        alea = compute_aleatoric_entropy_multiclass(gn, clt_var)
    else:
        clt_var = posterior.compute_vn(g0_to_gn[:, 0], type="pointwise") / n
        total = compute_total_entropy_binary(gn[0])
        alea = compute_aleatoric_entropy_binary(gn[0], clt_var)
    _ = total - alea
    res["t_vn_entropy"] = time.perf_counter() - t0
    res["t_ours_total"] = res["t_gn"] + res["t_prefix"] + res["t_vn_entropy"]
    print(f"[{label}] vn+entropy: {res['t_vn_entropy']:.4f}s", flush=True)

    # --- Monte-Carlo one-step-ahead arm (real-data figures only)
    if mc_samples > 0:
        key = jr.key(1907 + seed)
        key_others, _ = jr.split(key)
        t0 = time.perf_counter()
        posterior.sample_gn_plus_1(
            key_others, clf, t, x_grid, x_prev, y_prev, size=mc_samples
        )
        res["t_mc_un"] = time.perf_counter() - t0
        res["mc_samples"] = mc_samples
        print(f"[{label}] mc-un ({mc_samples}): {res['t_mc_un']:.2f}s", flush=True)

    # --- bootstrap (run-bootstrap.py logic, B refits)
    if bootstrap_b > 0:
        rng = np.random.default_rng(seed + 10391)
        t0 = time.perf_counter()
        for _ in range(bootstrap_b):
            idx = rng.integers(0, n, size=n)
            posterior.compute_gn(clf, t, x_grid, x_prev[idx], y_prev[idx])
        res["t_bootstrap"] = time.perf_counter() - t0
        print(f"[{label}] bootstrap (B={bootstrap_b}): {res['t_bootstrap']:.2f}s", flush=True)

    return res


def fmt_s(x: float | None) -> str:
    if x is None:
        return "--"
    if x < 100:
        return f"{x:.1f}"
    return f"{x:.0f}"


def collect(out_dir: Path) -> str:
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        rows.append(json.loads(f.read_text()))
    order = list(SETTINGS)
    rows.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else 99)

    lines = [
        "| Setting | n | grid | ours: prefix+g_n+entropies (s) | MC arm, 1000 draws (s) | bootstrap B=200 (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        b = r.get("t_bootstrap")
        if b is not None and r.get("bootstrap_b") != 200:
            b = b * 200 / r["bootstrap_b"]
        lines.append(
            f"| {r['label']} | {r['n']} | {r['grid_points']} | "
            f"{fmt_s(r.get('t_ours_total'))} | {fmt_s(r.get('t_mc_un'))} | {fmt_s(b)} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-root", type=str, default="outputs")
    p.add_argument("--out", type=str, default="results_timing")
    p.add_argument("--settings", nargs="+", default=["all"])
    p.add_argument("--bootstrap-b", type=int, default=200)
    p.add_argument("--mc-samples", type=int, default=1000,
                   help="MC draws for the one-step-ahead arm (real-data settings only)")
    p.add_argument("--n-estimators", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--collect", type=str, default=None,
                   help="Directory of result JSONs; print the markdown table and exit")
    args = p.parse_args()

    if args.collect:
        print(collect(Path(args.collect)))
        return

    labels = list(SETTINGS) if args.settings == ["all"] else args.settings
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.outputs_root)

    for label in labels:
        rep_rel, multiclass = SETTINGS[label]
        rep_dir = root / rep_rel
        if not (rep_dir / "data.pickle").exists():
            print(f"[skip] {label}: no data.pickle under {rep_dir}", flush=True)
            continue
        mc = args.mc_samples if label in REAL_DATA else 0
        res = time_setting(
            label, rep_dir, multiclass,
            n_estimators=args.n_estimators,
            bootstrap_b=args.bootstrap_b,
            mc_samples=mc,
            device=args.device,
        )
        (out_dir / f"{label}.json").write_text(json.dumps(res, indent=1))

    print("\n" + collect(out_dir))


if __name__ == "__main__":
    main()
