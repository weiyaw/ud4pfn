#!/usr/bin/env bash
set -euo pipefail

# 1. Frequentist coverage
for seed in $(seq 1000 1049); do
    for setup in gaussian-linear-multivariate gaussian-linear-dependent-error-multivariate poisson-linear-multivariate probit-mixture-multivariate categorical-linear-multivariate; do
        for data_size in 200 500 1000; do
            uv run python -m experiments.coverage.run \
                setup="$setup" data_size="$data_size" seed="$seed"
        done
    done
done

for rep_dir in outputs/coverage/*/; do
    uv run python -m experiments.coverage.run_bootstrap \
        rep_dir="$rep_dir" bootstrap_samples=200
done

for rep_dir in outputs/coverage/*/; do
    if [[ $(basename "$rep_dir") =~ ^setup=(gaussian|poisson) ]]; then
        uv run python -m experiments.coverage.run_copula rep_dir="$rep_dir"
    fi
done

# 2. Gap in observations
for setup in gaussian-linear gaussian-polynomial gaussian-linear-dependent-error gaussian-sine poisson-linear probit-mixture categorical-linear; do
    for data_size in 200 500 1000; do
        uv run python -m experiments.gap.run \
            setup="$setup" data_size="$data_size" seed=1000
    done
done

# 3. Real-data illustrations
uv run python -m experiments.real_analysis.run setup=labour-force seed=1000
uv run python -m experiments.real_analysis.run setup=fibre-strength seed=1000

# 4. Entropic uncertainty decomposition
for data_size in 15 50 75 150; do
    uv run python -m experiments.entropic_ud.run \
        setup=logistic-linear data_size="$data_size" fix_data=true seed=1000
done

for setup_size in two-moons-1:30 two-moons-2:30 two-moons-1:100 two-moons-2:100 spiral:200; do
    setup=${setup_size%%:*}
    data_size=${setup_size##*:}
    uv run python -m experiments.entropic_ud.run \
        setup="$setup" x_design=null data_size="$data_size" fix_data=false seed=1000
done

for seed in $(seq 1000 1049); do
    for data_size in $(seq 75 5 200); do
        uv run python -m experiments.entropic_ud.run --config-name vary_n \
            data_size="$data_size" seed="$seed"
    done
done
