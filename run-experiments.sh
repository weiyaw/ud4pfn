# simple quasi-check
for outer_idx in $(seq 0 99); do
    python run-quasi-simple.py id="2026-01-33" fix_data=False n_estimators=8 outer_idx=$outer_idx seed=1000
    python run-quasi-simple.py id="2026-01-33" fix_data=False n_estimators=16 outer_idx=$outer_idx seed=1000
done

# coverage
for seed in $(seq 1000 1099); do
    for setup in gaussian-linear gaussian-polynomial gaussian-linear-dependent-error gaussian-sine poisson-linear probit-mixture categorical-linear; do
        for data_size in 200 500 1000; do
            python run-ghat.py id="2026-01-22" setup="$setup" n_estimators=64 x_design="uniform-1d" seed=$seed data_size=$data_size
        done
    done
done


# coverage (n_est=16)
for seed in $(seq 1000 1099); do
    for setup in gaussian-linear gaussian-polynomial gaussian-linear-dependent-error gaussian-sine poisson-linear probit-mixture categorical-linear; do
        for data_size in 200 500 1000; do
            python run-ghat.py id="2026-01-23" setup="$setup" n_estimators=16 x_design="uniform-1d" seed=$seed data_size=$data_size
        done
    done
done


# coverage (multivariate, n_est=16)
for seed in $(seq 1000 1049); do
    for setup in gaussian-linear-multivariate gaussian-linear-dependent-error poisson-linear-multivariate probit-mixture-multivariate categorical-linear-multivariate; do
        for data_size in 200 500 1000; do
            for x_design in "sobol-10d"; do
                python run-ghat.py id="2026-01-23" setup="$setup" n_estimators=16 x_design="$x_design" seed=$seed data_size=$data_size
            done
        done
    done
done


# bootstrap samples for coverage (required by visual-coverage.py)
for rep_dir in outputs/2026-01-22/*/; do
    python run-bootstrap.py rep_dir="$rep_dir" bootstrap_samples=200
done

for rep_dir in outputs/2026-01-23/*/; do
    python run-bootstrap.py rep_dir="$rep_dir" bootstrap_samples=200
done

for rep_dir in outputs/2026-01-23/*/; do
    if [[ $(basename "$rep_dir") =~ ^setup=(gaussian|poisson) ]]; then
        python run-copula.py rep_dir="$rep_dir"
    fi
done

# gap
for setup in gaussian-linear gaussian-polynomial gaussian-linear-dependent-error gaussian-sine poisson-linear probit-mixture categorical-linear; do
    for data_size in 200 500 1000; do
        python run-ghat.py id="2026-01-12" setup="$setup" n_estimators=64 x_design="one-gap" seed=1000 data_size=$data_size
    done
done



# real data analysis
python run-real-analysis.py id="2026-01-51" setup="labour-force" n_estimators=64 seed=1000
python run-real-analysis.py id="2026-01-51" setup="fibre-strength" n_estimators=64 seed=1000


# entropic uncertainty decomposition (varying x^star)
for data_size in 15 50 75 150; do
    python run-ghat.py id="2026-01-61" setup=logistic-linear n_estimators=64 x_design="gaussian:1.5:3.0" data_size=$data_size fix_data=True seed=1000
done


# entropic uncertainty decomposition (2D and 3D examples)
python run-ghat.py id="2026-01-61" setup=two-moons-1 n_estimators=64 x_design=None data_size=30 fix_data=False seed=1000
python run-ghat.py id="2026-01-61" setup=two-moons-2 n_estimators=64 x_design=None data_size=30 fix_data=False seed=1000
python run-ghat.py id="2026-01-61" setup=two-moons-1 n_estimators=64 x_design=None data_size=100 fix_data=False seed=1000
python run-ghat.py id="2026-01-61" setup=two-moons-2 n_estimators=64 x_design=None data_size=100 fix_data=False seed=1000
python run-ghat.py id="2026-01-61" setup=spiral n_estimators=64 x_design=None data_size=200 fix_data=False seed=1000

# entropic uncertainty decomposition (varying number of data size)
for seed in $(seq 1000 1049); do
    for data_size in $(seq 75 5 200); do
        python run-ghat.py id="2026-01-62" setup=logistic-linear n_estimators=64 x_design="gaussian:1.5:3.0" data_size=$data_size fix_data=False mc_samples=0 seed=$seed
    done
done
