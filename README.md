# Reproducibility package: off-policy evaluation of renewal pricing under weak overlap

Code and result files for the study "Off-policy evaluation of renewal pricing under weak overlap: a diagnostic and benchmark study on motor-insurance data". The package contains the semi-synthetic benchmark, the real-data overlap diagnostic, the stored results of every configuration reported in the paper and the scripts that derive the secondary tables from them.

## Contents
```
code/
  ope_env.py            benchmark: environment, estimators, replications, bootstrap (one run = one configuration)
  run_all.sh            the command line of every configuration in results/benchmark
  support_diagnostic.py real-data overlap diagnostic on the first portfolio
  reward_check.py       simulation check that the mean simulated reward equals the intended mean reward
  score_variance.py     standard deviation of the reward and of the doubly robust score under uniform logging
  smoothing_bias.py     exact value of the kernel-smoothed policy for each bandwidth
  paired_comparison.py, policy_selection.py, bias_sd.py, extension_120.py, extensions.py
                        derived tables, computed from the stored per-replication errors
data/SHA256SUMS.txt     checksums of the two input files
results/
  benchmark/            summary_*.csv, meta_*.json (arguments and true policy values), raw_*.csv (error of every estimator,
                        policy, scale and replication), ast_*.csv (support-threshold rule), one set per tag
  derived/              outputs of the derived-table scripts and of the two variance scripts
  real_data/            output of the real-data diagnostic
```

## Data (not redistributed)
1. First portfolio: Segura-Gisbert, Lledó and Pavía, Mendeley Data, doi:10.17632/5cxyb5fp4f.2 (CC BY 4.0). Save as `seg.csv` (semicolon-separated, as downloaded).
2. Second portfolio: Guillén, Bolancé, Frees and Valdez, Data in Brief 39:107639 (2021), doi:10.1016/j.dib.2021.107639. Save as `gui.csv` (comma-separated, with the row-number first column).

Compare with `data/SHA256SUMS.txt`; a different export of the same data may differ in bytes.

## Environment
`pip install -r requirements.txt` (Python 3.13; the versions listed were used for all results). Results can differ slightly across scikit-learn versions. Seeds are fixed in `ope_env.py`.

## Run
Copy `seg.csv` and `gui.csv` into `code/`, then:
```
cd code
bash run_all.sh                                   # all configurations; outputs in code/out/
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_base --reps 40    # one configuration
```
Quick check (about one minute): add `--reps 2 --scales 0.10`; the log line for the first portfolio reports the true policy values V = oracle 113.4, oracle_cap10 111.1, hold 101.9 (they do not depend on the number of replications), and the per-replication errors equal the first two replications stored in `results/benchmark/raw_nh_base.csv`.

The derived tables are computed from the stored results; run them from the package root:
```
python3 -I code/paired_comparison.py ; python3 -I code/policy_selection.py ; python3 -I code/bias_sd.py
python3 -I code/extension_120.py ; python3 -I code/extensions.py
python3 -I code/support_diagnostic.py seg.csv results/real_data
```

## Configurations (tags)
| Tag | Content |
|---|---|
| `nh_base` | first portfolio, base configuration (40 replications of 20,000 contract-years sampled by customer; elasticity 6, expense 20%, six-level grid, logit response) |
| `nh_e2`, `nh_e3`, `nh_e10`, `nh_e12` | elasticity level 2, 3, 10, 12 |
| `nh_probit`, `nh_exp10`, `nh_exp30`, `nh_clv50`, `nh_grid9` | probit response; expense 10% and 30%; retention value 0.5 P0; nine-level grid |
| `nh_delay30_naive`, `nh_delay30_adj`, `nh_da20`, `nh_da40` | 30% of claims unreported, unadjusted; adjusted with the true factor; adjusted with an assumed factor of 0.2 and 0.4 |
| `nh_basex1`, `nh_basex2` | replications 40-79 and 80-119 of the base configuration (scales 0.03 and 0.10); with `nh_base` they give 120 replications |
| `nh_boot500` | 500 bootstrap resamples (scales 0.01 and 0.10) |
| `nh_n40k`, `nh_n80k` | 40,000 and 80,000 contract-years, uniform logging |
| `abl_timing` | base configuration without policies in force, maximum policies and maximum products |
| `hist_base` | base configuration with the claims-history covariates included |
| `fact`, `fact01`, `fact40k`, `fact80k` | factorised direct method (renewal probability x expected claim cost) in the base configuration and at larger samples |
| `renewalt`, `renewaltfact` | alternative definition of renewal (a contract-year followed by a later row of the same customer counts as renewed), without and with the factorised direct method |
| `g_base`, `g_e2`, `g_e12`, `g_probit` | second portfolio |
| `gfactA`, `gfactB` | second portfolio, factorised direct method (12 replications; scales 0.01-0.03 and 0.10-uniform) |

`nh_` marks runs without the claims-history covariates `N_claims_history` and `R_Claims_history`, which appear to include the current contract year (`--nohist 1`). `--group 1` samples, cross-fits and bootstraps whole customers. A scale is the width of the logging distribution around the engine action (0.01 near-deterministic, `inf` uniform).

## Notes
- Elasticity, expense load and the unreported-claim fraction are assumptions of the benchmark, not estimates.
- The second portfolio has no premium variable; the reference premium is reconstructed as the expected claim cost divided by 0.65, and the engine action there is driven by the risk level.
- Licence: MIT (see `LICENSE`).
#
