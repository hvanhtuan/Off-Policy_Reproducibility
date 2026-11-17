#!/bin/bash
# Every benchmark configuration. Run inside code/ with seg.csv and gui.csv copied there; outputs go to out/.
# Configurations are independent; run them in parallel as the machine allows. The first portfolio takes about 4 s per replication and scale,
# the second about 90 s. Seeds are fixed in ope_env.py.
set -e
export OMP_NUM_THREADS=1
mkdir -p out

# First portfolio, base configuration and one-at-a-time changes of the environment (no claims-history covariates)
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_base --reps 40
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_e2 --reps 24 --e0 2
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_e3 --reps 24 --e0 3
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_e10 --reps 24 --e0 10
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_e12 --reps 24 --e0 12
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_probit --reps 24 --link probit
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_exp10 --reps 24 --expense 0.1
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_exp30 --reps 24 --expense 0.3
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_clv50 --reps 24 --clv 0.5
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_grid9 --reps 24 --grid 9
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_delay30_naive --reps 24 --delay 0.3
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_delay30_adj --reps 24 --delay 0.3 --adjust 1
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_da20 --reps 24 --delay 0.3 --adjust 1 --delay_assumed 0.2
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_da40 --reps 24 --delay 0.3 --adjust 1 --delay_assumed 0.4

# First portfolio, extensions: 120 replications, 500 bootstrap resamples, larger samples
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_basex1 --reps 40 --rep0 40 --scales 0.03,0.10
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_basex2 --reps 40 --rep0 80 --scales 0.03,0.10
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_boot500 --reps 24 --nboot 500 --scales 0.01,0.10
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_n40k --reps 10 --n 40000 --scales inf
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag nh_n80k --reps 10 --n 80000 --scales inf

# First portfolio, covariates whose timing is unclear removed; variant with the claims-history covariates
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag abl_timing --reps 40 --scales 0.10,inf --dropvars Policies_in_force,Max_policies,Max_products
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --tag hist_base --reps 40

# First portfolio, factorised direct method and alternative definition of renewal
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag fact --reps 40 --scales 0.03,0.10,inf --dm_fact 1
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag fact01 --reps 40 --scales 0.01 --dm_fact 1
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag fact40k --reps 10 --n 40000 --scales inf --dm_fact 1
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag fact80k --reps 10 --n 80000 --scales inf --dm_fact 1
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag renewalt --reps 40 --renew_alt 1
python3 -I ope_env.py --csv seg.csv --dataset segura --out out --group 1 --nohist 1 --tag renewaltfact --reps 40 --dm_fact 1 --renew_alt 1

# Second portfolio
python3 -I ope_env.py --csv gui.csv --dataset guillen --out out --group 1 --tag g_base --reps 40
python3 -I ope_env.py --csv gui.csv --dataset guillen --out out --group 1 --tag g_e12 --reps 24 --e0 12
python3 -I ope_env.py --csv gui.csv --dataset guillen --out out --group 1 --tag g_e2 --reps 24 --e0 2
python3 -I ope_env.py --csv gui.csv --dataset guillen --out out --group 1 --tag g_probit --reps 24 --link probit
python3 -I ope_env.py --csv gui.csv --dataset guillen --out out --group 1 --tag gfactA --reps 12 --scales 0.01,0.03 --dm_fact 1
python3 -I ope_env.py --csv gui.csv --dataset guillen --out out --group 1 --tag gfactB --reps 12 --scales 0.10,inf --dm_fact 1
