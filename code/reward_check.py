"""Simulation check that the mean simulated reward equals the intended mean reward mu(x, a).

Usage: python3 -I reward_check.py --csv CSV --out DIR --group 1 [--dataset segura|guillen --nohist 1 --reps N --tag T]
"""
import argparse, time, json, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.model_selection import GroupKFold, KFold
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="base")
ap.add_argument("--e0", type=float, default=6.0); ap.add_argument("--expense", type=float, default=0.20)
ap.add_argument("--grid", type=int, default=6); ap.add_argument("--delay", type=float, default=0.0)
ap.add_argument("--adjust", type=int, default=0); ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--n", type=int, default=20000); ap.add_argument("--nboot", type=int, default=200)
ap.add_argument("--scales", default="0.01,0.03,0.10,inf")
ap.add_argument("--link", default="logit"); ap.add_argument("--clv", type=float, default=0.0)
ap.add_argument("--nohist", type=int, default=0); ap.add_argument("--group", type=int, default=0); ap.add_argument("--delay_assumed", type=float, default=-1.0); ap.add_argument("--dataset", default="segura"); ap.add_argument("--lr_target", type=float, default=0.65)
a = ap.parse_args()
A = np.array([-.10, -.05, 0, .05, .10, .15] + ([.20, .25, .30] if a.grid == 9 else [])); K = 5000.0; EPS = 0.05; CAP = 0.10
i0 = int(np.where(A == 0)[0][0]); NA = len(A)

if a.dataset == "segura":
    d = pd.read_csv(a.csv, sep=";")
    for c in ["Date_last_renewal", "Date_birth", "Date_driving_licence"]:
        d[c] = pd.to_datetime(d[c], format="%d/%m/%Y", errors="coerce")
    d["age"] = (d.Date_last_renewal - d.Date_birth).dt.days / 365.25
    d["lic"] = (d.Date_last_renewal - d.Date_driving_licence).dt.days / 365.25
    d["veh_age"] = d.Date_last_renewal.dt.year - d.Year_matriculation
    d["fuel"] = (d.Type_fuel == "D").astype(float); d.loc[d.Type_fuel.isna(), "fuel"] = np.nan
    d["logP0"] = np.log(d.Premium)
    F = ["age","lic","veh_age","Seniority","Policies_in_force","Max_policies","Max_products","Payment","Distribution_channel","Type_risk",
         "Area","Second_driver","Power","Cylinder_capacity","Value_vehicle","N_doors","fuel","Length","Weight","N_claims_history","R_Claims_history","logP0"]
    if a.nohist: F = [f for f in F if f not in ("N_claims_history", "R_Claims_history")]
    X = d[F].astype(float).values; P0 = d.Premium.values.astype(float); n_all = len(d)
    claim = np.minimum(d.Cost_claims_year.values, K); renew = (d.Lapse.values == 0).astype(int); groups = d.ID.values
else:
    d = pd.read_csv(a.csv)
    F = ["gender","Age_client","year","age_of_car_M","Car_power_M","Car_2ndDriver_M","num_policiesC","metro_code","Policy_PaymentMethodA","Policy_PaymentMethodH",
         "Insuredcapital_content_re","Insuredcapital_continent_re","appartment","Client_Seniority"]
    X = d[F].astype(float).values; n_all = len(d); P0 = np.ones(n_all)   # P0 reconstructed below
    claim = np.minimum(d.Claims1.values, K); renew = d.Retention.values.astype(int); groups = d.PolID.values
    d["age"] = d.Age_client.astype(float); d["Seniority"] = d.Client_Seniority.astype(float); d["Distribution_channel"] = d.Policy_PaymentMethodA.astype(float)
def oof(fn, y, kind):
    out = np.zeros(n_all)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = fn().fit(X[tr], y[tr]); out[te] = m.predict_proba(X[te])[:, 1] if kind == "clf" else m.predict(X[te])
    return out
t0 = time.time()
m_hat = np.clip(oof(lambda: HGR(loss="poisson", max_iter=100, learning_rate=.05, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0), claim, "reg"), 1.0, .5 * K)
if a.dataset != "segura":
    P0 = m_hat / a.lr_target
q_hat = oof(lambda: HGC(max_iter=150, learning_rate=.08, early_stopping=False, random_state=0), (claim > 0).astype(int), "clf")
q = np.clip(np.maximum(q_hat, m_hat / K), .005, 1.0); sev = m_hat / q
p_base = np.clip(oof(lambda: HGC(max_iter=150, learning_rate=.08, early_stopping=False, random_state=0), renew, "clf"), .05, .98)
z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
age_z = z(np.nan_to_num(d.age.values, nan=np.nanmedian(d.age.values))); sen_z = z(d.Seniority.values.astype(float)); ch = d.Distribution_channel.values.astype(float)
elas = np.clip(a.e0 * np.exp(-.30 * age_z - .25 * sen_z + .30 * ch), a.e0 / 3, a.e0 * 2.5)
lg = np.log(p_base / (1 - p_base))
if a.link == "probit": Prn = norm.cdf(norm.ppf(p_base)[:, None] - elas[:, None] * A[None, :] / 1.7)
else: Prn = 1 / (1 + np.exp(-(lg[:, None] - elas[:, None] * A[None, :])))
dA = a.delay if a.delay_assumed < 0 else a.delay_assumed
el_impl = float(np.mean(-(np.log(np.clip(Prn[:, A.tolist().index(.05)], 1e-9, 1)) - np.log(np.clip(Prn[:, i0], 1e-9, 1))) / np.log(1.05)))  # implied own-price elasticity of renewal prob
Price = P0[:, None] * (1 + A[None, :]); EXPN = a.expense * P0
Rexp = Prn * (Price - m_hat[:, None] - EXPN[:, None] + a.clv * P0[:, None])
lr = np.log(m_hat / P0); a_cont = .08 * (lr - lr.mean()) / lr.std() + .025
a_star = np.abs(A[None, :] - np.clip(a_cont, A.min(), A.max())[:, None]).argmin(1)
capmask = A <= CAP + 1e-9
POL = {"oracle": Rexp.argmax(1), "oracle_cap10": np.where(capmask[None, :], Rexp, -np.inf).argmax(1),
       "shift_up": np.minimum(a_star + 1, NA - 1), "hold": np.full(n_all, i0), "engine": a_star}
def logging_probs(s):
    if np.isinf(s): return np.full((n_all, NA), 1 / NA)
    w = np.exp(-np.abs(A[None, :] - A[a_star][:, None]) / s); return w / w.sum(1, keepdims=True)
Vtrue = {k: float(Rexp[np.arange(n_all), v].mean()) for k, v in POL.items()}
print(f"[{a.tag}] el={el_impl:.2f} base {time.time()-t0:.0f}s renew={renew.mean():.3f} m={m_hat.mean():.1f} V={ {k: round(v,1) for k,v in Vtrue.items()} }", flush=True)

# ---- check: E[R | x, a] from the simulation mechanism equals mu(x, a) = Rexp[x, a]
rng = np.random.default_rng(2026); M_DRAWS = 4000; sub = rng.choice(n_all, 400, replace=False); out = []
for adj_mode, label in [(0, "delta=0"), (1, "delta=0.3 adjusted"), (2, "delta=0.3 naive")]:
    dl = 0.0 if adj_mode == 0 else 0.3
    err = []; mc = []
    for ai_ in range(NA):
        for i in sub[:200]:
            rn = rng.random(M_DRAWS) < Prn[i, ai_]; has = rng.random(M_DRAWS) < q[i]; rep = rng.random(M_DRAWS) >= dl
            cl = has * rep * sev[i] / ((1 - 0.3) if adj_mode == 1 else 1.0)
            Rm = (rn * (Price[i, ai_] - cl - EXPN[i] + a.clv * P0[i])).mean()
            tgt = Rexp[i, ai_] if adj_mode < 2 else Prn[i, ai_] * (Price[i, ai_] - 0.7 * m_hat[i] - EXPN[i] + a.clv * P0[i])
            err.append(Rm - tgt); mc.append(np.std(rn * (Price[i, ai_] - has * sev[i] - EXPN[i])) / np.sqrt(M_DRAWS))
    err = np.array(err); print(f"{label}: mean(MC - mu)={err.mean():+.3f}  mean abs={np.abs(err).mean():.3f}  expected MC se~{np.mean(mc):.3f}  max sigma-units={np.abs(err/np.array(mc)).max():.1f}  sev<=K: {bool((sev<=K+1e-9).all())}  q range [{q.min():.3f},{q.max():.3f}]")
# R is zero for every non-renewal by construction: the renewal indicator rn multiplies the whole bracket (premium, claim cost, expense and retention value)
print("R = rn * (premium - claims - expense + retention value) is 0 for every non-renewal by construction")
