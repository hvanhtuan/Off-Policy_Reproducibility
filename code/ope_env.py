"""Semi-synthetic renewal-pricing environment and off-policy value estimators.

Estimators: IPS, SNIPS, DM (single reward model), factorised DM (renewal x claims), DR, DR-clip, DR with estimated
propensity, SWITCH-DR, kernel-smoothed IPS (KIPS) and the support-restricted value J_S, evaluated against known policy
values. Customers are sampled whole, cross-fitted and bootstrapped as clusters.

Usage: python3 -I ope_env.py --csv CSV --dataset segura|guillen --out DIR --group 1 [--tag T --e0 6 --expense .2
       --grid 6|9 --delay 0 --adjust 0 --reps 40 --n 20000 --scales 0.01,0.03,0.10,inf --nohist 1 --dm_fact 1 --renew_alt 1]
Outputs in DIR: summary_TAG.csv, meta_TAG.json, raw_TAG.csv (per-replication errors), ast_TAG.csv (support-threshold rule).
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
ap.add_argument("--n", type=int, default=20000); ap.add_argument("--rep0", type=int, default=0); ap.add_argument("--nboot", type=int, default=200)
ap.add_argument("--scales", default="0.01,0.03,0.10,inf")
ap.add_argument("--link", default="logit"); ap.add_argument("--clv", type=float, default=0.0)
ap.add_argument("--dropvars", default=""); ap.add_argument("--dm_fact", type=int, default=0); ap.add_argument("--renew_alt", type=int, default=0); ap.add_argument("--nohist", type=int, default=0); ap.add_argument("--group", type=int, default=0); ap.add_argument("--delay_assumed", type=float, default=-1.0); ap.add_argument("--dataset", default="segura"); ap.add_argument("--lr_target", type=float, default=0.65)
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
    if a.dropvars: F = [f for f in F if f not in a.dropvars.split(",")]   # ablation of covariates whose timing relative to the quote is not stated; drop claims-history covariates that appear to include the current contract year
    X = d[F].astype(float).values; P0 = d.Premium.values.astype(float); n_all = len(d)
    claim = np.minimum(d.Cost_claims_year.values, K); renew = (d.Lapse.values == 0).astype(int); groups = d.ID.values
    if a.renew_alt:   # a contract-year followed by a row of the same customer (matching Date_next_renewal) counts as renewed even if Lapse > 0
        _o = np.lexsort((d.Date_last_renewal.values, d.ID.values)); _id = d.ID.values[_o]; _hn = np.zeros(n_all, bool); _hn[_o[:-1]] = _id[:-1] == _id[1:]
        renew = np.maximum(renew, _hn.astype(int)); print("renew_alt: rows changed", int((renew != (d.Lapse.values == 0)).sum()), flush=True)
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
# on the second portfolio P0 = m_hat/lr_target, so log(m_hat/P0) is constant (float round-off only); the engine then uses the risk level log(m_hat)
lr = np.log(m_hat / P0) if a.dataset == "segura" else np.log(m_hat); a_cont = .08 * (lr - lr.mean()) / lr.std() + .025
a_star = np.abs(A[None, :] - np.clip(a_cont, A.min(), A.max())[:, None]).argmin(1)
capmask = A <= CAP + 1e-9
POL = {"oracle": Rexp.argmax(1), "oracle_cap10": np.where(capmask[None, :], Rexp, -np.inf).argmax(1),
       "shift_up": np.minimum(a_star + 1, NA - 1), "hold": np.full(n_all, i0), "engine": a_star}
def logging_probs(s):
    if np.isinf(s): return np.full((n_all, NA), 1 / NA)
    w = np.exp(-np.abs(A[None, :] - A[a_star][:, None]) / s); return w / w.sum(1, keepdims=True)
Vtrue = {k: float(Rexp[np.arange(n_all), v].mean()) for k, v in POL.items()}
print(f"[{a.tag}] el={el_impl:.2f} base {time.time()-t0:.0f}s renew={renew.mean():.3f} m={m_hat.mean():.1f} V={ {k: round(v,1) for k,v in Vtrue.items()} }", flush=True)

def fit_mu(Xtr, atr, rtr, Xte):
    m = HGR(max_iter=120, learning_rate=.08, early_stopping=False, random_state=0).fit(np.c_[Xtr, A[atr]], rtr)
    return np.column_stack([m.predict(np.c_[Xte, np.full(len(Xte), v)]) for v in A])
def _cboot(vec, gidx, B=None):
    B = B or a.nboot; _u, c = np.unique(gidx, return_inverse=True); G = len(_u)
    Sc = np.bincount(c, weights=vec, minlength=G); nc = np.bincount(c, minlength=G).astype(float); out = np.empty(B)
    for b_ in range(B):
        d_ = rng_b.integers(0, G, G); out[b_] = Sc[d_].sum() / nc[d_].sum()
    return out
def _cse(z, gi):   # cluster-robust SE of the contract-year mean (ratio of cluster sums to N)
    _u, c = np.unique(gi, return_inverse=True); G = len(_u); Sc = np.bincount(c, weights=z, minlength=G); nc = np.bincount(c, minlength=G).astype(float); m_ = Sc.sum() / nc.sum()
    return float(np.sqrt(G / max(G - 1, 1) * ((Sc - m_ * nc) ** 2).sum()) / nc.sum())
rows = []; arows = []; N = a.n; rng_b = np.random.default_rng(99); EG = (0.02, 0.05, 0.10, 0.20); CVMAX = 0.10; ESSMIN = 100.0
for sc in [float(s) for s in a.scales.split(",")]:
    L = logging_probs(sc); Ls = L * Prn; Ls = Ls / Ls.sum(1, keepdims=True); Ss = Ls >= EPS   # propensity among renewers (survivor-selected rows)
    Jse = {e_: {k: float((Rexp[np.arange(n_all), v] * (L[np.arange(n_all), v] >= e_)).mean()) for k, v in POL.items()} for e_ in (0.01, 0.05, 0.10)}; Js = Jse[0.05]
    Jg = {e_: {k: float((Rexp[np.arange(n_all), v] * (L[np.arange(n_all), v] >= e_)).mean()) for k, v in POL.items()} for e_ in EG}
    for r in range(a.rep0, a.rep0 + a.reps):
        rng = np.random.default_rng(1000 * r + int(min(sc, 5) * 100) + 1)
        if a.group:   # sample whole customers (rows of a customer stay together), cross-fit and bootstrap by customer; target stays contract-weighted
            _ug, _gi = np.unique(groups, return_inverse=True); _rk = rng.permutation(len(_ug))[_gi]
            idx = np.argsort(_rk, kind="stable")[:N]; gidx = _gi[idx]
        else:
            idx = rng.choice(n_all, N, replace=False); gidx = None
        Li = L[idx]; ai = (rng.random(N)[:, None] > Li.cumsum(1)).sum(1); ar = np.arange(N)
        rn = rng.random(N) < Prn[idx, ai]; has = rng.random(N) < q[idx]; rep = rng.random(N) >= a.delay      # claim occurs / is reported
        cl_obs = has * rep * sev[idx] / ((1 - dA) if a.adjust else 1.0)
        R = rn * (Price[idx, ai] - cl_obs - EXPN[idx] + a.clv * P0[idx]); Xi = X[idx]; M = np.zeros((N, NA)); Ph = np.zeros((N, NA))
        for tr, te in (GroupKFold(2).split(Xi, groups=gidx) if a.group else KFold(2, shuffle=True, random_state=r).split(Xi)):
            M[te] = fit_mu(Xi[tr], ai[tr], R[tr], Xi[te])
            if np.isinf(sc): Ph[te] = 1 / NA
            else:
                cm = HGC(max_iter=60, learning_rate=.1, early_stopping=False, random_state=0).fit(Xi[tr], ai[tr]); Ph[np.ix_(te, cm.classes_)] = cm.predict_proba(Xi[te])
        M2 = None
        if a.dm_fact:   # factorised outcome model: P(renew | x, a) x (price - E[claim | x] - expense + retention value)
            M2 = np.zeros((N, NA)); cl_r = cl_obs
            for tr, te in (GroupKFold(2).split(Xi, groups=gidx) if a.group else KFold(2, shuffle=True, random_state=r).split(Xi)):
                pm = HGC(max_iter=120, learning_rate=.08, early_stopping=False, random_state=0).fit(np.c_[Xi[tr], A[ai[tr]]], rn[tr].astype(int))
                pt = np.column_stack([pm.predict_proba(np.c_[Xi[te], np.full(len(te), v)])[:, 1] for v in A])
                rr = tr[rn[tr]]; cm_ = HGR(max_iter=100, learning_rate=.05, min_samples_leaf=200, early_stopping=False, random_state=0).fit(Xi[rr], cl_r[rr]); ch_ = np.clip(cm_.predict(Xi[te]), 0, None)
                M2[te] = pt * (Price[idx[te]] - ch_[:, None] - EXPN[idx[te]][:, None] + a.clv * P0[idx[te]][:, None])
        Ph_nf = np.clip(Ph, 1e-4, 1); Ph_nf /= Ph_nf.sum(1, keepdims=True)   # learned propensity with a negligible floor
        Ph = np.clip(Ph, .01, 1); Ph /= Ph.sum(1, keepdims=True)
        for k, pol in POL.items():
            pi = pol[idx]; hit = ai == pi; w = hit / Li[ar, ai]; wh = hit / Ph[ar, ai]; mu_pi = M[ar, pi]; mu_a = M[ar, ai]; S = Li[ar, pi] >= EPS; Se = Ph[ar, pi] >= EPS; Se = Ph[ar, pi] >= EPS
            Kd = np.exp(-0.5 * ((A[None, :] - A[pi][:, None]) / 0.05) ** 2); Kd /= Kd.sum(1, keepdims=True); wk = Kd[ar, ai] / Li[ar, ai]   # kernel-smoothed target (bandwidth 0.05)
            psi = mu_pi + w * (R - mu_a)
            e = {"IPS": (w * R).mean(), "SNIPS": (w * R).sum() / max(w.sum(), 1e-9), "DM": mu_pi.mean(), "DR": psi.mean(),
                 "DR_clip20": (mu_pi + np.minimum(w, 20) * (R - mu_a)).mean(), "DR_estprop": (mu_pi + wh * (R - mu_a)).mean(), "DR_supp(J_S)": (S * psi).mean(), "DR_suppEst(J_S)": (Se * psi).mean(), "KIPS": (wk * R).mean(),
                 "SWITCH_DR20": (mu_pi + np.where(w <= 20, w, 0.0) * (R - mu_a)).mean(), "DR_supp01(J_S)": ((Li[ar, pi] >= 0.01) * psi).mean(), "DR_supp10(J_S)": ((Li[ar, pi] >= 0.10) * psi).mean(),
                 "DR_trueFloor01": (mu_pi + (hit / np.maximum(Li[ar, ai], 0.01)) * (R - mu_a)).mean(), "DR_estNoFloor": (mu_pi + (hit / Ph_nf[ar, ai]) * (R - mu_a)).mean()}
            if M2 is not None:
                m2p = M2[ar, pi]; m2a = M2[ar, ai]; e["DM_fact"] = m2p.mean(); e["DR_fact"] = (m2p + w * (R - m2a)).mean(); e["DR_fact_clip20"] = (m2p + np.minimum(w, 20) * (R - m2a)).mean()
            for C_ in (5, 10, 50, 100): e[f"DR_clip{C_}"] = (mu_pi + np.minimum(w, C_) * (R - mu_a)).mean()
            for t_ in (5, 10, 50): e[f"SWITCH_DR{t_}"] = (mu_pi + np.where(w <= t_, w, 0.0) * (R - mu_a)).mean()
            for h_ in (0.02, 0.10, 0.20):
                Kh = np.exp(-0.5 * ((A[None, :] - A[pi][:, None]) / h_) ** 2); Kh /= Kh.sum(1, keepdims=True); e[f"KIPS_h{h_}"] = ((Kh[ar, ai] / Li[ar, ai]) * R).mean()
            wmax = float(w.max()); w99 = float(np.percentile(w[hit], 99)) if hit.any() else 0.0; hitshare = float(hit.mean())
            for mode, pv_, wv_ in (("known", Li[ar, pi], w), ("est", Ph[ar, pi], wh)):
                psi_m = mu_pi + wv_ * (R - mu_a); cand = []
                for e_ in EG:
                    Sm = pv_ >= e_; z_ = Sm * psi_m; vh = z_.mean(); se_row = z_.std() / np.sqrt(N); se = _cse(z_, gidx) if a.group else se_row; ws = Sm * wv_; es = ws.sum() ** 2 / max((ws ** 2).sum(), 1e-9)
                    cand.append((e_, vh, se, es, (se / abs(vh) if abs(vh) > 1e-9 else np.inf), se_row))
                ok = [c for c in cand if c[4] <= CVMAX and c[3] >= ESSMIN]; flag = 0 if ok else 1; sel = ok[0] if ok else cand[-1]
                for nm, c in [("AST", sel)] + [(f"fixed{c[0]:.2f}", c) for c in cand]:
                    tg = Jg[c[0]][k]; arows.append(dict(tag=a.tag, scale=sc, policy=k, mode=mode, rule=nm, eps=c[0], flag=(flag if nm == "AST" else 0), rep=r, est=c[1], tgt=tg, err=c[1] - tg, cover=float(abs(c[1] - tg) <= 1.96 * c[2]), cover_row=float(abs(c[1] - tg) <= 1.96 * c[5]), share=tg / Vtrue[k], cv=c[4], ess=c[3]))
            ess = w.sum() ** 2 / max((w ** 2).sum(), 1e-9)
            ci = {}
            for en, vec in (("DR", psi), ("DR_supp(J_S)", S * psi)):
                bs = (np.array([vec[rng_b.integers(0, N, N)].mean() for _ in range(a.nboot)]) if not a.group else _cboot(vec, gidx)); ci[en] = np.percentile(bs, [2.5, 97.5])
            for en, ev in e.items():
                tgt = (Jse[0.01][k] if "supp01" in en else Jse[0.10][k] if "supp10" in en else Js[k]) if "(J_S)" in en else Vtrue[k]; c = ci.get(en)
                rows.append(dict(tag=a.tag, scale=sc, policy=k, est=en, rep=r, err=ev - tgt, tgt=tgt, ess=ess, wmax=wmax, w99=w99, hit=hitshare, supp=S.mean(), supp_est=Se.mean(), supp_miss=(S & ~Se).mean(), supp_false=(~S & Se).mean(), surv_supp=Ss[idx][ar, pi].mean(),
                                 cover=(np.nan if c is None else float(c[0] <= tgt <= c[1])), width=(np.nan if c is None else float(c[1] - c[0]))))
        if r % 5 == 4: print(f"[{a.tag}] scale={sc} rep={r+1}/{a.reps} {time.time()-t0:.0f}s", flush=True)
os.makedirs(a.out, exist_ok=True); res = pd.DataFrame(rows); res.to_csv(f"{a.out}/raw_{a.tag}.csv", index=False)
pd.DataFrame(arows).to_csv(f"{a.out}/ast_{a.tag}.csv", index=False)
g = res.groupby(["tag", "scale", "policy", "est"]).agg(wmax=("wmax", "mean"), w99=("w99", "mean"), hit=("hit", "mean"), bias=("err", "mean"), rmse=("err", lambda v: np.sqrt((v ** 2).mean())), tgt=("tgt", "first"),
        ess=("ess", "mean"), supp=("supp", "mean"), supp_est=("supp_est", "mean"), supp_miss=("supp_miss", "mean"), supp_false=("supp_false", "mean"), surv_supp=("surv_supp", "mean"), coverage=("cover", "mean"), width=("width", "mean")).reset_index()
g.to_csv(f"{a.out}/summary_{a.tag}.csv", index=False)
json.dump({"args": vars(a), "Vtrue": Vtrue, "implied_elasticity": el_impl, "oracle_action_shares": (np.bincount(POL["oracle"], minlength=NA) / n_all).tolist(), "engine_action_shares": (np.bincount(a_star, minlength=NA) / n_all).tolist()}, open(f"{a.out}/meta_{a.tag}.json", "w"), indent=1)
print(f"[{a.tag}] done {time.time()-t0:.0f}s", flush=True)
