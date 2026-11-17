"""INS-03: overlap diagnostic on REAL Segura-Gisbert renewal data. Usage: python3 -I support_diagnostic.py CSV OUT_DIR
Action = realised relative premium change vs the previous contract-year row of the same ID, binned to the 6-level grid.
CAVEAT: a row at t>1 exists only if the customer renewed, so the observed price change is a survivor-selected sample and
the price offered to eventual lapsers is NOT observed."""
import sys, json, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.model_selection import GroupKFold
CSV, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(CSV, sep=";")
for c in ["Date_last_renewal", "Date_birth", "Date_driving_licence"]: df[c] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
df = df.sort_values(["ID", "Date_last_renewal"]); g = df.groupby("ID")
df["prevP"] = g.Premium.shift(1); df["prev_nclaims"] = g.N_claims_year.shift(1); df["prev_cost"] = g.Cost_claims_year.shift(1)
d = df.dropna(subset=["prevP"]).copy(); d = d[(d.prevP > 0) & (d.Premium > 0)]
d["chg"] = d.Premium / d.prevP - 1
edges = [-np.inf, -.075, -.025, .025, .075, .125, np.inf]; labels = ["<=-10%", "-5%", "0%", "+5%", "+10%", ">=+15%"]
d["a"] = pd.cut(d.chg, edges, labels=False).astype(int)
d["age"] = (d.Date_last_renewal - d.Date_birth).dt.days / 365.25; d["lic"] = (d.Date_last_renewal - d.Date_driving_licence).dt.days / 365.25
d["veh_age"] = d.Date_last_renewal.dt.year - d.Year_matriculation; d["fuel"] = (d.Type_fuel == "D").astype(float); d["logPprev"] = np.log(d.prevP)
F = ["age","lic","veh_age","Seniority","Policies_in_force","Max_policies","Max_products","Payment","Distribution_channel","Type_risk","Area",
     "Second_driver","Power","Cylinder_capacity","Value_vehicle","N_doors","fuel","Length","Weight","prev_nclaims","prev_cost","logPprev"]
X = d[F].astype(float).values; y = d.a.values; grp = d.ID.values; n = len(d)
P = np.zeros((n, 6))
for tr, te in GroupKFold(5).split(X, y, grp):
    m = HGC(max_iter=100, learning_rate=.08, early_stopping=False, random_state=0).fit(X[tr], y[tr]); P[np.ix_(te, m.classes_)] = m.predict_proba(X[te])
P = np.clip(P, 1e-4, 1); P /= P.sum(1, keepdims=True)
res = {"n_rows": int(n), "n_ids": int(d.ID.nunique()), "action_share": {l: float((y == k).mean()) for k, l in enumerate(labels)},
       "mean_max_propensity": float(P.max(1).mean()), "share_max_prop_ge_0.9": float((P.max(1) >= .9).mean()),
       "multiclass_logloss": float(-np.log(P[np.arange(n), y]).mean()), "baseline_logloss_marginal": float(-np.log(np.bincount(y) / n)[y].mean())}
sup = []
for k, l in enumerate(labels):
    row = {"policy": f"everyone gets {l}"}
    for eps in (.01, .05, .10): row[f"support_frac_eps{eps}"] = float((P[:, k] >= eps).mean())
    hit = (y == k); w = hit / P[:, k]; row["ESS_IPS"] = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-9)); row["ESS_ratio"] = row["ESS_IPS"] / n; sup.append(row)
# cap-at-+10% policy: only rows with observed >=+15% bin change action to +10%
aff = (y == 5); row = {"policy": "cap price rise at +10%", "affected_share": float(aff.mean())}
for eps in (.01, .05, .10): row[f"support_frac_among_affected_eps{eps}"] = float((P[aff, 4] >= eps).mean())
sup.append(row)
aff2 = (y >= 3); row = {"policy": "freeze price (a=0) for all with rise>=+5%", "affected_share": float(aff2.mean())}
for eps in (.01, .05, .10): row[f"support_frac_among_affected_eps{eps}"] = float((P[aff2, 2] >= eps).mean())
sup.append(row)
pd.DataFrame(sup).to_csv(f"{OUT}/overlap_real_support.csv", index=False); json.dump(res, open(f"{OUT}/overlap_real_meta.json", "w"), indent=1)
print(json.dumps(res, indent=1)); print(pd.DataFrame(sup).round(3).to_string())
