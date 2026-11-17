# Bias and standard deviation of each estimator for the oracle policy (base configuration).
# Run from the package root: python3 code/bias_sd.py
import pandas as pd, numpy as np, json
ESTS=[('IPS','IPS'),('SNIPS','SNIPS'),('DM','DM'),('DR','DR'),('DR_clip20','DR-clip'),('DR_estprop','DR-est'),('SWITCH_DR20','SWITCH-DR'),('KIPS','KIPS')]
d=pd.read_csv('results/benchmark/raw_nh_base.csv'); d=d[d.policy=='oracle']
rows=[]
for sc,lab in [(0.10,'0.10'),(np.inf,'Uniform')]:
    x=d[np.isclose(d.scale,sc) if np.isfinite(sc) else np.isinf(d.scale)]
    r=[lab]
    for e,l in ESTS:
        g=x[x.est==e].err.values; b=g.mean(); sd=g.std(ddof=1)
        r.append(f"{b:+.1f} / {sd:.1f}")
    rows.append(r)
print(rows)
json.dump(rows,open('results/derived/bias_sd.json','w'),ensure_ascii=False)
