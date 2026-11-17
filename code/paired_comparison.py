# Paired bootstrap of the difference in squared error between KIPS and the best other estimator, SNIPS and DR-clip (oracle etc., scale 0.10).
# Run from the package root: python3 code/paired_comparison.py
import pandas as pd, numpy as np, json
cfg=[("Base","base"),("e0=2","e2"),("e0=3","e3"),("e0=10","e10"),("e0=12","e12"),("Probit link","probit"),("Expense 10%","exp10"),("Expense 30%","exp30"),("Retention value 0.5 P0","clv50"),("9-level grid","grid9")]
rng=np.random.default_rng(7)
def load(c):
    d=pd.read_csv(f'results/benchmark/raw_nh_{c}.csv'); return d
ESTS=['IPS','SNIPS','DM','DR','DR_clip20','DR_estprop','SWITCH_DR20','KIPS']
out=[]; rows=[]
for name,c in cfg:
    d=load(c); d=d[np.isclose(d.scale,0.10)&d.est.isin(ESTS)]
    for pol in ['oracle','oracle_cap10','shift_up','hold','engine']:
        p=d[d.policy==pol].pivot_table(index='rep',columns='est',values='err')
        n=len(p); mse=(p**2).mean(); best_other=mse.drop('KIPS').idxmin()
        for comp in dict.fromkeys([best_other,'SNIPS','DR_clip20']):
            dif=(p['KIPS']**2-p[comp]**2).values  # negative favours KIPS
            bs=np.array([dif[rng.integers(0,n,n)].mean() for _ in range(2000)])
            lo,hi=np.percentile(bs,[2.5,97.5])
            out.append(dict(cfg=name,pol=pol,n=n,comp=comp,which=('best' if comp==best_other else comp),dmse=dif.mean(),lo=lo,hi=hi,sig=(hi<0 or lo>0),
                kips_rmse=float(np.sqrt(mse['KIPS'])),comp_rmse=float(np.sqrt(mse[comp])),kips_best=(best_other is None or mse['KIPS']<=mse.drop('KIPS').min())))
D=pd.DataFrame(out); D.to_csv('results/derived/paired_comparison.csv',index=False)
for w in ['best','SNIPS','DR_clip20']:
    s=D[D.comp==w] if w!='best' else D[D.which=='best']
    print(w,'cases',len(s),'KIPS lower MSE',(s.dmse<0).sum(),'KIPS sig better',((s.dmse<0)&s.sig).sum(),'KIPS sig worse',((s.dmse>0)&s.sig).sum())
# oracle only table
o=D[(D.pol=='oracle')&(D.which=='best')][['cfg','comp','kips_rmse','comp_rmse','dmse','lo','hi','sig']]
print(o.round(2).to_string())
o2=D[(D.pol=='oracle')&(D.comp=='SNIPS')][['cfg','dmse','lo','hi','sig']]; print(o2.round(1).to_string())
