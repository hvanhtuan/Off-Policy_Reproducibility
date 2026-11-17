# Policy selection: probability of choosing the best policy, pairwise accuracy and regret at scale 0.10.
# Run from the package root: python3 code/policy_selection.py
import pandas as pd, numpy as np, itertools, json
ESTS=['DM','IPS','SNIPS','DR','KIPS']; POLS=['oracle','oracle_cap10','shift_up','hold','engine']
res={}
rows=[]
for lab,c in [("Base","base"),("Delay 30%, unadjusted","delay30_naive"),("Delay 30%, adjusted","delay30_adj")]:
    d=pd.read_csv(f'results/benchmark/raw_nh_{c}.csv'); d=d[np.isclose(d.scale,0.10)&d.est.isin(ESTS)].copy()
    d['est_v']=d.err+d.tgt
    tv=d.groupby('policy').tgt.mean(); best=tv.idxmax()
    for e in ESTS:
        p=d[d.est==e].pivot_table(index='rep',columns='policy',values='est_v')[POLS]
        n=len(p); ch=p.idxmax(axis=1); hit=(ch==best).astype(float)
        regret=tv[best]-tv[ch].values
        pairs=[(i,j) for i,j in itertools.combinations(POLS,2)]
        acc=np.mean([ ((p[i]>p[j])==(tv[i]>tv[j])).astype(float).values for i,j in pairs],axis=0)
        se=lambda x: float(np.std(x,ddof=1)/np.sqrt(len(x)))
        rows.append([lab,e,n,f"{100*hit.mean():.0f} ({100*se(hit):.0f})",f"{100*acc.mean():.0f} ({100*se(acc):.0f})",f"{regret.mean():.1f} ({se(regret):.1f})"])
    print(lab,'best',best,tv.round(1).to_dict())
for r in rows: print(r)
json.dump(rows,open('results/derived/policy_selection.json','w'),ensure_ascii=False)
