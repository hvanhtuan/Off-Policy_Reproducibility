# Extension of the base configuration from 40 to 120 replications (scales 0.03 and 0.10).
# Run from the package root: python3 code/extension_120.py
import pandas as pd, numpy as np, json
ESTS=['IPS','SNIPS','DM','DR','DR_clip20','DR_estprop','SWITCH_DR20','KIPS']
dn={'DR_clip20':'DR-clip','DR_estprop':'DR-est','SWITCH_DR20':'SWITCH-DR'}
a=pd.read_csv('results/benchmark/raw_nh_base.csv'); a=a[np.isclose(a.scale,0.03)|np.isclose(a.scale,0.10)]; x1=pd.read_csv('results/benchmark/raw_nh_basex1.csv'); x2=pd.read_csv('results/benchmark/raw_nh_basex2.csv')
a['tag']='nh_base'; d=pd.concat([a,x1,x2]); d=d[d.est.isin(ESTS)]
print(d.groupby('tag').rep.nunique().to_dict())
rng=np.random.default_rng(11); out={}
res=[]
for sc in [0.03,0.10]:
  x=d[np.isclose(d.scale,sc)]
  for pol in ['oracle','oracle_cap10','shift_up','hold','engine']:
    p=x[x.policy==pol].pivot_table(index='rep',columns='est',values='err'); n=len(p)
    rm=np.sqrt((p**2).mean()); bias=p.mean(); sd=p.std(ddof=1)
    bo=rm.drop('KIPS').idxmin()
    dif=(p['KIPS']**2-p[bo]**2).values
    bs=np.array([dif[rng.integers(0,n,n)].mean() for _ in range(4000)]); lo,hi=np.percentile(bs,[2.5,97.5])
    res.append(dict(scale=sc,pol=pol,n=n,**{e:round(float(rm[e]),1) for e in ESTS},best=bo,dmse=float(dif.mean()),lo=float(lo),hi=float(hi),kips_bias=float(bias['KIPS']),kips_sd=float(sd['KIPS']),bo_bias=float(bias[bo]),bo_sd=float(sd[bo])))
R=pd.DataFrame(res); R.to_csv('results/derived/extension_120.csv',index=False)
print(R[['scale','pol','n','KIPS','SNIPS','DR','DR_clip20','best','dmse','lo','hi']].round(2).to_string())
