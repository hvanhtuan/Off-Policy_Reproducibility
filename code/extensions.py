# (1) 500 versus 200 bootstrap resamples; (2) bias of the single-model direct method against sample size under uniform logging.
# Run from the package root: python3 code/extensions.py
import pandas as pd, numpy as np, json
# 1) bootstrap 500 vs 200: coverage of DR and J_S for oracle on the same 24 reps (reps 0-23 of the base run)
b5=pd.read_csv('results/benchmark/raw_nh_boot500.csv')
b2=pd.read_csv('results/benchmark/raw_nh_base.csv'); b2=b2[b2.rep<24]
rows=[]
for sc in [0.01,0.10]:
    for est,lab in [('DR','DR'),('DR_supp(J_S)','J_S')]:
        r=[]
        for nm,d in [('200',b2),('500',b5)]:
            x=d[(d.policy=='oracle')&(d.est==est)&np.isclose(d.scale,sc)]
            r+= [f"{x.cover.mean():.2f}", f"{x.width.mean():.1f}", len(x)]
        rows.append([f"{lab}, scale {sc:.2f}"]+[r[0],r[1],r[3],r[4]])
        print(lab,sc,r)
# point estimates identical?
pa=b2[(b2.policy=='oracle')&(b2.est=='DR')&np.isclose(b2.scale,0.10)].sort_values('rep').err.values[:24]
pb=b5[(b5.policy=='oracle')&(b5.est=='DR')&np.isclose(b5.scale,0.10)].sort_values('rep').err.values
print('same samples:',np.allclose(pa,pb))
json.dump(rows,open('results/derived/bootstrap_500.json','w'))
# 2) DM learning curve under uniform logging (oracle)
lc=[]
def row(d,lab):
    out=[lab]
    for e in ['DM','DR','KIPS']:
        x=d[(d.policy=='oracle')&(d.est==e)&np.isinf(d.scale)].err
        out+= [f"{x.mean():+.1f}", f"{x.std(ddof=1):.1f}"]
    out.append(int(d[(d.policy=='oracle')&(d.est=='DM')&np.isinf(d.scale)].shape[0])); return out
base=pd.read_csv('results/benchmark/raw_nh_base.csv'); base=base[base.rep<10]
lc=[row(base,'20,000'),row(pd.read_csv('results/benchmark/raw_nh_n40k.csv'),'40,000'),row(pd.read_csv('results/benchmark/raw_nh_n80k.csv'),'80,000')]
for r in lc: print(r)
json.dump(lc,open('results/derived/learning_curve.json','w'))
