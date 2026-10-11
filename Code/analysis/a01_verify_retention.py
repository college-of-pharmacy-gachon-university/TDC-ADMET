# Part of the ADMET-Stack revision analysis set.
# Checks the 94.3% retention figure against the published 54x22 grid.
# Inputs: data/si_tables.json
import json, statistics as st
import numpy as np
T=json.load(open(r'C:\Users\omidm\AppData\Local\Temp\claude\E--ADMET-Stack-project-for-review261003\ab97d4a4-4d24-44b2-8541-9304c48e8355\scratchpad\si_tables.json'))
eps=list(T.keys())
cfgs=sorted({n for e in eps for n,_,_ in T[e]['rows']})
print('n configs', len(cfgs))
score={}; sd={}
for e in eps:
    for n,v,s in T[e]['rows']:
        score[(e,n)]=v; sd[(e,n)]=s
def best(e):
    vals=[score[(e,c)] for c in cfgs]
    return min(vals) if T[e]['metric']=='MAE' else max(vals)
def ret(e,c):
    b=best(e)
    return (b/score[(e,c)]) if T[e]['metric']=='MAE' else (score[(e,c)]/b)
for c in ['R','M_A_E','M_A_E_R','M_A_E_S_R_MC_TI']:
    r=[ret(e,c) for e in eps]
    ge95=sum(1 for x in r if x>=0.95)
    isbest=sum(1 for e in eps if abs(ret(e,c)-1.0)<1e-12)
    print(f'{c:20s} mean={np.mean(r)*100:.1f}% median={np.median(r)*100:.1f}% >=95%:{ge95}/22 best:{isbest}/22')
print()
print('per-endpoint retention for R:')
for e in eps:
    print(f'  {e:22s} {T[e]["metric"]:9s} R={score[(e,"R")]:.4f} best={best(e):.4f} ret={ret(e,"R")*100:.1f}%')
