# Part of the ADMET-Stack revision analysis set.
# Table S26: linear mixed-effects model with a random intercept per endpoint.
# Inputs: data/si_tables.json
# -*- coding: utf-8 -*-
"""Mixed-effects re-analysis of block marginal contributions (endpoint random effects),
addressing Reviewer 2 point 4 (dependence assumptions)."""
import json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import os
DATA = Path(os.environ.get('ADMET_DATA',
                           Path(__file__).resolve().parent.parent / 'data'))

T=json.load(open(DATA / "si_tables.json", encoding="utf-8"))
eps=list(T); cfgs=sorted({n for e in eps for n,_,_ in T[e]['rows']})
S={(e,n):v for e in eps for n,v,_ in T[e]['rows']}
MIN={e:(T[e]['metric']=='MAE') for e in eps}
dev={(e,c): (abs(S[(e,c)]) if MIN[e] else abs(1-S[(e,c)])) for e in eps for c in cfgs}
norm={}
for e in eps:
    ds=[dev[(e,c)] for c in cfgs]; lo,hi=min(ds),max(ds)
    for c in cfgs: norm[(e,c)]=(dev[(e,c)]-lo)/(hi-lo)
core=[c for c in cfgs if c.startswith('M_A_E')]
rows=[]
for e in eps:
    for c in core:
        t=c.split('_')
        rows.append(dict(endpoint=e, cfg=c, y=norm[(e,c)],
            S=int('S' in t), R=int('R' in t), MC=int('MC' in t), TI=int('TI' in t),
            EM32=int('EM32' in t), EM64=int('EM64' in t)))
df=pd.DataFrame(rows)
print(f'rows = {len(df)} (= {len(eps)} endpoints x {len(core)} core configs)')
m=smf.mixedlm('y ~ S + R + MC + TI + EM32 + EM64', df, groups=df['endpoint']).fit()
print('\n--- Linear mixed model: normalised deviation ~ blocks + (1|endpoint) ---')
print('(negative coefficient = block improves performance)')
pr=m.summary().tables[1]
print(pr)
# convert per-endpoint coefficient to the manuscript's 22-endpoint total-rank-score scale
print('\nCoefficient x 22 endpoints = effect on total rank score:')
for b in ['S','R','MC','TI','EM32','EM64']:
    co=m.params[b]; se=m.bse[b]; p=m.pvalues[b]
    print(f'  {b:6s} delta_total = {co*22:+.3f}  (95% CI {co*22-1.96*se*22:+.3f} to {co*22+1.96*se*22:+.3f}), p = {p:.2e}')
