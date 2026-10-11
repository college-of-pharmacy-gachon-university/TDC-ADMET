# Part of the ADMET-Stack revision analysis set.
# Table S26: endpoint-cluster bootstrap for feature-block marginal effects.
# Inputs: data/si_tables.json
# -*- coding: utf-8 -*-
"""Cluster bootstrap over ENDPOINTS for block marginal contributions (vectorised).
Endpoints are the independent sampling units; configurations within an endpoint are not."""
import json
import numpy as np
from pathlib import Path
import os
DATA = Path(os.environ.get('ADMET_DATA',
                           Path(__file__).resolve().parent.parent / 'data'))

T=json.load(open(DATA / "si_tables.json", encoding="utf-8"))
eps=list(T); cfgs=sorted({n for e in eps for n,_,_ in T[e]['rows']})
S={(e,n):v for e in eps for n,v,_ in T[e]['rows']}
MIN={e:(T[e]['metric']=='MAE') for e in eps}
N=np.zeros((len(eps),len(cfgs)))
for i,e in enumerate(eps):
    d=np.array([abs(S[(e,c)]) if MIN[e] else abs(1-S[(e,c)]) for c in cfgs])
    N[i]=(d-d.min())/(d.max()-d.min())
ci={c:j for j,c in enumerate(cfgs)}
core=[c for c in cfgs if c.startswith('M_A_E')]
def has(c,b): return b in c.split('_')
def drop(c,b): return '_'.join([t for t in c.split('_') if t!=b])
rng=np.random.default_rng(5); NB=20000
boot_idx=rng.integers(0,len(eps),(NB,len(eps)))
print(f'{"block":7s}{"pairs":>6s}{"dMean":>9s}{"endpoint-cluster 95% CI":>28s}{"p_boot":>9s}')
for b in ['S','R','MC','TI','EM32','EM64']:
    P=[(ci[c],ci[drop(c,b)]) for c in core if has(c,b) and drop(c,b) in cfgs]
    A=np.array([p[0] for p in P]); Z=np.array([p[1] for p in P])
    D=N[:,A]-N[:,Z]                      # endpoints x pairs
    obs=D.sum(axis=0).mean()
    per_ep=D.mean(axis=1)                # endpoint-level mean pair difference
    bs=per_ep[boot_idx].sum(axis=1)      # cluster bootstrap of the 22-endpoint total
    lo,hi=np.percentile(bs,[2.5,97.5]); p=min(1.0,2*min((bs>=0).mean(),(bs<=0).mean()))
    print(f'{b:7s}{len(P):6d}{obs:+9.3f}   [{lo:+7.3f}, {hi:+7.3f}]{p:9.4f}')
