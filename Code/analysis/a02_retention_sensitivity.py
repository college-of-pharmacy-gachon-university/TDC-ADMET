# Part of the ADMET-Stack revision analysis set.
# Table S27: retention under five aggregation rules plus normalised regret.
# Inputs: data/si_tables.json
# -*- coding: utf-8 -*-
import json, itertools, sys
import numpy as np
from scipy import stats
from pathlib import Path
import os
DATA = Path(os.environ.get('ADMET_DATA',
                           Path(__file__).resolve().parent.parent / 'data'))

T=json.load(open(DATA / "si_tables.json", encoding="utf-8"))
eps=list(T.keys())
cfgs=sorted({n for e in eps for n,_,_ in T[e]['rows']})
S={}; SD={}
for e in eps:
    for n,v,s in T[e]['rows']: S[(e,n)]=v; SD[(e,n)]=s
MIN={e:(T[e]['metric']=='MAE') for e in eps}
def best(e): return min(S[(e,c)] for c in cfgs) if MIN[e] else max(S[(e,c)] for c in cfgs)
def worst(e): return max(S[(e,c)] for c in cfgs) if MIN[e] else min(S[(e,c)] for c in cfgs)
def ret(e,c): return best(e)/S[(e,c)] if MIN[e] else S[(e,c)]/best(e)
def rank(e,c):
    vals=sorted([S[(e,x)] for x in cfgs], reverse=not MIN[e])
    return vals.index(S[(e,c)])+1
def regret(e,c):  # min-max normalised deviation, 0=best 1=worst
    b,w=best(e),worst(e)
    return abs(S[(e,c)]-b)/abs(w-b)
def skill(e,c):
    """floor-corrected: AUROC floor .5; Spearman floor 0; AUPRC floor unknown->0; MAE ratio."""
    m=T[e]['metric']; b=best(e); v=S[(e,c)]
    if m=='AUROC': return (v-0.5)/(b-0.5)
    if m in ('Spearman','AUPRC'): return v/b
    return b/v

print('=== 1. RETENTION BY AGGREGATION METHOD ===')
focus=['R','M_A_E','M_A_E_R','M_A_E_S_R_MC_TI','M_A_E_R_MC']
hdr=f'{"config":22s}{"mean":>8s}{"median":>8s}{"geo":>8s}{"trim20":>8s}{"skillMean":>10s}{"meanRank":>10s}{"meanRegret":>11s}'
print(hdr)
for c in focus:
    r=np.array([ret(e,c) for e in eps]); sk=np.array([skill(e,c) for e in eps])
    rk=np.array([rank(e,c) for e in eps]); rg=np.array([regret(e,c) for e in eps])
    geo=np.exp(np.mean(np.log(r)))
    trim=stats.trim_mean(r,0.2)
    print(f'{c:22s}{r.mean()*100:7.1f}%{np.median(r)*100:7.1f}%{geo*100:7.1f}%{trim*100:7.1f}%{sk.mean()*100:9.1f}%{rk.mean():10.1f}{rg.mean():11.3f}')

print()
print('=== 2. RETENTION BY METRIC FAMILY (config R) ===')
fams={}
for e in eps: fams.setdefault(T[e]['metric'],[]).append(e)
for m,es in fams.items():
    r=[ret(e,'R') for e in es]
    print(f'  {m:9s} n={len(es):2d}  mean={np.mean(r)*100:5.1f}%  min={min(r)*100:5.1f}% ({es[int(np.argmin(r))]})')

print()
print('=== 3. IS R WITHIN NOISE OF BEST? (|R-best| vs pooled SD) ===')
cnt1=cnt2=0
for e in eps:
    bc=min(cfgs,key=lambda c:S[(e,c)]) if MIN[e] else max(cfgs,key=lambda c:S[(e,c)])
    d=abs(S[(e,'R')]-S[(e,bc)]); pooled=np.hypot(SD[(e,'R')],SD[(e,bc)])
    z=d/pooled if pooled>0 else np.inf
    if z<=1: cnt1+=1
    if z<=2: cnt2+=1
    print(f'  {e:22s} gap={d:8.4f} pooledSD={pooled:7.4f} z={z:7.2f} {"within1SD" if z<=1 else ("within2SD" if z<=2 else "")}')
print(f'  -> R within 1 pooled SD of best on {cnt1}/22; within 2 SD on {cnt2}/22')
