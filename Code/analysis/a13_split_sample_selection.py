# Part of the ADMET-Stack revision analysis set.
# Runs the split-sample selection that a08 summarises.
# Inputs: data/honest_selection.json
# -*- coding: utf-8 -*-
"""Split-sample (honest) evaluation of per-endpoint configuration selection - fast version."""
import pickle, json, sys, time
import numpy as np
from pathlib import Path
import os
DATA = Path(os.environ.get('ADMET_DATA',
                           Path(__file__).resolve().parent.parent / 'data'))

DB=pickle.load(open((DATA / "DB.pkl"),'rb'))
FIXED=['R','M_A_E','M_A_E_R','M_A_E_S_R_MC_TI']

def _rank(x):
    o=np.argsort(x, kind='mergesort'); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    # average ties
    xs=x[o]; i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[j+1]==xs[i]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2.0
        i=j+1
    return r
def auroc(y,p):
    r=_rank(p); npos=y.sum(); nneg=len(y)-npos
    return (r[y==1].sum()-npos*(npos+1)/2.0)/(npos*nneg)
def auprc(y,p):
    o=np.argsort(-p, kind='mergesort'); ys=y[o]
    tp=np.cumsum(ys); k=np.arange(1,len(ys)+1)
    prec=tp/k; rec=tp/ys.sum()
    drec=np.diff(np.concatenate(([0.0],rec)))
    return float((prec*drec).sum())
def spearman(y,p):
    a=_rank(y); b=_rank(p)
    a=a-a.mean(); b=b-b.mean()
    return float((a@b)/np.sqrt((a@a)*(b@b)))
def mae(y,p): return float(np.abs(y-p).mean())

def met(m,y,P,idx):
    yy=y[idx]
    if m in ('AUROC','AUPRC') and (yy.sum()==0 or yy.sum()==len(yy)): return None
    f={'AUROC':auroc,'AUPRC':auprc,'Spearman':spearman,'MAE':mae}[m]
    return float(np.mean([f(yy,P[idx,k]) for k in range(P.shape[1])]))

B=250
rng=np.random.default_rng(2026)
res={}
hdr=f'{"endpoint":32s}{"metric":9s}{"n":>5s}{"reported":>10s}{"honest":>9s}{"oracleB":>9s}{"R":>8s}{"M_A_E_R":>9s}{"best7":>8s}{"gain":>8s}'
print(hdr); sys.stdout.flush()
t0=time.time()
for ep,E in DB.items():
    m=E['metric']; y=np.asarray(E['y']); cfgs=sorted(E['preds']); lower=(m=='MAE')
    n=len(y); idx_all=np.arange(n)
    full={c:met(m,y,E['preds'][c],idx_all) for c in cfgs}
    rep_best=min(full.values()) if lower else max(full.values())
    acc={'sel':[],'oracle':[], **{f:[] for f in FIXED}}
    yb = y.astype(int) if m in ('AUROC','AUPRC') else None
    for b in range(B):
        if yb is not None:
            A=np.concatenate([rng.permutation(idx_all[yb==c])[:max(1,(yb==c).sum()//2)] for c in (0,1)])
        else:
            A=rng.permutation(idx_all)[:n//2]
        Bi=np.setdiff1d(idx_all,A)
        sA={c:met(m,y,E['preds'][c],A) for c in cfgs}
        sB={c:met(m,y,E['preds'][c],Bi) for c in cfgs}
        if any(v is None for v in sA.values()) or any(v is None for v in sB.values()): continue
        pick=min(sA,key=sA.get) if lower else max(sA,key=sA.get)
        acc['sel'].append(sB[pick]); acc['oracle'].append(min(sB.values()) if lower else max(sB.values()))
        for f in FIXED: acc[f].append(sB[f])
    out={k:float(np.mean(v)) for k,v in acc.items()}
    out['reported_full_test_best']=rep_best; out['n_test']=int(n); out['metric']=m
    out['full_test']={c:full[c] for c in FIXED}
    bf=min(out[f] for f in FIXED) if lower else max(out[f] for f in FIXED)
    out['gain_over_best_fixed']=(bf-out['sel']) if lower else (out['sel']-bf)
    res[ep]=out
    print(f'{ep:32s}{m:9s}{n:5d}{rep_best:10.4f}{out["sel"]:9.4f}{out["oracle"]:9.4f}{out["R"]:8.4f}{out["M_A_E_R"]:9.4f}{out["M_A_E_S_R_MC_TI"]:8.4f}{out["gain_over_best_fixed"]:+8.4f}')
    sys.stdout.flush()
json.dump(res, open((DATA / "honest_selection.json"),'w'), indent=1)
print('elapsed', round(time.time()-t0,1),'s')
