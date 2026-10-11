# Part of the ADMET-Stack revision analysis set.
# Winner's-curse bias in the retention reference and the MCDA block ranking.
# Inputs: data/si_tables.json
# -*- coding: utf-8 -*-
import json, itertools
import numpy as np
from scipy import stats
from pathlib import Path
import os
DATA = Path(os.environ.get('ADMET_DATA',
                           Path(__file__).resolve().parent.parent / 'data'))

T=json.load(open(DATA / "si_tables.json", encoding="utf-8"))
eps=list(T); cfgs=sorted({n for e in eps for n,_,_ in T[e]['rows']})
S={(e,n):v for e in eps for n,v,_ in T[e]['rows']}
SD={(e,n):s for e in eps for n,_,s in T[e]['rows']}
MIN={e:(T[e]['metric']=='MAE') for e in eps}

# --- MCDA total rank score ---
dev={}
for e in eps:
    for c in cfgs:
        dev[(e,c)] = abs(S[(e,c)]) if MIN[e] else abs(1-S[(e,c)])
norm={}
for e in eps:
    ds=[dev[(e,c)] for c in cfgs]; lo,hi=min(ds),max(ds)
    for c in cfgs: norm[(e,c)]=(dev[(e,c)]-lo)/(hi-lo)
total={c: sum(norm[(e,c)] for e in eps) for c in cfgs}
order=sorted(cfgs,key=lambda c:total[c])
print('=== MCDA total rank score (verification) ===')
for c in ['M_A_E_S_R_MC_TI','R','M_A_E']:
    print(f'  {c:22s} {total[c]:.3f}  (rank {order.index(c)+1}/54)')
print(f'  median total score = {np.median(list(total.values())):.3f}')
print(f'  top5: {[(c,round(total[c],3)) for c in order[:5]]}')

# --- matched-pair marginal block contributions with effect sizes + CI ---
core=[c for c in cfgs if c.startswith('M_A_E')]
print(f'\n=== core configurations: {len(core)} ===')
def parse(c):
    toks=c.split('_')
    s = {'S':'S' in toks, 'R':'R' in toks, 'MC':'MC' in toks, 'TI':'TI' in toks,
         'EM32':'EM32' in toks, 'EM64':'EM64' in toks}
    return s
def build(sig):
    parts=['M','A','E']
    if sig['S']: parts.append('S')
    if sig['R']: parts.append('R')
    if sig['MC']: parts.append('MC')
    if sig['TI']: parts.append('TI')
    if sig['EM32']: parts.append('EM32')
    if sig['EM64']: parts.append('EM64')
    return '_'.join(parts)
def holm(ps):
    idx=np.argsort(ps); adj=np.empty(len(ps)); m=len(ps); prev=0
    for r,i in enumerate(idx):
        v=min(1,(m-r)*ps[i]); prev=max(prev,v); adj[i]=prev
    return adj
def rb(d):
    d=np.array([x for x in d if x!=0]); n=len(d)
    r=stats.rankdata(np.abs(d)); return (r[d>0].sum()-r[d<0].sum())/(n*(n+1)/2)
res=[]
for blk in ['S','R','MC','TI','EM32','EM64']:
    pairs=[]
    for c in core:
        sg=parse(c)
        if not sg[blk]: continue
        sg2=dict(sg); sg2[blk]=False
        c2=build(sg2)
        if c2 in total: pairs.append((c,c2))
    d=np.array([total[a]-total[b] for a,b in pairs])
    with_=np.array([total[a] for a,b in pairs])
    p=stats.wilcoxon(d).pvalue
    # bootstrap CI of mean difference
    rng=np.random.default_rng(0)
    bs=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(20000)])
    res.append((blk,len(pairs),with_.mean(),with_.std(ddof=1),d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),p,rb(d)))
adj=holm(np.array([r[7] for r in res]))
print(f'{"block":7s}{"pairs":>6s}{"meanTot":>9s}{"SD":>7s}{"dMean":>8s}{"95%CI":>18s}{"p_raw":>9s}{"p_Holm":>9s}{"r_rb":>7s}')
for (blk,n,mu,sd,dm,lo,hi,p,r),a in zip(res,adj):
    print(f'{blk:7s}{n:6d}{mu:9.3f}{sd:7.3f}{dm:8.3f}   [{lo:6.3f},{hi:6.3f}]{p:9.4f}{a:9.4f}{r:7.2f}')

# --- winner's curse estimate (parametric, seed noise only) ---
print('\n=== 4. WINNER-CURSE (seed-noise-only lower bound) ===')
rng=np.random.default_rng(1)
tot_bias=[]
for e in eps:
    mu=np.array([S[(e,c)] for c in cfgs])
    se=np.array([SD[(e,c)] for c in cfgs])/np.sqrt(5)
    sign=-1 if MIN[e] else 1
    obs_best=(mu*sign).max()
    sims=[]
    for _ in range(4000):
        draw=mu+rng.normal(0,se)
        k=int(np.argmax(draw*sign))
        sims.append(mu[k]*sign)          # TRUE value of the config selected on noisy data
    bias=obs_best-np.mean(sims)
    tot_bias.append((e,obs_best*sign,np.mean(sims)*sign,bias))
    print(f'  {e:22s} reported_best={obs_best*sign:8.4f} expected_true_of_selected={np.mean(sims)*sign:8.4f} optimism={bias:.4f}')
