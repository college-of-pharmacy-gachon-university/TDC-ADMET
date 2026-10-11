# Part of the ADMET-Stack revision analysis set.
# Table S29: baseline comparison using fixed configurations, with Holm correction.
# Inputs: data/si_tables.json
# -*- coding: utf-8 -*-
import json
import numpy as np
from scipy import stats
from pathlib import Path
import os
DATA = Path(os.environ.get('ADMET_DATA',
                           Path(__file__).resolve().parent.parent / 'data'))

T=json.load(open(DATA / "si_tables.json", encoding="utf-8"))
cfgs=sorted({n for e in T for n,_,_ in T[e]['rows']})
S={(e,n):v for e in T for n,v,_ in T[e]['rows']}
MIN={e:(T[e]['metric']=='MAE') for e in T}
# endpoint order & baselines from Table S26
B = {
 'Caco2_Wang':       dict(task='R', ai=0.285, trix=0.326, mini=0.350, maplight=0.287, sota=0.256),
 'HIA_Hou':          dict(task='C', ai=0.988, trix=0.985, mini=0.993, maplight=0.989, sota=0.993),
 'Pgp_Broccatelli':  dict(task='C', ai=0.935, trix=0.926, mini=None,  maplight=0.938, sota=0.938),
 'Bioavailability':  dict(task='C', ai=0.748, trix=0.749, mini=0.942, maplight=0.938, sota=0.942),
 'Lipophilicity':    dict(task='R', ai=0.456, trix=0.512, mini=0.456, maplight=0.525, sota=0.456),
 'Solubility':       dict(task='R', ai=0.761, trix=0.789, mini=0.741, maplight=0.789, sota=0.741),
 'BBB_Martins':      dict(task='C', ai=0.915, trix=0.906, mini=0.924, maplight=0.913, sota=0.924),
 'PPBR':             dict(task='R', ai=7.788, trix=8.200, mini=7.696, maplight=7.526, sota=7.440),
 'VDss_Lombardo':    dict(task='R', ai=0.627, trix=0.475, mini=0.535, maplight=0.713, sota=0.713),
 'CYP3A4_Sub.':      dict(task='C', ai=0.662, trix=0.642, mini=0.663, maplight=0.647, sota=0.667),
 'CYP2D6_Sub.':      dict(task='C', ai=0.736, trix=0.699, mini=0.695, maplight=0.720, sota=0.736),
 'CYP2C9_Sub.':      dict(task='C', ai=0.441, trix=0.377, mini=0.474, maplight=0.434, sota=0.474),
 'CYP3A4_In.':       dict(task='C', ai=0.904, trix=0.884, mini=0.877, maplight=0.916, sota=0.916),
 'CYP2D6_In.':       dict(task='C', ai=0.739, trix=0.718, mini=0.719, maplight=0.790, sota=0.790),
 'CYP2C9_In.':       dict(task='C', ai=0.839, trix=0.789, mini=0.823, maplight=0.859, sota=0.859),
 'Half_Life':        dict(task='R', ai=0.547, trix=0.372, mini=0.495, maplight=0.557, sota=0.576),
 'Clearance H.':     dict(task='R', ai=0.440, trix=0.447, mini=0.446, maplight=0.498, sota=0.536),
 'Clearance M.':     dict(task='R', ai=0.625, trix=0.552, mini=0.628, maplight=0.630, sota=0.630),
 'LD50_Zhu':         dict(task='R', ai=0.552, trix=0.573, mini=0.585, maplight=0.633, sota=0.552),
 'hERG':             dict(task='C', ai=0.874, trix=0.836, mini=0.846, maplight=0.880, sota=0.880),
 'AMES':             dict(task='C', ai=0.871, trix=0.870, mini=0.849, maplight=0.869, sota=0.871),
 'DILI':             dict(task='C', ai=0.925, trix=0.906, mini=0.956, maplight=0.917, sota=0.956),
}
eps=list(B.keys())
assert set(eps)==set(T.keys()), set(eps)^set(T.keys())

def stackval(e, mode):
    if mode=='best':
        return min(S[(e,c)] for c in cfgs) if MIN[e] else max(S[(e,c)] for c in cfgs)
    return S[(e,mode)]

def improvement(e, ours, base_):
    if MIN[e]: return (base_-ours)/base_*100
    return (ours-base_)/base_*100

def holm(ps):
    idx=np.argsort(ps); adj=np.empty(len(ps)); m=len(ps); prev=0
    for r,i in enumerate(idx):
        v=min(1,(m-r)*ps[i]); prev=max(prev,v); adj[i]=prev
    return adj

def rank_biserial(d):
    d=np.array([x for x in d if x!=0]); n=len(d)
    r=stats.rankdata(np.abs(d)); Rp=r[d>0].sum(); Rm=r[d<0].sum()
    return (Rp-Rm)/(n*(n+1)/2)

def run(mode,label):
    print(f'\n===== ADMET-Stack value = {label} =====')
    rows=[]; ps=[]; keys=[]
    for bl,key in [('ADMET-AI','ai'),('ADMETrix','trix'),('MiniMol','mini'),('MapLight+GNN','maplight')]:
        for cat,sel in [('Cls.','C'),('Reg.','R')]:
            d=[improvement(e,stackval(e,mode),B[e][key]) for e in eps
               if B[e][key] is not None and B[e]['task']==sel]
            d=np.array(d)
            p=stats.wilcoxon(d).pvalue
            rows.append((bl,cat,len(d),d.mean(),d.std(ddof=1),np.median(d),(d>0).sum(),p,rank_biserial(d)))
            ps.append(p); keys.append((bl,cat))
    adj=holm(np.array(ps))
    # BH
    o=np.argsort(ps); bh=np.empty(len(ps)); m=len(ps); prev=1
    for r in range(m-1,-1,-1):
        i=o[r]; v=min(1,m/(r+1)*ps[i]); prev=min(prev,v); bh[i]=prev
    print(f'{"baseline":14s}{"cat":6s}{"n":>3s}{"mean%":>9s}{"SD":>8s}{"med%":>8s}{"win":>5s}{"p_raw":>9s}{"p_Holm":>9s}{"p_BH":>8s}{"r_rb":>8s}')
    for (bl,cat,n,mu,sd,md,w,p,rb),a,b_ in zip(rows,adj,bh):
        print(f'{bl:14s}{cat:6s}{n:3d}{mu:9.2f}{sd:8.2f}{md:8.2f}{w:5d}{p:9.4f}{a:9.4f}{b_:8.4f}{rb:8.2f}')
    # win counts vs SOTA
    w=sum(1 for e in eps if (stackval(e,mode)<B[e]['sota'] if MIN[e] else stackval(e,mode)>B[e]['sota']))
    t=sum(1 for e in eps if abs(stackval(e,mode)-B[e]['sota'])<1e-9)
    print(f'  beats TDC SOTA on {w}/22 endpoints (ties {t})')
    tot=0; win=0
    for e in eps:
        for key in ['ai','trix','mini','maplight']:
            if B[e][key] is None: continue
            tot+=1
            ours=stackval(e,mode)
            if (ours<B[e][key]) if MIN[e] else (ours>B[e][key]): win+=1
    print(f'  pairwise wins {win}/{tot} = {win/tot*100:.1f}%')

run('best','per-endpoint best of 54 (as published)')
run('M_A_E_S_R_MC_TI','FIXED single config M_A_E_S_R_MC_TI (best overall MCDA rank)')
run('M_A_E_R','FIXED single config M_A_E_R')
run('R','FIXED single config R (RDKit descriptors only)')
