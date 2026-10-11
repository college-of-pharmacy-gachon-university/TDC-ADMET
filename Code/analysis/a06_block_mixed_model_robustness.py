# Part of the ADMET-Stack revision analysis set.
# Robustness of the mixed-effects block estimates.
# Inputs: data/si_tables.json
# -*- coding: utf-8 -*-
"""Sensitivity: block-contribution analysis after dropping the 4 configurations whose
saved feature-matrix width disagrees with their folder label."""
import json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
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
BAD={'M_A_E_R_TI_EM32','M_A_E_MC_TI_EM32','M_A_E_S_MC_TI_EM32','M_A_E_S_R_MC_TI_EM32'}
for drop in [set(), BAD]:
    core=[c for c in cfgs if c.startswith('M_A_E') and c not in drop]
    rows=[dict(endpoint=e,cfg=c,y=norm[(e,c)],
          S=int('S' in c.split('_')),R=int('R' in c.split('_')),MC=int('MC' in c.split('_')),
          TI=int('TI' in c.split('_')),EM32=int('EM32' in c.split('_')),EM64=int('EM64' in c.split('_')))
          for e in eps for c in core]
    df=pd.DataFrame(rows)
    m=smf.mixedlm('y ~ S + R + MC + TI + EM32 + EM64', df, groups=df['endpoint']).fit()
    lbl='ALL 48 core configs' if not drop else 'EXCLUDING the 4 mislabelled configs (44 core)'
    print(f'\n--- {lbl} (n={len(df)}) ---')
    for b in ['S','R','MC','TI','EM32','EM64']:
        co,se,p=m.params[b],m.bse[b],m.pvalues[b]
        print(f'  {b:6s} delta_total(x22) = {co*22:+7.3f}  95%CI [{(co-1.96*se)*22:+7.3f},{(co+1.96*se)*22:+7.3f}]  p={p:.2e}')
