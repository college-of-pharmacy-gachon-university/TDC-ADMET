# Part of the ADMET-Stack revision analysis set.
# Table S30: selection optimism measured by repeated split-sample selection.
# Inputs: data/honest_selection.json
# -*- coding: utf-8 -*-
"""Summarise the split-sample honest-selection results."""
import json
import numpy as np
from pathlib import Path
import os
DATA = Path(os.environ.get('ADMET_DATA',
                           Path(__file__).resolve().parent.parent / 'data'))

R=json.load(open(DATA / "honest_selection.json", encoding="utf-8"))
print(f'{"endpoint":32s}{"n":>5s}{"reported":>9s}{"honest":>8s}{"oracleB":>8s}{"optimism":>9s}{"fixed7_B":>9s}{"vs fixed7":>10s}')
opt=[]; vs=[]
for ep,d in R.items():
    lower=(d['metric']=='MAE')
    o=(d['sel']-d['oracle']) if lower else (d['oracle']-d['sel'])      # >=0, selection optimism
    v=(d['M_A_E_S_R_MC_TI']-d['sel']) if lower else (d['sel']-d['M_A_E_S_R_MC_TI'])  # >0 => selection wins
    opt.append((ep,o,d['n_test'])); vs.append((ep,v,d['metric']))
    print(f'{ep:32s}{d["n_test"]:5d}{d["reported_full_test_best"]:9.4f}{d["sel"]:8.4f}{d["oracle"]:8.4f}{o:9.4f}{d["M_A_E_S_R_MC_TI"]:9.4f}{v:+10.4f}')
o=np.array([x[1] for x in opt]); n=np.array([x[2] for x in opt])
print(f'\nselection optimism: median {np.median(o):.4f}  range {o.min():.4f}-{o.max():.4f}')
small=o[n<300]; big=o[n>=1000]
print(f'  endpoints with n_test < 300  (n={len(small)}): median {np.median(small):.4f}')
print(f'  endpoints with n_test >= 1000 (n={len(big)}): median {np.median(big):.4f}')
w=[x for x in vs if x[1]>0]
print(f'\nhonest per-endpoint selection beats the pre-specified fixed configuration on {len(w)}/22 endpoints:')
for ep,v,m in sorted(w,key=lambda t:-t[1]): print(f'   {ep:32s}{m:9s}{v:+.4f}')
print('\nlargest losses:')
for ep,v,m in sorted(vs,key=lambda t:t[1])[:5]: print(f'   {ep:32s}{m:9s}{v:+.4f}')
