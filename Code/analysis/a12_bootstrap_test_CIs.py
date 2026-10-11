# Part of the ADMET-Stack revision analysis set.
# Table S28: bootstrap CIs over test compounds for the reported performance.
# Inputs: run artifacts; set the path constant at the top
# -*- coding: utf-8 -*-
import os, glob, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from pathlib import Path
import os
DATA = Path(os.environ.get('ADMET_DATA',
                           Path(__file__).resolve().parent.parent / 'data'))

root=os.path.join(base,'outputs')
METRIC={'Caco2_Wang':'MAE','HIA_Hou':'AUROC','Pgp_Broccatelli':'AUROC','Bioavailability_Ma':'AUROC',
'Lipophilicity_AstraZeneca':'MAE','Solubility_AqSolDB':'MAE','BBB_Martins':'AUROC','PPBR_AZ':'MAE',
'VDss_Lombardo':'Spearman','CYP3A4_Substrate_CarbonMangels':'AUROC','CYP2D6_Substrate_CarbonMangels':'AUPRC',
'CYP2C9_Substrate_CarbonMangels':'AUPRC','CYP3A4_Veith':'AUPRC','CYP2D6_Veith':'AUPRC','CYP2C9_Veith':'AUPRC',
'Half_Life_Obach':'Spearman','Clearance_Hepatocyte_AZ':'Spearman','Clearance_Microsome_AZ':'Spearman',
'LD50_Zhu':'MAE','hERG':'AUROC','AMES':'AUROC','DILI':'AUROC'}
def metric(m, yt, yp):
    if m=='AUROC': return roc_auc_score(yt,yp)
    if m=='AUPRC': return average_precision_score(yt,yp)
    if m=='MAE':   return mean_absolute_error(yt,yp)
    return stats.spearmanr(yt,yp).statistic
rng=np.random.default_rng(7)
print(f'{"endpoint":34s}{"metric":9s}{"n_test":>7s}{"point":>9s}{"seedSD":>8s}{"boot 95% CI":>22s}')
out={}
for ep in sorted(os.listdir(root)):
    m=METRIC[ep]
    files=sorted(glob.glob(os.path.join(root,ep,'*predictions_seed*.csv')))
    if not files: print(ep,'NO FILES'); continue
    vals=[]; dfs=[]
    for f in files:
        d=pd.read_csv(f); dfs.append(d)
        col='y_proba' if 'y_proba' in d.columns else 'y_pred'
        vals.append(metric(m, d['y_true'].values, d[col].values))
    vals=np.array(vals)
    # pooled across seeds: average prediction then bootstrap test compounds
    col='y_proba' if 'y_proba' in dfs[0].columns else 'y_pred'
    yt=dfs[0]['y_true'].values
    P=np.column_stack([d[col].values for d in dfs])
    n=len(yt); bs=[]
    for _ in range(2000):
        idx=rng.integers(0,n,n)
        if m in ('AUROC','AUPRC') and len(np.unique(yt[idx]))<2: continue
        bs.append(np.mean([metric(m, yt[idx], P[idx,k]) for k in range(P.shape[1])]))
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f'{ep:34s}{m:9s}{n:7d}{vals.mean():9.4f}{vals.std(ddof=1):8.4f}   [{lo:7.4f}, {hi:7.4f}]')
    out[ep]=dict(metric=m,n=int(n),point=float(vals.mean()),seed_sd=float(vals.std(ddof=1)),ci=[float(lo),float(hi)])
json.dump(out,open(os.path.join(base,'bootstrap_ci.json'),'w'),indent=1)
