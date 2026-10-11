# Part of the ADMET-Stack revision analysis set.
# Table S24: feature dimensionality of each of the 54 configurations.
# Inputs: run artifacts; set the path constant at the top
# -*- coding: utf-8 -*-
import os,re,csv,json
import pandas as pd
ROOT=r'F:\00_00_TDC_data_reorganized'
MAP={'morgan':'M','avalon':'A','erg':'E','selfies':'S','rdkit':'R','maccs':'MC','tfidf':'TI','embed':'EM32','embed64':'EM64'}
ORDER=['M','A','E','S','R','MC','TI','EM32','EM64']
def cfg(name):
    m=re.match(r'^\d+_\d+_TDC_(.+)_variance$', name)
    if not m: return None
    toks=m.group(1).split('_'); out=[]
    for t in toks:
        if t=='embed64': out.append('EM64')
        elif t in MAP: out.append(MAP[t])
        else: return None
    return '_'.join(sorted(set(out), key=ORDER.index))
rows=[]
for ep in sorted(os.listdir(ROOT)):
    d=os.path.join(ROOT,ep)
    if not os.path.isdir(d): continue
    for fol in sorted(os.listdir(d)):
        c=cfg(fol)
        if not c: continue
        f=os.path.join(d,fol,'Outputs','data_shapes.csv')
        if not os.path.exists(f): continue
        t=pd.read_csv(f)
        tr=t.loc[t.Split=='Train','X_shape'].iloc[0]
        n,p=re.match(r'\((\d+), (\d+)\)',tr).groups()
        rows.append(dict(endpoint=ep,config=c,n_train=int(n),n_features=int(p)))
df=pd.DataFrame(rows)
df.to_csv(r'C:\Users\omidm\AppData\Local\Temp\claude\E--ADMET-Stack-project-for-review261003\ab97d4a4-4d24-44b2-8541-9304c48e8355\scratchpad\config_dims.csv',index=False)
print('rows',len(df),'endpoints',df.endpoint.nunique(),'configs',df.config.nunique())
# derive per-endpoint S and TI dimensions
piv=df.pivot(index='endpoint',columns='config',values='n_features')
base={'M':1024,'A':1024,'E':315,'R':217,'MC':167,'EM32':32,'EM64':64}
out=[]
for ep in piv.index:
    S=piv.loc[ep,'S'] if 'S' in piv.columns else None
    TI=piv.loc[ep,'TI'] if 'TI' in piv.columns else None
    out.append((ep,int(S),int(TI),int(piv.loc[ep,'M_A_E']),int(piv.loc[ep,'R']),int(piv.loc[ep,'M_A_E_S_R_MC_TI'])))
print(f'{"endpoint":34s}{"S_dim":>7s}{"TI_dim":>8s}{"M_A_E":>8s}{"R":>6s}{"best7(M_A_E_S_R_MC_TI)":>24s}')
for r in out: print(f'{r[0]:34s}{r[1]:7d}{r[2]:8d}{r[3]:8d}{r[4]:6d}{r[5]:24d}')
mn=[r[1] for r in out]; ti=[r[2] for r in out]
print(f'\nS dim range: {min(mn)}-{max(mn)}   TI dim range: {min(ti)}-{max(ti)}')
print('max total dims (any config, any endpoint):', int(df.n_features.max()), '| min:', int(df.n_features.min()))
