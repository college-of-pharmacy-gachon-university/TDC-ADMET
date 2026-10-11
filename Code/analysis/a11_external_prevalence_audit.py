# Part of the ADMET-Stack revision analysis set.
# Table S31: class prevalence, majority-class rate and threshold-corrected metrics.
# Inputs: none (self-contained)
# -*- coding: utf-8 -*-
rows=[
 ("AMES",25536,5810,19726,0.836,0.917,0.786,"In-house"),
 ("AMES",414,75,339,0.725,0.707,0.362,"PharmaBench"),
 ("BBB",654,361,293,0.660,0.741,0.763,"PharmaBench"),
 ("BBB (test)",66,44,22,0.833,0.920,0.958,"AutoML"),
 ("BBB (all)",323,228,95,0.851,0.921,0.962,"AutoML"),
 ("CYP2C9_In. (test)",387,304,83,0.429,0.949,0.984,"AutoML"),
 ("CYP2C9_In.",1928,1450,478,0.484,0.922,0.967,"AutoML"),
 ("CYP2D6_In. (test)",988,351,637,0.713,0.793,0.674,"AutoML"),
 ("CYP2D6_In.",4953,1590,3363,0.733,0.809,0.646,"AutoML"),
 ("CYP3A4_In. (test)",1186,494,692,0.709,0.795,0.707,"AutoML"),
 ("CYP3A4_In.",5994,2443,3551,0.720,0.795,0.720,"AutoML"),
 ("hERG (test)",1316,671,645,0.576,0.675,0.669,"AutoML"),
 ("hERG",12838,6444,6394,0.571,0.668,0.656,"AutoML"),
 ("P_gp (test)",528,289,239,0.590,0.664,0.706,"AutoML"),
 ("P_gp",2649,1418,1231,0.629,0.684,0.694,"AutoML"),
 ("AMES",82,56,26,0.732,0.854,0.933,"Admetica"),
 ("BBB",4924,2566,2358,0.429,0.269,0.387,"Admetica"),
 ("CYP2C19_Sub.",210,122,88,0.419,0.574,0.657,"Admetica"),
 ("CYP2C9_In.",647,58,589,0.866,0.763,0.424,"Admetica"),
 ("CYP2C9_Sub.",217,124,93,0.429,0.622,0.668,"Admetica"),
 ("CYP2D6_In.",632,202,430,0.786,0.895,0.783,"Admetica"),
 ("CYP2D6_Sub.",229,149,80,0.445,0.686,0.801,"Admetica"),
 ("CYP3A4_In.",546,123,423,0.833,0.891,0.806,"Admetica"),
 ("CYP3A4_Sub.",292,249,43,0.586,0.635,0.908,"Admetica"),
 ("hERG",20982,18173,2809,0.763,0.634,0.917,"Admetica"),
 ("P_gp",4,2,2,1.000,1.000,1.000,"Admetica"),
 ("HIA Hou",78,37,41,0.526,0.779,0.773,"Admetica"),
 ("Bioavailability",462,162,300,0.351,0.400,0.307,"Admetica"),
]
print(f'{"property":20s}{"source":12s}{"n":>7s}{"prev":>7s}{"ACC":>7s}{"maj":>7s}{"AUROC":>7s}{"AUPRC":>7s}{"AP-prev":>9s}{"flag":>6s}')
for p,n,a,i,acc,roc,prc,src in rows:
    prev=a/n; maj=max(a,i)/n; lift=prc-prev
    flags=[]
    if roc<0.5: flags.append("ROC<0.5")
    if lift<=0: flags.append("AP<=prev")
    if acc<maj: flags.append("ACC<maj")
    if n<100: flags.append("tiny")
    print(f'{p:20s}{src:12s}{n:7d}{prev:7.3f}{acc:7.3f}{maj:7.3f}{roc:7.3f}{prc:7.3f}{lift:+9.3f}  {",".join(flags)}')
