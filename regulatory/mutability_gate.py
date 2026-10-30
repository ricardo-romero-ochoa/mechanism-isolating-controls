#!/usr/bin/env python3
"""G1E final test: does the 6-mer polymorphism association survive independent mutability calibration?

Primary threshold was frozen before the calibrated outcome: beta_6mer <= -0.05 log-odds per SD and
promoter-cluster bootstrap 95% CI upper bound < 0.  This combines direction, uncertainty, and a minimum
material effect, avoiding a pass based only on a tiny effect in a large position table.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; SEED=1729
REQ=['promoter_id','snp_present','local_gc','reference_base','distance_bin','log_mut_rate','sensitivity_6mer','sensitivity_occupancy','sensitivity_dinucleotide']
MATERIAL_BETA=-0.05

def design(df,score_col,include_mut=True):
    z=df[score_col].to_numpy(float); z=(z-np.mean(z))/max(np.std(z),1e-12)
    cols=[np.ones((len(df),1)),z[:,None],df[['local_gc']].to_numpy(float)]
    if include_mut: cols.append(df[['log_mut_rate']].to_numpy(float))
    ref=pd.get_dummies(df['reference_base'].astype(str),drop_first=True,dtype=float)
    dist=pd.get_dummies(df['distance_bin'].astype(str),drop_first=True,dtype=float)
    cols += [ref.to_numpy(),dist.to_numpy()]
    return np.column_stack(cols)

def logistic_irls(X,y,row_weight=None,maxit=80,ridge=1e-8):
    b=np.zeros(X.shape[1]); I=np.eye(X.shape[1]); I[0,0]=0
    rw=np.ones(len(y)) if row_weight is None else np.asarray(row_weight,float)
    for _ in range(maxit):
        eta=np.clip(X@b,-30,30); p=1/(1+np.exp(-eta)); w=np.maximum(p*(1-p),1e-9)*rw
        H=X.T@(X*w[:,None])+ridge*I; g=X.T@((y-p)*rw)-ridge*(I@b)
        step=np.linalg.solve(H,g); b+=step
        if np.max(np.abs(step))<1e-8: break
    return b

def fit_boot(df,score,n_boot,rng,include_mut):
    X=design(df,score,include_mut); y=df.snp_present.to_numpy(float); beta=float(logistic_irls(X,y)[1])
    cl=df.promoter_id.astype(str).to_numpy(); uniq,inv=np.unique(cl,return_inverse=True); boots=[]
    for _ in range(n_boot):
        counts=np.bincount(rng.integers(0,len(uniq),size=len(uniq)),minlength=len(uniq)); rw=counts[inv]
        try: boots.append(float(logistic_irls(X,y,rw)[1]))
        except np.linalg.LinAlgError: pass
    q=np.percentile(boots,[2.5,97.5]) if boots else [np.nan,np.nan]
    return {'log_odds_per_SD':beta,'odds_ratio':float(np.exp(beta)),'ci_low':float(q[0]),'ci_high':float(q[1]),'bootstrap_replicates_used':len(boots)}

def self_test():
    rng=np.random.default_rng(3); n=3000; prom=np.repeat(np.arange(100),30); mut=rng.normal(size=n); s=rng.normal(size=n); eta=-2-0.4*s+0.6*mut; y=rng.random(n)<1/(1+np.exp(-eta))
    df=pd.DataFrame({'promoter_id':prom,'snp_present':y.astype(int),'local_gc':rng.random(n),'reference_base':rng.choice(list('ACGT'),n),'distance_bin':rng.integers(0,12,n),'log_mut_rate':mut,'sensitivity_6mer':s,'sensitivity_occupancy':rng.random(n),'sensitivity_dinucleotide':rng.random(n)})
    r=fit_boot(df,'sensitivity_6mer',20,rng,True); assert r['log_odds_per_SD']<0
    print('PASS G1E mutability-gate synthetic self-test')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input',nargs='?',default=str(ROOT/'regulatory'/'gate1_positions.csv')); ap.add_argument('n_boot',nargs='?',type=int,default=1000); ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
    if a.self_test:self_test();return
    df=pd.read_csv(a.input); miss=[c for c in REQ if c not in df]
    if miss: raise SystemExit(f'missing required columns: {miss}')
    df=df.dropna(subset=REQ).copy();
    if not set(df.snp_present.unique()).issubset({0,1}):raise SystemExit('snp_present must be binary')
    rng=np.random.default_rng(SEED); models={}
    for col,name in [('sensitivity_6mer','6-mer'),('sensitivity_occupancy','occupancy'),('sensitivity_dinucleotide','dinucleotide')]:
        un=fit_boot(df,col,a.n_boot,rng,False); adj=fit_boot(df,col,a.n_boot,rng,True)
        adj['materially_negative']=bool(np.isfinite(adj['ci_high']) and adj['ci_high']<0 and adj['log_odds_per_SD']<=MATERIAL_BETA)
        adj['beta_fraction_of_uncalibrated']=float(adj['log_odds_per_SD']/un['log_odds_per_SD']) if abs(un['log_odds_per_SD'])>1e-12 else None
        models[name]={'uncalibrated':un,'mutability_calibrated':adj}
    passed=models['6-mer']['mutability_calibrated']['materially_negative']
    out={'gate':'G1E regulatory evolutionary association versus mutability','protocol':'REGULATORY_GATE_FREEZE_v1.0','input':str(a.input),'n_positions':len(df),'n_promoters':int(df.promoter_id.nunique()),'n_snps':int(df.snp_present.sum()),'n_boot':a.n_boot,'mutation_covariate':'Zhu-2014 MA RC-pooled trinucleotide log relative rate','material_beta_threshold':MATERIAL_BETA,'models':models,'cross_system_claim_passes':bool(passed),'overall':'PASS_MUTABILITY_CALIBRATED_6MER_ASSOCIATION' if passed else 'FAIL_6MER_ASSOCIATION_AFTER_MUTABILITY_CALIBRATION'}
    RES.mkdir(exist_ok=True); (RES/'gate1_mutability_calibration.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
