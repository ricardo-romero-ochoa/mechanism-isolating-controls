#!/usr/bin/env python3
"""G1B: independent Reb1 6-mer validation with an exact label-permutation refit null.

The target was frozen before this validation outcome: TTACCC and its reverse complement GGGTAA.
The validation library is the separately constructed high-quality pTpA library in GSE104878.
The model is a shift-invariant linear 6-mer count model with geometry-scaled ridge.  Because ridge is
linear in y, the target coefficients under each label permutation are computed algebraically from the
same full-refit solution operator; this is exactly equivalent to refitting the ridge model for every
permutation, not an approximation.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from regulatory_common import download_resume, parse_seq_value, ptpa_insert, seq_kmers, kmer_index, index_kmer, r2

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'regulatory'/'cache'
OUT=ROOT/'results'/'g1b_reb1_independent_null.json'
URL='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20160503_average_promoter_ELs_per_seq_atLeast100Counts.txt.gz'
TARGETS=('TTACCC','GGGTAA')
SEED=1847

def make_X(rows,k=6):
    rr=[]; cc=[]; vv=[]
    for i,(seq,_) in enumerate(rows):
        ids=seq_kmers(seq,k)
        u,c=np.unique(ids,return_counts=True)
        rr.extend([i]*len(u)); cc.extend(u.tolist()); vv.extend(c.astype(np.float64).tolist())
    return sparse.csr_matrix((vv,(rr,cc)),shape=(len(rows),4**k),dtype=np.float64)

def ridge_operator(X,lam):
    n,p=X.shape; mean=np.asarray(X.mean(axis=0)).ravel()
    def xc_mv(v): return np.asarray(X@v).ravel()-float(mean@v)
    def xct_mv(r): return np.asarray(X.T@r).ravel()-mean*float(np.sum(r))
    def mv(v):
        z=xc_mv(v); return xct_mv(z)+lam*v
    return LinearOperator((p,p),matvec=mv,rmatvec=mv,dtype=np.float64), mean, xc_mv, xct_mv

def solve_beta(X,y,ridge_scale=1e-3,rtol=1e-9,maxiter=3000):
    y=np.asarray(y,float); yc=y-y.mean(); n,p=X.shape
    mean=np.asarray(X.mean(axis=0)).ravel()
    col_ss=np.asarray(X.power(2).sum(axis=0)).ravel()-n*mean*mean
    geom=max(float(np.mean(col_ss[col_ss>0])) if np.any(col_ss>0) else 1.0,1e-12)
    lam=ridge_scale*geom
    H,mean,xc,xct=ridge_operator(X,lam)
    rhs=xct(yc)
    beta,info=cg(H,rhs,rtol=rtol,atol=0,maxiter=maxiter)
    if info!=0: raise RuntimeError(f'CG did not converge for observed ridge fit; info={info}')
    return beta,float(y.mean()),mean,lam,H,xc

def heldout(X,y,ridge_scale=1e-3):
    n=len(y); idx=np.arange(n); test=(idx%5)==0; train=~test
    Xt=X[train]; yt=np.asarray(y)[train]
    beta,b,mean,lam,H,xc=solve_beta(Xt,yt,ridge_scale)
    pred=b+np.asarray(X[test]@beta).ravel()-float(mean@beta)
    return r2(np.asarray(y)[test],pred),int(train.sum()),int(test.sum()),lam

def run(path,n_perm,ridge_scale):
    rows=[]; rejected_geometry=0
    for seq,y in parse_seq_value(path,max_len=None):
        ins=ptpa_insert(seq)
        if ins is None:
            rejected_geometry+=1
            continue
        rows.append((ins,y))
    if len(rows)<200: raise RuntimeError(f'Only {len(rows)} valid 80-bp pTpA inserts parsed from {path}; rejected_geometry={rejected_geometry}')
    X=make_X(rows); y=np.array([v for _,v in rows],float)
    h_r2,ntr,nte,_=heldout(X,y,ridge_scale)
    beta,b,mean,lam,H,xc=solve_beta(X,y,ridge_scale)
    absb=np.abs(beta); order=np.argsort(-absb); ranks=np.empty_like(order); ranks[order]=np.arange(1,len(order)+1)
    tid=[kmer_index(t) for t in TARGETS]
    obs=float(max(abs(beta[j]) for j in tid)); target_ranks={t:int(ranks[j]) for t,j in zip(TARGETS,tid)}
    influences=[]
    for j in tid:
        e=np.zeros(X.shape[1]); e[j]=1.0
        v,info=cg(H,e,rtol=1e-9,atol=0,maxiter=3000)
        if info!=0: raise RuntimeError(f'CG did not converge for target influence {index_kmer(j,6)}; info={info}')
        influences.append(xc(v))
    yc=y-y.mean(); rng=np.random.default_rng(SEED); ge=0; null=[]
    for _ in range(n_perm):
        yp=rng.permutation(yc)
        s=max(abs(float(h@yp)) for h in influences)
        null.append(s); ge += int(s>=obs-1e-15)
    p=(ge+1)/(n_perm+1); rank_best=min(target_ranks.values())
    qc={'n_sequences_ge_1000':len(rows)>=1000,'heldout_R2_ge_0p20':h_r2>=0.20}
    primary={'target_best_rank_top_1pct':rank_best<=41,'empirical_p_le_0p01':p<=0.01}
    out={
      'gate':'G1B independent Reb1 recovery', 'protocol':'REGULATORY_GATE_FREEZE_v1.0',
      'source':'GSE104878 separately constructed high-quality pTpA library', 'input':str(path),
      'n_sequences':len(rows),'n_rejected_sequence_geometry':rejected_geometry,'sequence_parser':'pTpA_insert_17plus80plus13_v1.2','n_features':4096,'feature':'shift-invariant 6-mer counts',
      'ridge_scale_geometry':ridge_scale,'ridge_lambda':lam,'heldout_R2':h_r2,'heldout_train_n':ntr,'heldout_test_n':nte,
      'targets_frozen_before_validation_outcome':list(TARGETS),
      'target_coefficients':{t:float(beta[j]) for t,j in zip(TARGETS,tid)},'target_ranks_abs_effect':target_ranks,
      'best_target_abs_effect':obs,'n_perm':n_perm,'null_mean':float(np.mean(null)),'null_sd':float(np.std(null,ddof=1)),
      'empirical_two_sided_target_p':float(p),'full_refit_equivalence':'exact linear-ridge influence identity',
      'quality_checks':qc,'primary_checks':primary,
      'verdict':'PASS_REB1_INDEPENDENT_PIPELINE_NULL' if all(qc.values()) and all(primary.values()) else 'FAIL_OR_HOLD_REB1_VALIDATION'
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); return out

def self_test():
    rng=np.random.default_rng(7); n=1800; rows=[]
    for _ in range(n):
        s=''.join(rng.choice(list('ACGT'),80)); c=sum(s[i:i+6]=='TTACCC' for i in range(75)); y=1.8*c+rng.normal(0,1); rows.append((s,y))
    X=make_X(rows); y=np.array([v for _,v in rows]); b,_,_,_,_,_=solve_beta(X,y,1e-3)
    rank=int(np.where(np.argsort(-np.abs(b))==kmer_index('TTACCC'))[0][0])+1
    assert rank<=5,rank
    from regulatory_common import PTPA_FLANK5, PTPA_FLANK3
    probe='ACGT'*20
    assert ptpa_insert(PTPA_FLANK5+probe+PTPA_FLANK3)==probe
    assert ptpa_insert(probe)==probe
    assert ptpa_insert((PTPA_FLANK5+probe+PTPA_FLANK3)[-80:]) is not None
    print('PASS G1B synthetic target-recovery + pTpA geometry self-test; rank',rank)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input'); ap.add_argument('--n-perm',type=int,default=999); ap.add_argument('--ridge-scale',type=float,default=1e-3); ap.add_argument('--offline',action='store_true'); ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
    if a.self_test: self_test(); return
    path=Path(a.input) if a.input else CACHE/'GSE104878_20160503_average_promoter_ELs_per_seq_atLeast100Counts.txt.gz'
    if not path.exists():
        if a.offline: raise SystemExit(f'missing {path} in --offline mode')
        download_resume(URL,path)
    run(path,a.n_perm,a.ridge_scale)
if __name__=='__main__': main()
