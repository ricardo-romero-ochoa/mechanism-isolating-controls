#!/usr/bin/env python3
"""Reconstruct the random-library 6-mer and 16-dinucleotide models for G1E.

This is a fresh deterministic reconstruction from the public main pTpA library, not a claim of
bit-identical recovery of the historical fit.  G1E is allowed to proceed only if the reconstruction
passes outcome-independent predictive and Reb1 sanity checks frozen in REGULATORY_GATE_FREEZE_v1.0.
"""
from __future__ import annotations
import argparse, json, math, shutil
from pathlib import Path
import numpy as np
from regulatory_common import download_resume, parse_seq_value, ptpa_insert, seq_kmers, kmer_index, index_kmer, r2

ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'regulatory'/'cache'; RES=ROOT/'results'
URL='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_20160609_average_promoter_ELs_per_seq_pTpA_ALL.shuffled.txt.gz'
DATA=CACHE/'GSE104878_20160609_average_promoter_ELs_per_seq_pTpA_ALL.shuffled.txt.gz'
CHUNKS=CACHE/'g1e_large_chunks'
SEED=2191
PARSER_VERSION='pTpA_insert_17plus80plus13_v1.2'

def dinuc_counts(seq):
    x=np.zeros(16,dtype=np.uint8)
    b={'A':0,'C':1,'G':2,'T':3}
    for a,c in zip(seq[:-1],seq[1:]): x[4*b[a]+b[c]]+=1
    return x

def build_chunks(path,chunk_size=100000,force=False):
    manifest=CHUNKS/'manifest.json'
    if manifest.exists() and not force:
        old=json.loads(manifest.read_text())
        if old.get('parser_version')==PARSER_VERSION and old.get('n_valid',0)>0:
            return old
        print('stale/invalid chunk manifest detected; rebuilding with corrected pTpA insert parser',flush=True)
        force=True
    if force and CHUNKS.exists(): shutil.rmtree(CHUNKS)
    CHUNKS.mkdir(parents=True,exist_ok=True)
    tok=[]; di=[]; ys=[]; tests=[]; files=[]; n=0; rejected=0; seen=0
    def flush():
        nonlocal tok,di,ys,tests
        if not tok:return
        fp=CHUNKS/f'chunk_{len(files):04d}.npz'
        np.savez(fp,tok6=np.asarray(tok,dtype=np.uint16),dinuc=np.asarray(di,dtype=np.uint8),y=np.asarray(ys,dtype=np.float32),is_test=np.asarray(tests,dtype=bool))
        files.append(fp.name); tok=[];di=[];ys=[];tests=[]
    for seq,y in parse_seq_value(path,max_len=None):
        seen+=1
        ins=ptpa_insert(seq)
        if ins is None:
            rejected+=1
            if seen%500000==0: print(f'scanned {seen:,} rows; accepted {n:,}; rejected geometry {rejected:,}',flush=True)
            continue
        ids=seq_kmers(ins,6)
        if len(ids)!=75:
            rejected+=1
            continue
        tok.append(ids); di.append(dinuc_counts(ins)); ys.append(y); tests.append((n%10)==0); n+=1
        if len(tok)>=chunk_size:
            flush(); print(f'cached {n:,} valid inserts (scanned {seen:,})',flush=True)
        elif seen%500000==0:
            print(f'scanned {seen:,} rows; accepted {n:,}; rejected geometry {rejected:,}',flush=True)
    flush()
    m={'input':str(path),'parser_version':PARSER_VERSION,'n_seen_parseable':seen,'n_valid':n,'n_rejected':rejected,'chunks':files,'test_rule':'valid-row index modulo 10 == 0'}
    manifest.write_text(json.dumps(m,indent=2)+'\n'); return m

def train_stats(manifest):
    ntr=0; sy=0.; sy2=0.
    for fn in manifest['chunks']:
        z=np.load(CHUNKS/fn); m=~z['is_test']; y=z['y'][m].astype(float); ntr+=len(y); sy+=y.sum(); sy2+=y@y
    mean=sy/ntr; var=max(sy2/ntr-mean*mean,1e-12); return ntr,mean,math.sqrt(var)

def fit_6mer(manifest,device='auto',epochs=16,batch=16384,lr=0.02,weight_decay=1e-6):
    import torch
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    ntr,ym,ysd=train_stats(manifest)
    w=torch.zeros(4096,device=device,dtype=torch.float32,requires_grad=True); b=torch.zeros((),device=device,dtype=torch.float32,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=lr,weight_decay=weight_decay)
    logs=[]
    for ep in range(epochs):
        total=0.; count=0
        order=list(range(len(manifest['chunks']))); shift=ep%max(1,len(order)); order=order[shift:]+order[:shift]
        for ci in order:
            z=np.load(CHUNKS/manifest['chunks'][ci]); m=np.flatnonzero(~z['is_test']); t=z['tok6']; yy=z['y']
            for s in range(0,len(m),batch):
                ix=m[s:s+batch]
                tt=torch.as_tensor(t[ix].astype(np.int64),device=device); y=torch.as_tensor(((yy[ix]-ym)/ysd).astype(np.float32),device=device)
                pred=w[tt].sum(1)+b; loss=torch.mean((pred-y)**2)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=float(loss.detach())*len(ix); count+=len(ix)
        logs.append({'epoch':ep+1,'train_mse_standardized':total/count}); print('epoch',ep+1,'mse',total/count,flush=True)
    wr=w.detach().cpu().numpy().astype(np.float64)*ysd; br=float(b.detach().cpu())*ysd+ym
    return wr,br,logs,device

def evaluate_6mer(manifest,w,b):
    yy=[]; pp=[]
    for fn in manifest['chunks']:
        z=np.load(CHUNKS/fn); m=z['is_test']; t=z['tok6'][m].astype(np.int64); y=z['y'][m].astype(float)
        if len(y): yy.append(y); pp.append(w[t].sum(1)+b)
    y=np.concatenate(yy); p=np.concatenate(pp); return r2(y,p),len(y)

def fit_dinuc(manifest,ridge=1e-6):
    sx=np.zeros(16); sxx=np.zeros((16,16)); sxy=np.zeros(16); sy=0.; n=0
    for fn in manifest['chunks']:
        z=np.load(CHUNKS/fn); m=~z['is_test']; X=z['dinuc'][m].astype(float); y=z['y'][m].astype(float)
        sx+=X.sum(0); sxx+=X.T@X; sxy+=X.T@y; sy+=y.sum(); n+=len(y)
    mx=sx/n; my=sy/n; H=sxx-n*np.outer(mx,mx)+ridge*np.eye(16); rhs=sxy-n*mx*my
    w=np.linalg.solve(H,rhs); b=my-mx@w
    yy=[];pp=[]
    for fn in manifest['chunks']:
        z=np.load(CHUNKS/fn); m=z['is_test']; X=z['dinuc'][m].astype(float); y=z['y'][m].astype(float); yy.append(y);pp.append(X@w+b)
    return w,b,r2(np.concatenate(yy),np.concatenate(pp))

def rank_targets(w):
    order=np.argsort(-np.abs(w)); ranks=np.empty_like(order); ranks[order]=np.arange(1,len(w)+1)
    return {t:int(ranks[kmer_index(t)]) for t in ('TTACCC','GGGTAA')}

def self_test():
    assert len(seq_kmers('A'*80,6))==75 and dinuc_counts('A'*80)[0]==79
    from regulatory_common import PTPA_FLANK5, PTPA_FLANK3
    probe='ACGT'*20
    assert ptpa_insert(PTPA_FLANK5+probe+PTPA_FLANK3)==probe
    assert ptpa_insert(probe)==probe
    assert ptpa_insert('A'*110) is None
    print('PASS large-library model utility + pTpA geometry self-test')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input'); ap.add_argument('--offline',action='store_true'); ap.add_argument('--rebuild-cache',action='store_true'); ap.add_argument('--device',default='auto'); ap.add_argument('--epochs',type=int,default=16); ap.add_argument('--batch',type=int,default=16384); ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
    if a.self_test:self_test();return
    path=Path(a.input) if a.input else DATA
    if not path.exists():
        if a.offline: raise SystemExit(f'missing {path}')
        download_resume(URL,path)
    man=build_chunks(path,force=a.rebuild_cache)
    if man['n_valid']<500000: raise RuntimeError(f'Only {man["n_valid"]} valid 80-bp rows; expected a large random library')
    w6,b6,logs,dev=fit_6mer(man,a.device,a.epochs,a.batch); r6,nt=evaluate_6mer(man,w6,b6)
    wd,bd,rd=fit_dinuc(man); ranks=rank_targets(w6)
    qc={'n_valid_ge_500k':man['n_valid']>=500000,'sixmer_R2_ge_0p48':r6>=0.48,'dinucleotide_R2_ge_0p35':rd>=0.35,'reb1_best_rank_top_1pct':min(ranks.values())<=41}
    RES.mkdir(exist_ok=True)
    np.savez(RES/'g1e_large_library_models.npz',sixmer_weights=w6,sixmer_intercept=b6,dinucleotide_weights=wd,dinucleotide_intercept=bd)
    out={'gate':'G1E large-library reconstruction precondition','protocol':'REGULATORY_GATE_FREEZE_v1.0','input':str(path),'n_valid':man['n_valid'],'n_test':nt,'sixmer_R2':r6,'dinucleotide_R2':rd,'reb1_target_ranks':ranks,'device':dev,'epochs_fixed':a.epochs,'training_log':logs,'quality_checks':qc,'verdict':'PASS_MODEL_RECONSTRUCTION' if all(qc.values()) else 'HOLD_MODEL_RECONSTRUCTION_MISMATCH'}
    (RES/'g1e_large_library_model_status.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
