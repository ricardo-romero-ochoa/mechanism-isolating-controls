#!/usr/bin/env python3
"""Rebuild the public sacCer3 promoter-position table required for G1E.

Frozen design: chrI/III/VI/IX; 300 bp upstream of SGD transcript starts, strand oriented; exclude any
position overlapping an SGD exon or lacking sacCer3 7-way phastCons; 12 x 25-bp distance bins; 21-bp
local GC; EVA SNP release 8 substitutions; sensitivities are mean absolute latent-score changes over
three alternate bases.  Mutation opportunity is the independent Zhu-2014 RC-pooled trinucleotide rate.
"""
from __future__ import annotations
import argparse, gzip, json, math, re, tarfile
from pathlib import Path
import numpy as np, pandas as pd
from regulatory_common import download_resume, parse_fasta, revcomp, fetch_json, kmer_index

ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'regulatory'/'cache'; RES=ROOT/'results'; OUT=ROOT/'regulatory'/'gate1_positions.csv'
CHROMS=['chrI','chrIII','chrVI','chrIX']; BASES='ACGT'; B2I={b:i for i,b in enumerate(BASES)}
SGD=CACHE/'sgdGene.txt.gz'; SGD_URL='https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/database/sgdGene.txt.gz'
FA_TAR=CACHE/'sacCer3.chromFa.tar.gz'; FA_URL='https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/chromFa.tar.gz'; FA_DIR=CACHE/'sacCer3_chromFa'
PHAST_URL='https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/phastCons7way/sacCer3.{chrom}.wigFixed.gz'

def ensure_public(offline=False):
    if not SGD.exists():
        if offline: raise SystemExit(f'missing {SGD}')
        download_resume(SGD_URL,SGD)
    if not FA_TAR.exists():
        if offline: raise SystemExit(f'missing {FA_TAR}')
        download_resume(FA_URL,FA_TAR)
    if not FA_DIR.exists():
        FA_DIR.mkdir(parents=True)
        with tarfile.open(FA_TAR,'r:gz') as t: t.extractall(FA_DIR)
    for c in CHROMS:
        p=CACHE/f'sacCer3.{c}.wigFixed.gz'
        if not p.exists():
            if offline: raise SystemExit(f'missing {p}')
            download_resume(PHAST_URL.format(chrom=c),p)

def genome_sequences():
    seq={}
    for fp in FA_DIR.rglob('chr*.fa'): seq.update(parse_fasta(fp))
    return seq

def _parse_sgd_row(fields):
    if len(fields) >= 11 and fields[2] in CHROMS and fields[3] in ('+','-'):
        o=1
    elif len(fields) >= 10 and fields[1] in CHROMS and fields[2] in ('+','-'):
        o=0
    else:
        return None
    try:
        return {
            'name': fields[o+0],
            'chrom': fields[o+1],
            'strand': fields[o+2],
            'txStart': int(fields[o+3]),
            'txEnd': int(fields[o+4]),
            'exonStarts': [int(x) for x in fields[o+8].strip(',').split(',') if x],
            'exonEnds': [int(x) for x in fields[o+9].strip(',').split(',') if x],
        }
    except (ValueError, IndexError):
        return None

def parse_genes():
    genes=[]
    n_total=0; n_schema=0
    with gzip.open(SGD,'rt') as f:
        for line in f:
            n_total += 1
            g=_parse_sgd_row(line.rstrip().split('\t'))
            if g is None:
                continue
            n_schema += 1
            genes.append(g)
    seen=set(); out=[]
    for g in genes:
        key=(g['name'],g['chrom'],g['strand'],g['txStart'],g['txEnd'])
        if key not in seen:
            seen.add(key); out.append(g)
    if not out:
        raise RuntimeError(
            f'Parsed zero SGD genes on frozen chromosomes from {SGD}; '
            f'total rows={n_total}, schema-matched rows={n_schema}. '
            'Expected UCSC sgdGene GenePred with optional leading bin field.'
        )
    print(f'parsed SGD genes on frozen chromosomes: {len(out):,} (schema rows {n_schema:,}/{n_total:,})', flush=True)
    return out

def exon_masks(genes,genome):
    m={c:np.zeros(len(genome[c]),dtype=bool) for c in CHROMS}
    for g in genes:
        for a,b in zip(g['exonStarts'],g['exonEnds']):
            m[g['chrom']][max(0,a):min(len(m[g['chrom']]),b)]=True
    return m

def phast_array(chrom,n):
    a=np.full(n,np.nan,dtype=np.float32); pos=None; step=1; span=1
    with gzip.open(CACHE/f'sacCer3.{chrom}.wigFixed.gz','rt') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith('fixedStep'):
                kv=dict(re.findall(r'(\w+)=([^\s]+)',line)); pos=int(kv.get('start','1'))-1; step=int(kv.get('step','1')); span=int(kv.get('span','1')); continue
            if line.startswith(('track','variableStep','#')): continue
            if pos is None: continue
            try:v=float(line)
            except ValueError:continue
            a[pos:min(n,pos+span)]=v; pos+=step
    return a

def extract_track_list(obj):
    for k,v in obj.items():
        if isinstance(v,list) and (not v or isinstance(v[0],dict)):
            if v and ('chromStart' in v[0] or 'start' in v[0]): return v
    return []

def eva_positions(chrom,offline=False):
    p=CACHE/f'evaSnp8_{chrom}.json'
    if p.exists(): obj=json.loads(p.read_text())
    else:
        if offline: raise SystemExit(f'missing {p}')
        url=f'https://api.genome.ucsc.edu/getData/track?genome=sacCer3;track=evaSnp8;chrom={chrom}'
        obj=fetch_json(url,p)
    rows=extract_track_list(obj); pos=set(); nrecords=0
    for r in rows:
        vc=str(r.get('varClass','')).lower().replace(' ','_')
        ref=str(r.get('ref','')).upper(); alt=str(r.get('alt','')).upper()
        try:s=int(r.get('chromStart'));e=int(r.get('chromEnd'))
        except Exception:continue
        if vc not in ('substitution','snv') or e-s!=1 or len(ref)!=1 or ref not in BASES:continue
        alts=re.split('[,/]',alt)
        if not any(len(x)==1 and x in BASES and x!=ref for x in alts):continue
        pos.add(s); nrecords+=1
    return pos,nrecords

def canon3(s):
    rc=revcomp(s); return min(s,rc)

def load_mut_rates():
    p=RES/'zhu2014_trinuc_mutation_rates.csv'
    if not p.exists(): raise RuntimeError('missing results/zhu2014_trinuc_mutation_rates.csv; run zhu2014_mutation_model.py first')
    d=pd.read_csv(p); return {r.canonical_context:float(r.log_relative_rate) for _,r in d.iterrows()}

def load_models():
    p=RES/'g1e_large_library_models.npz'; st=RES/'g1e_large_library_model_status.json'
    if not p.exists() or not st.exists(): raise RuntimeError('run fit_large_library_models.py first')
    js=json.loads(st.read_text())
    if js.get('verdict')!='PASS_MODEL_RECONSTRUCTION': raise RuntimeError('large-library reconstruction did not pass; G1E remains HOLD')
    z=np.load(p); mats=np.load(RES/'phase3_factor_matrices.npy'); fac=pd.read_csv(RES/'phase3_factors.csv')
    return z['sixmer_weights'].astype(float),z['dinucleotide_weights'].astype(float),mats.astype(float),fac['weight_a'].to_numpy(float),fac['mu'].to_numpy(float)

def six_sensitivity(seq,w):
    L=len(seq); old=[kmer_index(seq[p:p+6]) for p in range(L-5)]; out=np.zeros(L)
    for i,ref in enumerate(seq):
        vals=[]
        for alt in BASES:
            if alt==ref:continue
            d=0.
            for p in range(max(0,i-5),min(i,L-6)+1):
                s=list(seq[p:p+6]); s[i-p]=alt; d+=w[kmer_index(''.join(s))]-w[old[p]]
            vals.append(abs(d))
        out[i]=np.mean(vals)
    return out

def dinuc_sensitivity(seq,w):
    L=len(seq); out=np.zeros(L)
    for i,ref in enumerate(seq):
        vals=[]
        for alt in BASES:
            if alt==ref:continue
            d=0.
            if i>0:d+=w[4*B2I[seq[i-1]]+B2I[alt]]-w[4*B2I[seq[i-1]]+B2I[ref]]
            if i<L-1:d+=w[4*B2I[alt]+B2I[seq[i+1]]]-w[4*B2I[ref]+B2I[seq[i+1]]]
            vals.append(abs(d))
        out[i]=np.mean(vals)
    return out

def occupancy_sensitivity(seq,mats,a,mu):
    L=len(seq); F,W,_=mats.shape; enc=np.array([B2I[c] for c in seq],int)
    expu=[]; Z=[]; theta=[]
    for f in range(F):
        u=np.empty(L-W+1)
        for p in range(L-W+1): u[p]=mu[f]+sum(mats[f,j,enc[p+j]] for j in range(W))
        e=np.exp(np.clip(u,-50,50)); expu.append(e); z=float(e.sum()); Z.append(z); theta.append(z/(1+z))
    phi0=float(np.dot(a,theta)); out=np.zeros(L)
    for i,ref in enumerate(seq):
        vals=[]
        for alt in BASES:
            if alt==ref:continue
            ai=B2I[alt]; ri=B2I[ref]; th=[]
            for f in range(F):
                e=expu[f]; z=Z[f]; delta=0.
                for p in range(max(0,i-W+1),min(i,L-W)+1):
                    j=i-p; old=e[p]; new=old*math.exp(float(mats[f,j,ai]-mats[f,j,ri])); delta+=new-old
                zn=max(z+delta,1e-300); th.append(zn/(1+zn))
            vals.append(abs(float(np.dot(a,th))-phi0))
        out[i]=np.mean(vals)
    return out

def local_gc(seq,i,half=10):
    s=seq[max(0,i-half):min(len(seq),i+half+1)]; return (s.count('G')+s.count('C'))/len(s)

def count_full_promoter_windows(genes, genome):
    nwin=0
    for g in genes:
        c=g['chrom']; strand=g['strand']; n=len(genome[c])
        if strand=='+': start,end=g['txStart']-300,g['txStart']
        else: start,end=g['txEnd'],g['txEnd']+300
        if start<1 or end>n-1 or end-start!=300:
            continue
        raw=genome[c][start:end]
        if any(x not in BASES for x in raw):
            continue
        nwin += 1
    return nwin


def recheck_existing(offline=False):
    if not OUT.exists():
        raise RuntimeError(f'missing {OUT}; run the full position reconstruction first')
    ensure_public(offline)
    genome=genome_sequences(); genes=parse_genes()
    nwin=count_full_promoter_windows(genes,genome)
    df=pd.read_csv(OUT)
    npos=len(df); nsnp=int(df['snp_present'].sum()) if len(df) else 0
    nret=int(df['promoter_id'].nunique()) if len(df) else 0
    eva_n=0
    for c in CHROMS:
        pos,_=eva_positions(c,offline); eva_n += len(pos)
    qc={
        'promoters_within_2pct_of_688':abs(nwin-688)<=14,
        'positions_within_3pct_of_167426':abs(npos-167426)<=5023,
        'eva_four_chrom_SNV_within_5pct_of_9839':abs(eva_n-9839)<=492,
        'promoter_SNP_within_10pct_of_1542':abs(nsnp-1542)<=154,
        'all_12_distance_bins':df.distance_bin.nunique()==12 if len(df) else False,
    }
    out={
        'gate':'G1E public-position reconstruction',
        'protocol':'REGULATORY_GATE_FREEZE_v1.0',
        'parser_version':'sgdGene_optional_bin_v1.4_promoter_count_fix',
        'bookkeeping_correction':'post-outcome technical only: compare historical 688 to full 300-bp promoter windows before per-position exclusions; v1.3 incorrectly used unique promoter IDs remaining in the filtered position table',
        'n_sgd_genes_four_chrom':len(genes),
        'n_promoters':nwin,
        'n_promoter_windows':nwin,
        'n_promoters_with_retained_positions':nret,
        'n_positions':npos,
        'n_snps':nsnp,
        'eva_unique_snv_positions_four_chrom':eva_n,
        'quality_checks':qc,
        'verdict':'PASS_POSITION_RECONSTRUCTION' if all(qc.values()) else 'HOLD_POSITION_RECONSTRUCTION_MISMATCH',
        'output':str(OUT),
    }
    (RES/'g1e_position_reconstruction_status.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    return out


def self_test():
    rng=np.random.default_rng(4); seq=''.join(rng.choice(list(BASES),40)); w=rng.normal(size=4096); d=rng.normal(size=16)
    s=six_sensitivity(seq,w); sd=dinuc_sensitivity(seq,d); assert np.all(np.isfinite(s)) and np.all(s>=0) and np.all(sd>=0)
    row='585\tYAL069W\tchrI\t+\t334\t649\t334\t649\t1\t334,\t649,\tn/a'.split('\t')
    g=_parse_sgd_row(row)
    assert g is not None and g['name']=='YAL069W' and g['chrom']=='chrI' and g['strand']=='+'
    assert g['txStart']==334 and g['txEnd']==649 and g['exonStarts']==[334] and g['exonEnds']==[649]
    g2=_parse_sgd_row(row[1:])
    assert g2 == g
    print('PASS G1E position/sensitivity + SGD GenePred parser self-test')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--offline',action='store_true'); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--recheck-existing',action='store_true'); a=ap.parse_args()
    if a.self_test:self_test();return
    if a.recheck_existing: recheck_existing(a.offline); return
    ensure_public(a.offline); genome=genome_sequences(); genes=parse_genes(); exmask=exon_masks(genes,genome); ph={c:phast_array(c,len(genome[c])) for c in CHROMS}; mut=load_mut_rates(); w6,wd,mats,aa,mu=load_models()
    eva={}; eva_n=0
    for c in CHROMS:
        eva[c],nr=eva_positions(c,a.offline); eva_n+=len(eva[c]); print(c,'EVA unique SNV positions',len(eva[c]),flush=True)
    rows=[]; promoters=0
    for gi,g in enumerate(genes):
        c=g['chrom']; strand=g['strand']; n=len(genome[c])
        if strand=='+': start,end=g['txStart']-300,g['txStart']
        else:start,end=g['txEnd'],g['txEnd']+300
        if start<1 or end>n-1 or end-start!=300:continue
        raw=genome[c][start:end]
        if any(x not in BASES for x in raw):continue
        seq=raw if strand=='+' else revcomp(raw); promoters+=1
        s6=six_sensitivity(seq,w6); sd=dinuc_sensitivity(seq,wd); so=occupancy_sensitivity(seq,mats,aa,mu)
        for j in range(300):
            gp=start+j if strand=='+' else end-1-j
            if exmask[c][gp] or not np.isfinite(ph[c][gp]):continue
            tri=genome[c][gp-1:gp+2]
            if len(tri)!=3 or any(x not in BASES for x in tri):continue
            key=canon3(tri)
            if key not in mut:continue
            dist=300-j
            rows.append((g['name'],c,gp,strand,j,int(gp in eva[c]),local_gc(seq,j),seq[j],int((dist-1)//25),mut[key],s6[j],so[j],sd[j],float(ph[c][gp])))
        if (gi+1)%100==0:print('processed genes',gi+1,'rows',len(rows),flush=True)
    cols=['promoter_id','chrom','chrom_pos0','strand','oriented_pos0','snp_present','local_gc','reference_base','distance_bin','log_mut_rate','sensitivity_6mer','sensitivity_occupancy','sensitivity_dinucleotide','phastCons7way']
    df=pd.DataFrame(rows,columns=cols); OUT.parent.mkdir(exist_ok=True); df.to_csv(OUT,index=False)
    npos=len(df); nsnp=int(df.snp_present.sum()); nprom_retained=int(df.promoter_id.nunique())
    qc={'promoters_within_2pct_of_688':abs(promoters-688)<=14,'positions_within_3pct_of_167426':abs(npos-167426)<=5023,'eva_four_chrom_SNV_within_5pct_of_9839':abs(eva_n-9839)<=492,'promoter_SNP_within_10pct_of_1542':abs(nsnp-1542)<=154,'all_12_distance_bins':df.distance_bin.nunique()==12}
    out={'gate':'G1E public-position reconstruction','protocol':'REGULATORY_GATE_FREEZE_v1.0','parser_version':'sgdGene_optional_bin_v1.4_promoter_count_fix','bookkeeping_correction':'post-outcome technical only: compare historical 688 to full 300-bp promoter windows before per-position exclusions; v1.3 incorrectly used unique promoter IDs remaining in the filtered position table','n_sgd_genes_four_chrom':len(genes),'n_promoters':promoters,'n_promoter_windows':promoters,'n_promoters_with_retained_positions':nprom_retained,'n_positions':npos,'n_snps':nsnp,'eva_unique_snv_positions_four_chrom':eva_n,'quality_checks':qc,'verdict':'PASS_POSITION_RECONSTRUCTION' if all(qc.values()) else 'HOLD_POSITION_RECONSTRUCTION_MISMATCH','output':str(OUT)}
    (RES/'g1e_position_reconstruction_status.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
