#!/usr/bin/env python3
"""Build an independent trinucleotide mutation-opportunity covariate from Zhu et al. 2014 MA SNMs.

The outcome under G1E is promoter polymorphism.  This model is derived only from independent mutation-
accumulation mutations and sacCer3 sequence opportunity.  Reverse-complement trinucleotides are pooled
before estimating relative rates.  A Jeffreys 0.5 pseudocount is fixed before G1E outcomes.
"""
from __future__ import annotations
import argparse, csv, json, re, tarfile
from pathlib import Path
from collections import Counter
import numpy as np
from regulatory_common import download_resume, open_text, revcomp, parse_fasta, roman

ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'regulatory'/'cache'; RES=ROOT/'results'
SUPP=CACHE/'1323011111_pnas.1323011111.sd01.txt'
FA_TAR=CACHE/'sacCer3.chromFa.tar.gz'; FA_DIR=CACHE/'sacCer3_chromFa'
SUPP_URLS=[
 'https://pmc-oa-opendata.s3.amazonaws.com/PMC4050626.1/1323011111_pnas.1323011111.sd01.txt',
 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4050626/bin/1323011111_pnas.1323011111.sd01.txt',
 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4050626/bin/1323011111_pnas.1323011111.sd01.txt',
 'https://www.pnas.org/doi/suppl/10.1073/pnas.1323011111/suppl_file/pnas.1323011111.sd01.txt',
]
FA_URL='https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/chromFa.tar.gz'
BASE=set('ACGT')

def _file_probe(path):
    p=Path(path)
    if not p.exists(): return {'exists':False,'size':0,'looks_html':False}
    head=p.read_bytes()[:4096].lstrip().lower()
    return {'exists':True,'size':p.stat().st_size,
            'looks_html':head.startswith(b'<!doctype html') or head.startswith(b'<html') or b'<title>' in head[:1000]}

def _download_candidate(url, dest):
    tmp=Path(str(dest)+'.candidate')
    for q in (tmp, tmp.with_suffix(tmp.suffix+'.part')):
        try:q.unlink()
        except FileNotFoundError:pass
    download_resume(url,tmp,retries=4)
    return tmp

def get_supp(path,offline=False):
    path=Path(path)
    if path.exists():
        rec=parse_snm_records(path)
        if 800 <= len(rec) <= 900:
            return path
        if offline:
            pr=_file_probe(path)
            raise SystemExit(f'cached Zhu supplement is unusable in --offline mode: parsed={len(rec)} probe={pr}; rerun without --offline')
    if offline: raise SystemExit(f'missing {path}')
    errors=[]; best=None; best_n=-1
    for u in SUPP_URLS:
        try:
            cand=_download_candidate(u,path)
            pr=_file_probe(cand)
            n=len(parse_snm_records(cand)) if not pr['looks_html'] else 0
            if n>best_n:
                best,best_n=cand,n
            if 800 <= n <= 900:
                path.parent.mkdir(parents=True,exist_ok=True)
                cand.replace(path)
                return path
            errors.append(f'{u}: parsed {n} SNMs, probe={pr}')
        except Exception as e:
            errors.append(f'{u}: {type(e).__name__}: {e}')
    if best is not None and best.exists():
        diag=Path(str(path)+'.best_failed_candidate')
        try: best.replace(diag)
        except Exception: pass
    raise RuntimeError('Could not obtain a Zhu Dataset S1 payload that parses to 800-900 SNMs.\n'+'\n'.join(errors))

def get_genome(tar_path,offline=False):
    if not FA_TAR.exists():
        if offline: raise SystemExit(f'missing {FA_TAR}')
        download_resume(FA_URL,FA_TAR)
    if not FA_DIR.exists():
        FA_DIR.mkdir(parents=True)
        with tarfile.open(FA_TAR,'r:gz') as t: t.extractall(FA_DIR)
    seq={}
    for fp in FA_DIR.rglob('chr*.fa'):
        seq.update(parse_fasta(fp))
    if len([k for k in seq if k.startswith('chr')])<16: raise RuntimeError('failed to extract 16 nuclear sacCer3 chromosomes')
    return seq

def norm_chrom(x):
    s=str(x).strip().upper().replace('CHR','').replace('*','')
    romans={'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10,'XI':11,'XII':12,'XIII':13,'XIV':14,'XV':15,'XVI':16}
    if s in romans: n=romans[s]
    else:
        m=re.search(r'\b(1[0-6]|[1-9])\b',s)
        if not m:return None
        n=int(m.group(1))
    return 'chr'+roman(n)

def parse_snm_records(path):
    lines=Path(path).read_text(encoding='utf-8',errors='replace').splitlines()
    rec=[]; header=None; active=False
    for line in lines:
        raw=line.strip()
        if not raw: continue
        parts=[p.strip() for p in re.split(r'\t+',raw)]
        if len(parts)>=4 and any('position' in p.lower() for p in parts) and any(('chr' in p.lower() or 'chrom' in p.lower()) for p in parts):
            hp=[p.lower() for p in parts]
            def idx(keys):
                for i,p in enumerate(hp):
                    if any(k in p for k in keys): return i
                return None
            cand={'line':idx(['line','strain']),'chrom':idx(['chrm','chrom']),'pos':idx(['position','pos']),'ref':idx(['ref']),'alt':idx(['alt','mutant','derived'])}
            if cand['chrom'] is not None and cand['pos'] is not None and cand['ref'] is not None and cand['alt'] is not None:
                header=cand; active=True; continue
        if active and header is not None:
            try:
                mx=max(i for i in header.values() if i is not None)
                if len(parts)<=mx: continue
                chrom=norm_chrom(parts[header['chrom']]); pos=int(re.sub('[^0-9]','',parts[header['pos']]))
                ref=parts[header['ref']].upper().strip('` '); alt=parts[header['alt']].upper().strip('` ')
                lineid=parts[header['line']] if header['line'] is not None else ''
                looks_ma=(re.search(r'MA\s*\d+',lineid,re.I) is not None) or (lineid.isdigit() and 1<=int(lineid)<=200)
                if chrom and len(ref)==1 and len(alt)==1 and ref in BASE and alt in BASE and ref!=alt and looks_ma:
                    rec.append((chrom,pos,ref,alt,lineid)); continue
            except Exception: pass
    if not (800<=len(rec)<=900):
        rec2=[]
        for raw in lines:
            if not re.search(r'\bMA\s*\d+\b',raw,re.I): continue
            p=re.split(r'\s+|\t+',raw.strip())
            ma=next((x for x in p if re.fullmatch(r'MA\d+',x,re.I)),None)
            bases=[x.upper().strip('` ') for x in p if len(x.strip('` '))==1 and x.upper().strip('` ') in BASE]
            nums=[int(x) for x in p if x.isdigit()]
            chrom=None
            for x in p:
                c=norm_chrom(x)
                if c: chrom=c; break
            if chrom and len(nums)>=2 and len(bases)>=2:
                pos=max(nums); ref,alt=bases[-2],bases[-1]
                if ref!=alt: rec2.append((chrom,pos,ref,alt,ma))
        if len(rec2)>len(rec): rec=rec2
    seen=set(); out=[]
    for r in rec:
        key=r[:4]+(str(r[4]),)
        if key not in seen: seen.add(key); out.append(r)
    return out

def canon3(s):
    rc=revcomp(s); return min(s,rc)

def build(records,genome):
    opp=Counter()
    for chrom,seq in genome.items():
        if chrom=='chrM' or not chrom.startswith('chr'): continue
        for i in range(1,len(seq)-1):
            tri=seq[i-1:i+2]
            if set(tri)<=BASE: opp[canon3(tri)]+=1
    muts=Counter(); ref_match=0; usable=0
    for chrom,pos,ref,alt,line in records:
        seq=genome.get(chrom)
        if not seq or pos<2 or pos>=len(seq): continue
        i=pos-1; tri=seq[i-1:i+2]
        if set(tri)<=BASE:
            usable+=1; ref_match += int(seq[i]==ref); muts[canon3(tri)]+=1
    contexts=sorted(opp); raw={c:(muts[c]+0.5)/(opp[c]+0.5) for c in contexts}
    logs=np.array([np.log(raw[c]) for c in contexts]); gm=float(np.exp(logs.mean()))
    rows=[]
    for c in contexts:
        rows.append({'canonical_context':c,'mutation_count':int(muts[c]),'opportunity_count':int(opp[c]),'relative_rate':float(raw[c]/gm),'log_relative_rate':float(np.log(raw[c]/gm))})
    return rows,usable,ref_match

def self_test():
    assert canon3('ACG')==canon3(revcomp('ACG'))
    assert norm_chrom('IV')=='chrIV' and norm_chrom('4')=='chrIV'
    print('PASS Zhu-2014 mutation-model utility self-test')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mutations'); ap.add_argument('--offline',action='store_true'); ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
    if a.self_test:self_test();return
    p=Path(a.mutations) if a.mutations else SUPP; p=get_supp(p,a.offline); genome=get_genome(FA_TAR,a.offline)
    rec=parse_snm_records(p); rows,usable,match=build(rec,genome)
    qc={'parsed_SNM_800_to_900':800<=len(rec)<=900,'usable_context_SNM_800_to_900':800<=usable<=900,'contexts_with_mutations_ge_25':sum(r['mutation_count']>0 for r in rows)>=25,'reference_match_fraction_ge_0p95':(match/max(1,usable))>=0.95}
    RES.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(RES/'zhu2014_trinuc_mutation_rates.csv',index=False)
    out={'gate':'G1E independent mutability model','source':'Zhu et al. 2014 PNAS mutation-accumulation SNMs','input':str(p),'input_probe':_file_probe(p),'parsed_SNM':len(rec),'usable_context_SNM':usable,'reference_match_fraction':match/max(1,usable),'pseudocount':0.5,'reverse_complement_pooled':True,'n_context_classes':len(rows),'quality_checks':qc,'verdict':'PASS_MUTABILITY_MODEL' if all(qc.values()) else 'HOLD_MUTATION_DATA_PARSE_OR_QC'}
    (RES/'zhu2014_mutation_model_status.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
