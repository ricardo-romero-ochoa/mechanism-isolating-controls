from __future__ import annotations
import gzip, hashlib, io, json, math, os, re, tarfile, time, urllib.error, urllib.request
from pathlib import Path
import numpy as np

BASES='ACGT'; B2I={b:i for i,b in enumerate(BASES)}
COMP=str.maketrans('ACGT','TGCA')

# pTpA processed-file sequence convention.  The public GEO processed files store
# a 110-nt sequence containing 17 nt of the distal scaffold, the random 80-nt
# insert, and 13 nt of the proximal scaffold.  Some derived/local files may
# already contain the 80-nt insert only.
PTPA_FLANK5='TGCATTTTTTTCACATC'
PTPA_FLANK3='GGTTACGGCTGTT'

def ptpa_insert(seq:str)->str|None:
    """Return the 80-nt random pTpA insert, or None if sequence geometry is invalid."""
    seq=seq.strip().upper()
    if len(seq)==80 and all(c in B2I for c in seq):
        return seq
    if seq.startswith(PTPA_FLANK5) and seq.endswith(PTPA_FLANK3):
        ins=seq[len(PTPA_FLANK5):len(seq)-len(PTPA_FLANK3)]
        if len(ins)==80 and all(c in B2I for c in ins):
            return ins
    return None

def revcomp(s:str)->str: return s.translate(COMP)[::-1]

def kmer_index(s:str)->int:
    x=0
    for c in s:
        x=(x<<2)|B2I[c]
    return x

def index_kmer(x:int,k:int)->str:
    out=['A']*k
    for i in range(k-1,-1,-1):
        out[i]=BASES[x&3]; x >>= 2
    return ''.join(out)

def seq_kmers(seq:str,k:int)->np.ndarray:
    seq=seq.upper()
    if len(seq)<k or any(c not in B2I for c in seq):
        return np.empty(0,dtype=np.uint16 if k<=8 else np.uint32)
    ids=np.empty(len(seq)-k+1,dtype=np.uint16 if k<=8 else np.uint32)
    mask=(1<<(2*k))-1; x=0
    for i,c in enumerate(seq):
        x=((x<<2)|B2I[c])&mask
        if i>=k-1: ids[i-k+1]=x
    return ids

def open_text(path:Path):
    return gzip.open(path,'rt',encoding='utf-8',errors='replace') if str(path).endswith('.gz') else open(path,'rt',encoding='utf-8',errors='replace')

def download_resume(url:str,dest:Path,retries:int=12,timeout:int=900):
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size>0: return dest
    part=dest.with_suffix(dest.suffix+'.part')
    for attempt in range(retries):
        start=part.stat().st_size if part.exists() else 0
        headers={'User-Agent':'mechanism-controls/1.0'}
        if start: headers['Range']=f'bytes={start}-'
        req=urllib.request.Request(url,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=timeout) as r:
                status=getattr(r,'status',200)
                mode='ab' if start and status==206 else 'wb'
                if mode=='wb' and start: start=0
                with open(part,mode) as f:
                    while True:
                        b=r.read(1<<20)
                        if not b: break
                        f.write(b)
            os.replace(part,dest); return dest
        except Exception as e:
            if attempt+1==retries: raise
            time.sleep(min(60,2**min(attempt,5)))
    return dest

def parse_seq_value(path:Path,max_len:int|None=None):
    """Yield (sequence,value) from de Boer processed files: sequence col 1, value col 2."""
    with open_text(path) as f:
        for ln,line in enumerate(f,1):
            if not line or line.startswith('#'): continue
            p=line.rstrip('\n\r').split('\t')
            if len(p)<2: continue
            seq=p[0].strip().upper()
            try: y=float(p[1])
            except ValueError: continue
            if max_len is not None and len(seq)>max_len: seq=seq[-max_len:]
            if len(seq)<6 or any(c not in B2I for c in seq) or not np.isfinite(y): continue
            yield seq,y

def r2(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    den=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-p)**2)/den) if den>0 else float('nan')

def sha256_file(path:Path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def fetch_json(url:str,cache:Path|None=None):
    if cache and cache.exists(): return json.loads(cache.read_text())
    req=urllib.request.Request(url,headers={'User-Agent':'mechanism-controls/1.0'})
    with urllib.request.urlopen(req,timeout=900) as r: obj=json.load(r)
    if cache:
        cache.parent.mkdir(parents=True,exist_ok=True); cache.write_text(json.dumps(obj))
    return obj

def parse_fasta(path:Path):
    out={}; name=None; buf=[]
    with open_text(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith('>'):
                if name is not None: out[name]=''.join(buf).upper()
                name=line[1:].split()[0]; buf=[]
            else: buf.append(line)
    if name is not None: out[name]=''.join(buf).upper()
    return out

def roman(n:int)->str:
    vals=[(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]; out=''
    for v,s in vals:
        while n>=v: out+=s; n-=v
    return out
