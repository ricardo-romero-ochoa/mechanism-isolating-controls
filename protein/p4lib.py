"""Phase 2 pipeline: Potts models vs tree-only nulls for substitution recurrence."""
import numpy as np, gc
AA = "ACDEFGHIKLMNPQRSTVWY-"
I20 = {c: i for i, c in enumerate(AA)}
FULL = (1 << 20) - 1


def read_sto(path):
    nm, d = [], {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("#") or line.startswith("//") or not line.strip():
            continue
        p = line.rstrip("\n").split(None, 1)
        if len(p) < 2:
            continue
        if p[0] not in d:
            d[p[0]] = []
            nm.append(p[0])
        d[p[0]].append(p[1].strip())
    return nm, ["".join(d[k]) for k in nm]


def match_matrix(path):
    """Pfam full alignment -> (names, int8 matrix over match states; 20 = gap)."""
    nm, sq = read_sto(path)
    A = np.array([list(s) for s in sq])
    m = (A == ".").sum(0) == 0
    A = np.char.upper(A[:, m])
    out = np.full(A.shape, 20, np.int8)
    for c, i in I20.items():
        out[A == c] = i
    out[A == "."] = 20
    return nm, out


def place_target(hmm_path, seq, name=b"target"):
    """Map 1-based residue positions of seq onto match-state columns via hmmalign."""
    import pyhmmer
    with pyhmmer.plan7.HMMFile(hmm_path) as fh:
        hmm = fh.read()
    abc = hmm.alphabet
    sl = pyhmmer.easel.TextSequence(name=name, sequence=seq).digitize(abc)
    msa = pyhmmer.hmmer.hmmalign(hmm, [sl])
    aln = msa.alignment[0]
    res2col, col, res = {}, 0, 0
    for ch in aln:
        if ch == "-":
            col += 1
        elif ch == ".":
            res += 1
        elif ch.isupper():
            res += 1
            col += 1
            res2col[res] = col - 1
        else:
            res += 1
    return res2col, hmm.M


def coverage_filter(A, thr=0.8):
    return (A != 20).mean(1) >= thr


def weights(A, thr=0.8, block=2000):
    n, L = A.shape
    X = np.zeros((n, L * 21), np.float32)
    X[np.arange(n)[:, None], np.arange(L)[None, :] * 21 + A] = 1.0
    cnt = np.zeros(n)
    for s in range(0, n, block):
        sim = X[s:s + block] @ X.T
        cnt[s:s + block] = (sim >= thr * L).sum(1)
    del X
    gc.collect()
    return 1.0 / np.maximum(cnt, 1)


def profile(A, w, lam):
    L = A.shape[1]
    M = w.sum()
    fi = np.zeros((L, 21))
    for a in range(21):
        fi[:, a] = (w[:, None] * (A == a)).sum(0)
    return (fi + lam / 21) / (M + lam), M


def mf_couplings(A, w, lam=None):
    """Mean-field DCA: returns J (L,L,20,20) zero-sum gauge and profile fields (L,20)."""
    n, L = A.shape
    M = w.sum()
    lam = M if lam is None else lam
    fi, _ = profile(A, w, lam)
    X = np.zeros((n, L * 21), np.float32)
    X[np.arange(n)[:, None], np.arange(L)[None, :] * 21 + A] = 1.0
    X *= np.sqrt(w / (M + lam))[:, None].astype(np.float32)
    C = X.T.astype(np.float64) @ X.astype(np.float64)
    del X
    gc.collect()
    C += lam / (21 * 21 * (M + lam))
    for i in range(L):
        blk = C[i * 21:(i + 1) * 21, i * 21:(i + 1) * 21]
        blk[:] = 0.0
        blk[np.arange(21), np.arange(21)] = fi[i]
    C -= np.outer(fi.ravel(), fi.ravel())
    sel = np.array([i * 21 + a for i in range(L) for a in range(20)])
    Cr = C[np.ix_(sel, sel)]
    del C
    gc.collect()
    J = -np.linalg.inv(Cr)
    del Cr
    gc.collect()
    J = J.reshape(L, 20, L, 20).transpose(0, 2, 1, 3)
    for i in range(L):
        J[i, i] = 0.0
    J -= J.mean(2, keepdims=True)
    J -= J.mean(3, keepdims=True)
    J += J.mean((2, 3), keepdims=True)
    return J, np.log(fi[:, :20]), M


def pdist(A, block=400):
    n, L = A.shape
    X = np.zeros((n, L * 21), np.float32)
    X[np.arange(n)[:, None], np.arange(L)[None, :] * 21 + A] = 1.0
    G = (A != 20).astype(np.float32)
    D = np.zeros((n, n), np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        both = G[s:e] @ G.T
        mt = X[s:e, :] @ X.T - ((A[s:e] == 20).astype(np.float32) @ (A == 20).T.astype(np.float32))
        D[s:e] = 1.0 - mt / np.maximum(both, 1)
    del X, G
    gc.collect()
    np.fill_diagonal(D, 0.0)
    return D


def neighbour_join2(D):
    """Neighbour joining with a proper final root. Returns (children, root, parent)."""
    n = D.shape[0]
    d = D.astype(np.float64).copy()
    act = list(range(n))
    nxt = n
    ch, par = {}, {}
    big = 1e18
    dm = np.full((2 * n, 2 * n), big)
    dm[:n, :n] = d
    np.fill_diagonal(dm, 0.0)
    while len(act) > 2:
        idx = np.array(act)
        sub = dm[np.ix_(idx, idx)]
        r = sub.sum(1)
        m = len(idx)
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, big)
        a, b = np.unravel_index(np.argmin(Q), Q.shape)
        ia, ib = idx[a], idx[b]
        ch[nxt] = (ia, ib)
        par[ia] = nxt
        par[ib] = nxt
        newd = 0.5 * (sub[a] + sub[b] - sub[a, b])
        dm[nxt, idx] = newd
        dm[idx, nxt] = newd
        dm[nxt, nxt] = 0.0
        act.remove(ia)
        act.remove(ib)
        act.append(nxt)
        nxt += 1
    ch[nxt] = (act[0], act[1])
    par[act[0]] = nxt
    par[act[1]] = nxt
    return ch, nxt, par


def traverse(ch, root):
    order, st = [], [root]
    while st:
        v = st.pop()
        order.append(v)
        c = ch.get(v)
        if c:
            st.extend(c)
    return order


def low_bit(m):
    out = np.zeros(m.shape, np.int8)
    mm = m.copy()
    for b in range(20):
        sel = (mm & (1 << b)) != 0
        out = np.where(sel & (out == 0) & (mm != 0), b, out)
        mm = np.where(sel, 0, mm)
    return out


def fitch(S, ch, root):
    """Fitch down/up pass. Returns (states per node, list of (parent, child) edges)."""
    Lx = S.shape[1]
    order = traverse(ch, root)
    NN = max([root] + [k for k, v in ch.items() if v] + [x for v in ch.values() if v for x in v]) + 1
    m = np.zeros((NN, Lx), np.uint32)
    for t in range(len(S)):
        s = S[t]
        m[t] = np.where(s < 20, 1 << np.minimum(s, 19).astype(np.uint32), FULL)
    for v in order[::-1]:
        c = ch.get(v)
        if not c:
            continue
        it = m[c[0]] & m[c[1]]
        m[v] = np.where(it != 0, it, m[c[0]] | m[c[1]])
    stt = np.zeros((NN, Lx), np.int8)
    stt[root] = low_bit(m[root])
    edges = []
    for v in order:
        c = ch.get(v)
        if not c:
            continue
        for k in c:
            kp = (m[k] & (1 << stt[v].astype(np.uint32))) != 0
            stt[k] = np.where(kp, stt[v], low_bit(m[k]))
            edges.append((v, k))
    return stt, edges


def poisson(X, y, off, maxit=80):
    Xd = np.column_stack([np.ones(len(y))] + ([] if X is None else list(np.atleast_2d(X.T))))
    b = np.zeros(Xd.shape[1])
    lo = np.log(np.maximum(off, 1e-12))
    for _ in range(maxit):
        mu = np.exp(np.clip(lo + Xd @ b, -30, 30))
        step = np.linalg.solve(Xd.T @ (Xd * mu[:, None]) + 1e-9 * np.eye(Xd.shape[1]), Xd.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    mu = np.exp(np.clip(lo + Xd @ b, -30, 30))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return b, mu, 2 * (t - (y - mu)).sum()


def concordance(v, y, ii, aa, sites, O):
    num = den = 0.0
    for i in sites:
        for a in range(20):
            if O[i, a] < 1:
                continue
            m = (ii == i) & (aa == a)
            if m.sum() < 2:
                continue
            yy, vv = y[m], v[m]
            dy = np.sign(yy[:, None] - yy[None, :])
            dv = np.sign(vv[:, None] - vv[None, :])
            num += (dy * dv > 0).sum()
            den += (dy != 0).sum()
    return num / den


def recurrence_table(S, ch, root):
    """Opportunities, counts, exchangeability and candidate index arrays."""
    L = S.shape[1]
    stt, edges = fitch(S, ch, root)
    par = np.array([e[0] for e in edges])
    chd = np.array([e[1] for e in edges])
    Ps, Cs = stt[par], stt[chd]
    B = len(edges)
    O = np.zeros((L, 20), np.int64)
    for a in range(20):
        O[:, a] = (Ps == a).sum(0)
    Nc = np.zeros((L, 20, 20), np.int64)
    db, di = np.nonzero(Ps != Cs)
    np.add.at(Nc, (di, Ps[db, di], Cs[db, di]), 1)
    Ni = Nc.sum((1, 2))
    q = Nc.sum(0).astype(float) + 0.5
    np.fill_diagonal(q, 0.0)
    q /= q.sum(1, keepdims=True)
    sites = np.flatnonzero(Ni > 0)
    ii, aa, bb = [], [], []
    for i in sites:
        for a in range(20):
            if O[i, a] < 1:
                continue
            for b in range(20):
                if b != a:
                    ii.append(i)
                    aa.append(a)
                    bb.append(b)
    # Explicit integer dtype is essential when there are no candidate
    # (site, ancestral, destination) tuples.  np.array([]) defaults to
    # float64, which cannot be used for NumPy indexing and caused the
    # Stage-B crash on highly conserved candidate clades.
    ii = np.asarray(ii, dtype=np.int64)
    aa = np.asarray(aa, dtype=np.int64)
    bb = np.asarray(bb, dtype=np.int64)
    return dict(stt=stt, Ps=Ps, Cs=Cs, B=B, O=O, Nc=Nc, Ni=Ni, q=q, sites=sites, ii=ii, aa=aa, bb=bb,
                y=Nc[ii, aa, bb].astype(float),
                E0=O[ii, aa] * (Ni[ii] / B) * q[aa, bb])


def stratum_means(U, Ps, O, perm=None):
    """Mean statistical energy of each state, averaged over the backgrounds of each (site, state) stratum."""
    L, _, B = U.shape
    cols = np.arange(B) if perm is None else perm
    M = np.zeros((L, 20, 20))
    for i in range(L):
        acc = np.zeros((20, 20))
        np.add.at(acc, Ps[:, i], U[i][:, cols].T)
        M[i] = acc / np.maximum(O[i][:, None], 1)
    return M


def background_energies(J, h, Ps):
    """U[i,a,branch] = h[i,a] + sum_j J[i,j,a,parent_state_j]."""
    L = J.shape[0]
    B = Ps.shape[0]
    Jm = np.ascontiguousarray(J.transpose(0, 2, 1, 3).reshape(L * 20, L * 20))
    Oh = np.zeros((L * 20, B))
    Oh[(np.arange(L)[:, None] * 20 + Ps.T), np.arange(B)[None, :]] = 1.0
    U = (Jm @ Oh).reshape(L, 20, B)
    del Jm, Oh
    gc.collect()
    return U + h[:, :, None]


def subfamily_scores(Atr, Ps, O, k=50, pc=0.01, block=200):
    """Log frequency of each state at each site among the k training sequences nearest each background."""
    Ntr, L = Atr.shape
    B = Ps.shape[0]
    Xt = np.zeros((Ntr, L * 21), np.float32)
    Xt[np.arange(Ntr)[:, None], np.arange(L)[None, :] * 21 + Atr] = 1.0
    Xp = np.zeros((B, L * 21), np.float32)
    Xp[np.arange(B)[:, None], np.arange(L)[None, :] * 21 + Ps] = 1.0
    M = np.zeros((L, 20, 20))
    acc = np.zeros((L, 20, 20))
    cnt = np.zeros((L, 20))
    for s in range(0, B, block):
        e = min(s + block, B)
        sim = Xp[s:e] @ Xt.T
        nb = np.argpartition(-sim, k, axis=1)[:, :k]
        F = Xt[nb].reshape(e - s, k, L, 21).mean(1)
        H = np.log(F[:, :, :20] + pc).astype(np.float64)
        for i in range(L):
            np.add.at(acc[i], Ps[s:e, i], H[:, i, :])
        for a in range(20):
            cnt[:, a] += (Ps[s:e] == a).sum(0)
    del Xt, Xp
    gc.collect()
    M = acc / np.maximum(cnt[:, :, None], 1)
    return M
