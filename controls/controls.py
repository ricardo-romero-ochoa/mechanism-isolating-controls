"""The three mechanism-isolating controls, independent of any one application.

gauge_rank              - control 1: how many parameter directions the data can constrain
best_match / motif_null - control 2: a null that repeats the search you ran
shuffle_control         - control 3a: permute the context, keep everything else
nearest_profile         - control 3b: mechanism-agnostic comparator with the same inputs
poisson_glm / ladder    - the regression the controls are compared in
"""
import numpy as np


def gauge_rank(X, tol=1e-10, centre=False):
    """Effective dimension of a feature set. X: (n_samples, n_features)."""
    A = np.asarray(X, float)
    if centre:
        A = A - A.mean(0)
    G = A.T @ A / max(len(A), 1)
    w = np.linalg.eigvalsh(G)[::-1]
    rank = int((w > tol * w[0]).sum())
    return {"spectrum": w, "n_params": G.shape[0], "rank": rank,
            "n_degenerate": int(G.shape[0] - rank),
            "condition_number": float(w[0] / w[rank - 1]) if rank else float("nan"),
            "cliff_ratio": float(w[rank - 1] / w[rank]) if rank < len(w) else float("inf")}


def best_match(mat, database):
    """Max Pearson r of mat against every profile in database, all offsets, both strands."""
    m = np.asarray(mat, float) - np.asarray(mat, float).mean()
    best = -1.0
    for p in database:
        p = np.asarray(p, float)
        for q in (p, p[::-1, ::-1]):
            w = m.shape[0]
            for off in range(max(q.shape[0] - w + 1, 0)):
                s = q[off:off + w] - q[off:off + w].mean()
                d = np.sqrt((m * m).sum() * (s * s).sum())
                if d > 0:
                    best = max(best, float((m * s).sum() / d))
    return best


def motif_null(learned, database, n_random=500, seed=0):
    """Observed best matches and a null from random matrices searched identically."""
    rng = np.random.default_rng(seed)
    obs = np.array([best_match(m, database) for m in learned])
    shape = np.asarray(learned[0]).shape
    null = np.array([best_match(rng.normal(size=shape), database) for _ in range(n_random)])
    p95 = float(np.percentile(null, 95))
    p_emp = np.array([(1 + np.count_nonzero(null >= x)) / (len(null) + 1) for x in obs])
    return {"observed": obs, "null": null, "empirical_p": p_emp,
            "null_mean": float(null.mean()), "null_p95": p95,
            "null_max": float(null.max()), "n_above_p95": int((obs > p95).sum())}


def poisson_glm(X, y, offset, maxit=80):
    """Poisson regression with a fixed log offset -> (coefs, fitted, deviance)."""
    y = np.asarray(y, float)
    cols = [np.ones(len(y))]
    if X is not None:
        cols += list(np.atleast_2d(np.asarray(X, float).T))
    Xd = np.column_stack(cols)
    b = np.zeros(Xd.shape[1])
    lo = np.log(np.maximum(np.asarray(offset, float), 1e-12))
    for _ in range(maxit):
        mu = np.exp(np.clip(lo + Xd @ b, -30, 30))
        step = np.linalg.solve(Xd.T @ (Xd * mu[:, None]) + 1e-9 * np.eye(Xd.shape[1]),
                               Xd.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    mu = np.exp(np.clip(lo + Xd @ b, -30, 30))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return b, mu, float(2 * (t - (y - mu)).sum())


def poisson_ladder(y, offset, predictors):
    """predictors: name -> array or list of arrays. Returns deviance explained per entry."""
    base = poisson_glm(None, y, offset)[2]
    out = {}
    for name, p in predictors.items():
        X = np.column_stack(p) if isinstance(p, (list, tuple)) else np.asarray(p)
        out[name] = base - poisson_glm(X, y, offset)[2]
    return {"null_deviance": base, "explained": out}


def context_means(U, ctx_state, counts, perm=None):
    """Mean score of each outcome within each (position, state) stratum of contexts.

    U: (n_positions, n_outcomes, n_contexts); ctx_state: (n_contexts, n_positions);
    counts: (n_positions, n_outcomes) opportunities; perm: permutation of contexts.
    """
    L, K, B = U.shape
    cols = np.arange(B) if perm is None else perm
    M = np.zeros((L, K, K))
    for i in range(L):
        acc = np.zeros((K, K))
        np.add.at(acc, ctx_state[:, i], U[i][:, cols].T)
        M[i] = acc / np.maximum(counts[i][:, None], 1)
    return M


def shuffle_control(U, ctx_state, counts, idx, y, offset, covariates=None, n_perm=5, seed=0):
    """Legacy aggregate-context shuffle retained for reproducing pre-gate results.

    This unrestricted shuffle changes the focal-state composition of the backgrounds. It is therefore
    not a valid mechanism-isolation test when the focal ancestral state must be preserved. A correct
    focal-state-preserving permutation must be performed before aggregation, at branch/event level;
    see ``protein/branch_context_gate.py``.
    """
    i, a, b = idx
    rng = np.random.default_rng(seed)
    cov = [] if covariates is None else [np.asarray(c) for c in covariates]
    Mt = context_means(U, ctx_state, counts)
    true_pred = Mt[i, a, b] - Mt[i, a, a]
    base = poisson_glm(np.column_stack(cov) if cov else None, y, offset)[2]
    ex_t = base - poisson_glm(np.column_stack(cov + [true_pred]), y, offset)[2]
    devs = []
    for _ in range(n_perm):
        Ms = context_means(U, ctx_state, counts, perm=rng.permutation(U.shape[2]))
        devs.append(poisson_glm(np.column_stack(cov + [Ms[i, a, b] - Ms[i, a, a]]), y, offset)[2])
    ex_s = base - float(np.mean(devs))
    sd = float(np.std(devs))
    return {"explained_true": ex_t, "explained_shuffled": ex_s, "shuffle_sd": sd,
            "pct_lost": 100.0 * (ex_t - ex_s) / ex_t if ex_t else float("nan"),
            "sds": abs(ex_t - ex_s) / sd if sd else float("inf"), "passes": bool(ex_t > ex_s)}


def nearest_profile(train, ctx, k=50, n_states=20, pseudo=0.01, block=200):
    """Mechanism-agnostic log-frequency profile among the k nearest training contexts."""
    train = np.asarray(train)
    ctx = np.asarray(ctx)
    n, L = train.shape
    q = int(max(train.max(), ctx.max())) + 1
    Xt = np.zeros((n, L * q), np.float32)
    Xt[np.arange(n)[:, None], np.arange(L)[None, :] * q + train] = 1.0
    acc = np.zeros((L, n_states, n_states))
    cnt = np.zeros((L, n_states))
    for s in range(0, ctx.shape[0], block):
        e = min(s + block, ctx.shape[0])
        Xp = np.zeros((e - s, L * q), np.float32)
        Xp[np.arange(e - s)[:, None], np.arange(L)[None, :] * q + ctx[s:e]] = 1.0
        nb = np.argpartition(-(Xp @ Xt.T), k, axis=1)[:, :k]
        H = np.log(Xt[nb].reshape(e - s, k, L, q).mean(1)[:, :, :n_states] + pseudo).astype(float)
        for i in range(L):
            np.add.at(acc[i], ctx[s:e, i], H[:, i, :])
        for a in range(n_states):
            cnt[:, a] += (ctx[s:e] == a).sum(0)
    return acc / np.maximum(cnt[:, :, None], 1)
