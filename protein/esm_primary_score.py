#!/usr/bin/env python3
"""Freeze/validate ESM-2 grouped masking for the five primary G1C families.

Outcome-independent approximation ladder, frozen before any primary-family ESM branch-context
result is inspected:
  G=10 -> G=20 -> G=40 -> G=L (exact one-position masking).
For each family, a deterministic exact subset (seed 1729; up to 64 nodes x 48 sites) is computed
once.  The smallest G passing the pre-existing Gate-1 approximation rule is selected:
  Pearson(delta)>=0.995, median|delta error|<=0.05, top-destination agreement>=0.95.
Only after a G passes is the full node x site masked-marginal tensor materialized.

This script writes protein/esmM_<TAG>.npz and results/gate1_exact_mask_<TAG>.json.
It never reads branch-context outcomes.
"""
from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parents[1]
PROT = ROOT / "protein"
RES = ROOT / "results"
FREEZE = RES / "g1c_primary_family_manifest.json"
AA = "ACDEFGHIKLMNPQRSTVWY-"
SEED = 1729
PASS_RULE = "Pearson contrast r>=0.995, median |delta error|<=0.05, top-destination agreement>=0.95"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def choose_device(x: str) -> torch.device:
    if x != "auto":
        return torch.device(x)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def token_base(tok, st: np.ndarray):
    seqs = ["".join(AA[int(c)] for c in row) for row in st]
    enc = tok(seqs, return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


def exact_subset(model, tok, aa_idx, mask_id, base, attn, nodes, positions, st, device, batch):
    jobs = [(ni, int(p)) for ni in range(len(nodes)) for p in positions]
    exact = np.zeros((len(jobs), 20), np.float32)
    with torch.inference_mode():
        for s in range(0, len(jobs), batch):
            chunk = jobs[s:s+batch]
            take = torch.tensor([int(nodes[j[0]]) for j in chunk], dtype=torch.long)
            ids = base.index_select(0, take).clone()
            am = attn.index_select(0, take).clone()
            for k, (_, p) in enumerate(chunk):
                ids[k, p+1] = mask_id
            ids, am = ids.to(device), am.to(device)
            lg = model(input_ids=ids, attention_mask=am).logits
            lp = torch.log_softmax(lg, dim=-1).index_select(-1, aa_idx)
            for k, (_, p) in enumerate(chunk):
                exact[s+k] = lp[k, p+1].float().cpu().numpy()
    anc = np.array([st[int(nodes[ni]), p] for ni, p in jobs], dtype=np.int64)
    return jobs, exact, anc


def grouped_subset(model, aa_idx, mask_id, base, attn, nodes, L, G, device, batch):
    n = len(nodes)
    out = np.zeros((n, L, 20), np.float32)
    jobs = [(ni, g) for ni in range(n) for g in range(G) if g < L]
    with torch.inference_mode():
        for s in range(0, len(jobs), batch):
            chunk = jobs[s:s+batch]
            take = torch.tensor([int(nodes[j[0]]) for j in chunk], dtype=torch.long)
            ids = base.index_select(0, take).clone()
            am = attn.index_select(0, take).clone()
            for k, (_, g) in enumerate(chunk):
                pos = torch.arange(g, L, G, dtype=torch.long) + 1
                ids[k, pos] = mask_id
            ids, am = ids.to(device), am.to(device)
            lg = model(input_ids=ids, attention_mask=am).logits
            lp = torch.log_softmax(lg, dim=-1).index_select(-1, aa_idx)
            for k, (ni, g) in enumerate(chunk):
                pos = np.arange(g, L, G, dtype=int)
                out[ni, pos] = lp[k, torch.as_tensor(pos+1, device=device)].float().cpu().numpy()
    return out


def metrics(exact, grouped, anc):
    valid = anc < 20
    exact, grouped, anc = exact[valid], grouped[valid], anc[valid]
    rows = np.arange(len(anc))
    ed = exact - exact[rows, anc][:, None]
    gd = grouped - grouped[rows, anc][:, None]
    mask = np.ones_like(ed, dtype=bool)
    mask[rows, anc] = False
    x, y = ed[mask], gd[mask]
    top_e = np.argmax(np.where(mask, ed, -np.inf), axis=1)
    top_g = np.argmax(np.where(mask, gd, -np.inf), axis=1)
    pr = float(pearsonr(x, y).statistic)
    sr = float(spearmanr(x, y).statistic)
    med = float(np.median(np.abs(x-y)))
    p95 = float(np.percentile(np.abs(x-y), 95))
    top = float(np.mean(top_e == top_g))
    return {
        "n_node_position_pairs": int(len(anc)),
        "pearson_delta": pr,
        "spearman_delta": sr,
        "median_abs_delta_error": med,
        "p95_abs_delta_error": p95,
        "top_destination_agreement": top,
        "prespecified_pass": bool(pr >= 0.995 and med <= 0.05 and top >= 0.95),
        "pass_rule": PASS_RULE,
    }


def full_grouped(model, aa_idx, mask_id, base, attn, L, G, device, batch, tag):
    n = base.shape[0]
    out = np.zeros((n, L, 20), np.float32)
    jobs = [(i, g) for i in range(n) for g in range(G) if g < L]
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(jobs), batch):
            chunk = jobs[s:s+batch]
            take = torch.tensor([j[0] for j in chunk], dtype=torch.long)
            ids = base.index_select(0, take).clone()
            am = attn.index_select(0, take).clone()
            for k, (_, g) in enumerate(chunk):
                pos = torch.arange(g, L, G, dtype=torch.long) + 1
                ids[k, pos] = mask_id
            ids, am = ids.to(device), am.to(device)
            lg = model(input_ids=ids, attention_mask=am).logits
            lp = torch.log_softmax(lg, dim=-1).index_select(-1, aa_idx)
            for k, (i, g) in enumerate(chunk):
                pos = np.arange(g, L, G, dtype=int)
                out[i, pos] = lp[k, torch.as_tensor(pos+1, device=device)].float().cpu().numpy()
            if (s // batch) % 100 == 0:
                print(f"  {tag} G={G} {s}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-nodes", type=int, default=64)
    ap.add_argument("--n-pos", type=int, default=48)
    ap.add_argument("--family", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    fr = json.loads(FREEZE.read_text())
    frozen = [x["tag"] for x in fr["primary_families"]]
    tags = args.family or frozen
    bad = [x for x in tags if x not in frozen]
    if bad:
        raise SystemExit(f"not frozen primary families: {bad}")
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(16, os.cpu_count() or 1))
    print(f"ESM model={args.model} device={device} batch={args.batch}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()
    aa_idx = torch.tensor([tok.convert_tokens_to_ids(c) for c in AA[:20]], dtype=torch.long, device=device)
    mask_id = tok.mask_token_id
    RES.mkdir(exist_ok=True)

    for tag in tags:
        bg = PROT / f"bg_{tag}.npz"
        dest = PROT / f"esmM_{tag}.npz"
        jout = RES / f"gate1_exact_mask_{tag}.json"
        if not args.force and dest.exists() and jout.exists():
            old = json.loads(jout.read_text())
            if old.get("prespecified_pass") and old.get("bg_sha256") == sha256(bg):
                print(f"[{tag}] existing validated score matches bg; skip", flush=True)
                continue
        d = np.load(bg)
        st = d["node_states"].astype(np.int64)
        n, L = st.shape
        rng = np.random.default_rng(SEED)
        nodes = np.sort(rng.choice(n, size=min(args.n_nodes, n), replace=False))
        positions = np.sort(rng.choice(L, size=min(args.n_pos, L), replace=False))
        base, attn = token_base(tok, st)
        print(f"[{tag}] nodes={n} L={L}; exact validation subset {len(nodes)}x{len(positions)}", flush=True)
        jobs, exact, anc = exact_subset(model, tok, aa_idx, mask_id, base, attn,
                                        nodes, positions, st, device, args.batch)
        ladder = []
        selected = None
        for G in list(dict.fromkeys([10, 20, 40, L])):
            if G > L:
                continue
            gs = grouped_subset(model, aa_idx, mask_id, base, attn, nodes, L, G, device, args.batch)
            grow = np.vstack([gs[ni, p] for ni, p in jobs]).astype(np.float16).astype(np.float32)
            m = metrics(exact, grow, anc)
            m["groups"] = int(G)
            m["mode"] = "exact_one_position" if G == L else "grouped"
            ladder.append(m)
            print(f"  validation G={G}: r={m['pearson_delta']:.6f} med={m['median_abs_delta_error']:.4g} top={m['top_destination_agreement']:.4f} pass={m['prespecified_pass']}", flush=True)
            if m["prespecified_pass"]:
                selected = G
                break
        if selected is None:
            raise RuntimeError(f"{tag}: even exact-mask fallback failed validation; abort")
        print(f"[{tag}] selected G={selected}; materializing full masked tensor", flush=True)
        full = full_grouped(model, aa_idx, mask_id, base, attn, L, selected, device, args.batch, tag)
        np.savez_compressed(dest, logp=full.astype(np.float16), model=np.array([args.model]),
                            groups=np.array([selected]), seed=np.array([SEED]),
                            bg_sha256=np.array([sha256(bg)]))
        record = {
            "tag": tag,
            "model": args.model,
            "seed": SEED,
            "n_nodes": int(len(nodes)),
            "n_positions": int(len(positions)),
            "mask_ladder_frozen_pre_ESM_outcome": [10,20,40,"L_exact"],
            "ladder_results": ladder,
            "selected_groups": int(selected),
            "selected_mode": "exact_one_position" if selected == L else "grouped",
            "prespecified_pass": True,
            "pass_rule": PASS_RULE,
            "bg_sha256": sha256(bg),
            "score_sha256": sha256(dest),
            "branch_context_outcomes_inspected_for_mask_selection": False,
        }
        jout.write_text(json.dumps(record, indent=2) + "\n")
        print(f"[{tag}] wrote {dest.name} and {jout.name}", flush=True)


if __name__ == "__main__":
    main()
