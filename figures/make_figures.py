#!/usr/bin/env python3
"""Regenerate the four analysis figures from machine-readable result records.

Usage:
    python figures/make_figures.py

Outputs PNG and vector PDF files to figures/generated/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUT = ROOT / "figures" / "generated"
FAMILIES = ["UBE4B_MOUSE", "TPMT_HUMAN", "HIS7_YEAST", "MTH3_HAEAE", "KKA2_KLEPN"]
DISPLAY = ["UBE4B", "TPMT", "HIS7", "MTH3", "KKA2"]


def style() -> None:
    mpl.rcParams.update({"figure.dpi":110,"savefig.dpi":300,"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"axes.spines.top":False,"axes.spines.right":False,"axes.titlelocation":"left","legend.frameon":False})


def save(fig, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure1(results: Path, outdir: Path) -> None:
    spec = json.loads((results / "gauge_spectrum_slice.json").read_text())
    eig = np.asarray(spec["values"], dtype=float)
    eig_start = int(spec["start_index"])
    identified, gauge = 952, 633
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), gridspec_kw={"wspace": 0.42})
    ax=axes[0]; ax.barh([0],[identified],label="identified"); ax.barh([0],[gauge],left=[identified],label="gauge"); ax.set_xlim(0,identified+gauge); ax.set_yticks([]); ax.set_xlabel("nominal parameters"); ax.set_title("A  Function-space dimension"); ax.text(identified/2,0,"952\nidentified",ha="center",va="center"); ax.text(identified+gauge/2,0,"633\ngauge",ha="center",va="center"); ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.38),ncol=2)
    ax=axes[1]; rank=np.arange(eig_start,eig_start+len(eig)); ax.semilogy(rank,np.maximum(np.abs(eig),1e-18),linewidth=1.0); ax.axvline(952.5,linestyle="--",linewidth=1.0); ax.set_xlim(850,1040); ax.set_xlabel("eigenvalue index"); ax.set_ylabel("uncentred Gram eigenvalue"); ax.set_title("B  Structural rank cliff"); i952=952-eig_start; i953=953-eig_start; ax.text(946,max(abs(eig[i952]),1e-10),"952",ha="right",va="bottom"); ax.text(959,max(abs(eig[i953]),1e-18),"953",ha="left",va="top")
    save(fig,outdir,"Figure1_function_space")


def figure2(results: Path, outdir: Path) -> None:
    disc=json.loads((results/"g1e_large_library_model_status.json").read_text()); val=json.loads((results/"g1b_reb1_independent_null.json").read_text()); discovery_rank=int(disc["reb1_target_ranks"]["TTACCC"]); validation_rank=int(min(val["target_ranks_abs_effect"].values())); cutoff=41
    fig,ax=plt.subplots(figsize=(5.6,3.5)); x=np.array([0,1]); ax.scatter(x,[discovery_rank,validation_rank],s=60,zorder=3); ax.axhspan(1,cutoff,alpha=0.12); ax.axhline(cutoff,linestyle="--",linewidth=1); ax.set_yscale("log"); ax.set_ylim(4096,1); ax.set_xticks(x); ax.set_xticklabels(["main library\nTTACCC","independent library\nbest fixed target"]); ax.set_ylabel("rank among 4,096 6-mers (lower is stronger)"); ax.set_title("Independent validation of the Reb1-like signal"); ax.text(0,discovery_rank,f"  rank {discovery_rank}",va="center"); ax.text(1,validation_rank,f"  rank {validation_rank}",va="center"); ax.text(0.98,cutoff*0.92,"top 1% cutoff",ha="right",va="bottom"); ax.text(1,300,f"999-permutation p = {val['empirical_two_sided_target_p']:.3f}",ha="center"); save(fig,outdir,"Figure2_Reb1_validation")


def _event_specific_fraction(rec: dict, f: str) -> float:
    x=rec["families"][f]; return 1.0-float(x["permuted_gain_mean_ll"])/float(x["context_gain_ll"])


def figure3(results: Path, outdir: Path) -> None:
    mf=json.loads((results/"g1c_primary_mf_status_optaudit.json").read_text()); esm=json.loads((results/"g1c_primary_esm_status_optaudit.json").read_text()); mfz=np.array([mf["families"][f]["context_pairing_z"] for f in FAMILIES]); esz=np.array([esm["families"][f]["context_pairing_z"] for f in FAMILIES]); mff=np.array([_event_specific_fraction(mf,f) for f in FAMILIES]); esf=np.array([_event_specific_fraction(esm,f) for f in FAMILIES]); x=np.arange(len(FAMILIES))
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.3),gridspec_kw={"wspace":0.38}); ax=axes[0]; ax.scatter(x-0.08,mfz,marker="o",s=42,label="MF-DCA"); ax.scatter(x+0.08,esz,marker="s",s=42,label="exact-masked ESM-2"); ax.axhline(2,linestyle="--",linewidth=1); ax.set_xticks(x); ax.set_xticklabels(DISPLAY,rotation=30,ha="right"); ax.set_ylabel("context-pairing z"); ax.set_title("A  Pairing effect"); ax.legend(); ax=axes[1]; ax.scatter(x-0.08,mff,marker="o",s=42,label="MF-DCA"); ax.scatter(x+0.08,esf,marker="s",s=42,label="exact-masked ESM-2"); ax.axhline(0.5,linestyle="--",linewidth=1); ax.set_xticks(x); ax.set_xticklabels(DISPLAY,rotation=30,ha="right"); ax.set_ylim(0,0.8); ax.set_ylabel("share of context gain lost on permutation"); ax.set_title("B  Event-specific share"); save(fig,outdir,"Figure3_protein_context")


def figure4(results: Path, outdir: Path) -> None:
    d=json.loads((results/"gate1_mutability_calibration.json").read_text()); models=["6-mer","occupancy","dinucleotide"]; y=np.arange(len(models)); un=np.array([d["models"][m]["uncalibrated"]["log_odds_per_SD"] for m in models]); unlo=np.array([d["models"][m]["uncalibrated"]["ci_low"] for m in models]); unhi=np.array([d["models"][m]["uncalibrated"]["ci_high"] for m in models]); ca=np.array([d["models"][m]["mutability_calibrated"]["log_odds_per_SD"] for m in models]); calo=np.array([d["models"][m]["mutability_calibrated"]["ci_low"] for m in models]); cahi=np.array([d["models"][m]["mutability_calibrated"]["ci_high"] for m in models]); fig,ax=plt.subplots(figsize=(6.2,3.5)); ax.errorbar(un,y-0.09,xerr=[un-unlo,unhi-un],fmt="o",capsize=3,label="uncalibrated"); ax.errorbar(ca,y+0.09,xerr=[ca-calo,cahi-ca],fmt="s",capsize=3,label="mutability calibrated"); ax.axvline(0,linestyle="--",linewidth=1); ax.axvline(-0.05,linestyle=":",linewidth=1); ax.set_yticks(y); ax.set_yticklabels(models); ax.invert_yaxis(); ax.set_xlabel("log-odds per SD of model sensitivity (95% bootstrap CI)"); ax.set_title("Polymorphism association after trinucleotide mutability calibration"); ax.legend(); save(fig,outdir,"Figure4_mutability_calibration")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--results",type=Path,default=DEFAULT_RESULTS); ap.add_argument("--out",type=Path,default=DEFAULT_OUT); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); style(); figure1(args.results,args.out); figure2(args.results,args.out); figure3(args.results,args.out); figure4(args.results,args.out); print(f"wrote 4 figures as PNG + PDF to {args.out}")


if __name__ == "__main__": main()
