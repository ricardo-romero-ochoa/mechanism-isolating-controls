#!/usr/bin/env python3
"""Summarize G1B/G1E regulatory-gate state without silently running large downloads."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'

def load(name):
    p=RES/name
    if not p.exists(): return None
    try:return json.loads(p.read_text())
    except Exception:return {'parse_error':True}

def main():
    b=load('g1b_reb1_independent_null.json'); mm=load('zhu2014_mutation_model_status.json'); lm=load('g1e_large_library_model_status.json'); pr=load('g1e_position_reconstruction_status.json'); e=load('gate1_mutability_calibration.json')
    sub={
      'G1B': b.get('verdict') if b else 'PENDING',
      'G1E_mutation_model': mm.get('verdict') if mm else 'PENDING',
      'G1E_library_model': lm.get('verdict') if lm else 'PENDING',
      'G1E_positions': pr.get('verdict') if pr else 'PENDING',
      'G1E_final': e.get('overall') if e else 'PENDING'}
    g1b_pass=sub['G1B']=='PASS_REB1_INDEPENDENT_PIPELINE_NULL'
    g1b_fail=sub['G1B']=='FAIL_REB1_INDEPENDENT_VALIDATION' or sub['G1B']=='FAIL_OR_HOLD_REB1_VALIDATION'
    g1b_hold=str(sub['G1B']).startswith('HOLD_')
    g1e=sub['G1E_final']=='PASS_MUTABILITY_CALIBRATED_6MER_ASSOCIATION'
    if any(v=='PENDING' for v in sub.values()) or g1b_hold or str(sub['G1E_mutation_model']).startswith('HOLD_') or str(sub['G1E_library_model']).startswith('HOLD_') or str(sub['G1E_positions']).startswith('HOLD_'):
        overall='INCOMPLETE_REGULATORY_GATES'
    elif g1b_pass and g1e: overall='PASS_G1B_AND_G1E'
    elif g1b_pass and not g1e: overall='G1B_PASS_G1E_FAIL_SELECTION_ASSOCIATION_UNSUPPORTED'
    elif g1b_fail: overall='G1B_FAIL_REB1_VALIDATION'
    else: overall='REGULATORY_GATES_REQUIRE_ADJUDICATION'
    out={'gate':'Regulatory Gate-1 completion','protocol':'REGULATORY_GATE_FREEZE_v1.0','subgates':sub,'overall':overall}
    (RES/'regulatory_gate_status.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
