"""ORCA/PySCF bridge of the S3 correction on the V5 windows.

C(f) = E_S3(f) - E_S2(f) per program (kJ/mol), same geometry, S2 charges and pseudo charges.
Per window: <C>, SD, lag-1 r1, n_eff, SE (as analyze_v5.py); ddC(W) = <C>_W - <C>_w00 for ORCA and PySCF and
their difference ORCA - PySCF (paired per frame, so its SE uses the per-frame difference).
Smoke identity: w00 f00 S2 vs the C2 pilot ORCA candidate 0, S3 vs V3 Task 5 (API path).  -> bridge_summary.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("units", ROOT / "openmmorca" / "units.py")
units = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(units)
EH = units.HARTREE_IN_KJ_MOL
V5 = ROOT / "experiments/2026-11-09_s3_v5_c1_windows"
WINDOWS, R = (0, 5, 6, 7, 15), 0


def load(base, w, f, s):
    p = base / "out" / f"w{w:02d}_f{f:02d}_{s}.json"
    return json.loads(p.read_text()) if p.exists() else None


def wstats(x):
    x = np.asarray(x, float)
    n = len(x)
    xc = x - x.mean()
    r1 = float((xc[1:] * xc[:-1]).sum() / (xc * xc).sum()) if n > 2 and (xc * xc).sum() > 0 else 0.0
    neff = n * (1 - max(r1, 0.0)) / (1 + max(r1, 0.0))
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    return dict(n=n, mean=float(x.mean()), sd=sd, r1=r1, n_eff=float(neff), se=sd / np.sqrt(neff))


out = dict(windows={}, ddC={}, smoke={})
C = {}
for w in WINDOWS:
    rows = []
    for f in range(20):
        o2, o3 = load(HERE, w, f, "S2"), load(HERE, w, f, "S3")
        p2, p3 = load(V5, w, f, "S2"), load(V5, w, f, "S3")
        if all((o2, o3, p2, p3)):
            rows.append((o2["step"], o3["energy_kj_mol"] - o2["energy_kj_mol"],
                         (p3["energy_eh"] - p2["energy_eh"]) * EH))
    if not rows:
        continue
    rows.sort()
    a = np.array(rows)
    C[w] = dict(orca=a[:, 1], pyscf=a[:, 2], diff=a[:, 1] - a[:, 2])
    out["windows"][w] = {k: wstats(v) for k, v in C[w].items()}
if R in C:
    for w in C:
        if w == R:
            continue
        o, p, d = (out["windows"][w][k]["mean"] - out["windows"][R][k]["mean"] for k in ("orca", "pyscf", "diff"))
        se = lambda k: float(np.hypot(out["windows"][w][k]["se"], out["windows"][R][k]["se"]))
        out["ddC"][w] = dict(orca=o, orca_se=se("orca"), pyscf=p, pyscf_se=se("pyscf"), diff=d, diff_se=se("diff"))
s2, s3 = load(HERE, 0, 0, "S2"), load(HERE, 0, 0, "S3")
if s2:
    out["smoke"]["S2_minus_pilot_kj_mol"] = s2["energy_kj_mol"] - json.loads(
        (ROOT / "experiments/2026-11-06_c2_matched_pilot/orca/candidate_00_S2.json").read_text())["energy_kj_mol"]
if s3:
    out["smoke"]["S3_minus_task5_kj_mol"] = s3["energy_kj_mol"] - json.loads(
        (ROOT / "experiments/2026-11-07_s3_v3_api/orca_00.json").read_text())["energy_kj_mol"]
(HERE / "bridge_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["smoke"], indent=1))
print("| window | n | ORCA <C> / SD | PySCF <C> / SD | ORCA-PySCF <dC> / SD |")
for w, v in out["windows"].items():
    print(f"| w{w:02d} | {v['orca']['n']} | {v['orca']['mean']:.3f} / {v['orca']['sd']:.3f} | "
          f"{v['pyscf']['mean']:.3f} / {v['pyscf']['sd']:.3f} | {v['diff']['mean']:+.3f} / {v['diff']['sd']:.3f} |")
print("| window | ORCA ddC +- SE | PySCF ddC +- SE | ORCA - PySCF +- SE |")
for w, v in out["ddC"].items():
    print(f"| w{w:02d} | {v['orca']:+.3f} +- {v['orca_se']:.3f} | {v['pyscf']:+.3f} +- {v['pyscf_se']:.3f} | "
          f"{v['diff']:+.3f} +- {v['diff_se']:.3f} |")
