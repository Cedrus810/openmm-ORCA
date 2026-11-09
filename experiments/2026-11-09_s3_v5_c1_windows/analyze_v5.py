"""V5 summary: C1 window statistics of the embedding error for S2 and S3 (PySCF r2SCAN/def2-SVP).

D_X(f) = E_X(f) - E_ENV20(f) (kJ/mol), X = S2, S3.  Per window: mean, SD (ddof=1), lag-1 autocorrelation r1 of
D in saved-step order, effective sample size n_eff = n (1 - r1)/(1 + r1) (r1 < 0 taken as 0), SE = SD/sqrt(n_eff).
dd(W) = <D>_W - <D>_R (R = w00), SE = sqrt(SE_W^2 + SE_R^2).  C1 target |dd| <= 5 kJ/mol (user, 2026-11-01).
Also: smoke identity of w00 frame 0 against the C2 pilot candidate 0 (same frame).  Writes v5_summary.json.
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
WINDOWS, R = (0, 5, 6, 7, 15), 0


def load(w, f, s):
    p = HERE / "out" / f"w{w:02d}_f{f:02d}_{s}.json"
    return json.loads(p.read_text()) if p.exists() else None


def window_stats(x):
    x = np.asarray(x, float)
    n = len(x)
    xc = x - x.mean()
    r1 = float((xc[1:] * xc[:-1]).sum() / (xc * xc).sum()) if n > 2 and (xc * xc).sum() > 0 else 0.0
    neff = n * (1 - max(r1, 0.0)) / (1 + max(r1, 0.0))
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    return dict(n=n, mean=float(x.mean()), sd=sd, r1=r1, n_eff=float(neff), se=sd / np.sqrt(neff))


out = dict(windows={}, dd={}, smoke={})
D = {}
for w in WINDOWS:
    rows = []
    for f in range(20):
        r = {s: load(w, f, s) for s in ("ENV20", "S2", "S3")}
        if all(r.values()):
            rows.append((r["ENV20"]["step"], r["ENV20"]["rc_nm"],
                         (r["S2"]["energy_eh"] - r["ENV20"]["energy_eh"]) * EH,
                         (r["S3"]["energy_eh"] - r["ENV20"]["energy_eh"]) * EH))
    if not rows:
        continue
    rows.sort()
    arr = np.array(rows)
    D[w] = {"S2": arr[:, 2], "S3": arr[:, 3]}
    out["windows"][w] = dict(n=len(rows), rc_mean=float(arr[:, 1].mean()), rc_min=float(arr[:, 1].min()),
                             rc_max=float(arr[:, 1].max()),
                             S2=window_stats(arr[:, 2]), S3=window_stats(arr[:, 3]))
if R in D:
    for w in D:
        if w == R:
            continue
        out["dd"][w] = {}
        for x in ("S2", "S3"):
            a, b = out["windows"][w][x], out["windows"][R][x]
            out["dd"][w][x] = dict(dd=a["mean"] - b["mean"], se=float(np.hypot(a["se"], b["se"])),
                                   within_5=bool(abs(a["mean"] - b["mean"]) <= 5.0))
pilot = ROOT / "experiments/2026-11-06_c2_matched_pilot/pyscf"
for s in ("S2", "ENV20"):
    r = load(0, 0, s)
    if r:
        out["smoke"][f"{s}_minus_pilot_eh"] = r["energy_eh"] - json.loads(
            (pilot / f"candidate_00_{s}.json").read_text())["energy_eh"]
r = load(0, 0, "S3")
if r:
    t0 = json.loads((ROOT / "experiments/2026-11-07_s3_v3_prep/out/candidate_00.json").read_text())
    out["smoke"]["S3_minus_task0_T2_P_r0.6_kj_mol"] = (r["energy_eh"] - t0["energies_eh"]["T2_P_r0.6"]) * EH
(HERE / "v5_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["smoke"], indent=1))
print("| window | n | RC mean (min..max) | S2 <D> / SD / r1 / n_eff | S3 <D> / SD / r1 / n_eff |")
for w, v in out["windows"].items():
    a, b = v["S2"], v["S3"]
    print(f"| w{w:02d} | {v['n']} | {v['rc_mean']:+.3f} ({v['rc_min']:+.3f}..{v['rc_max']:+.3f}) | "
          f"{a['mean']:.2f} / {a['sd']:.2f} / {a['r1']:.2f} / {a['n_eff']:.1f} | "
          f"{b['mean']:.3f} / {b['sd']:.3f} / {b['r1']:.2f} / {b['n_eff']:.1f} |")
print("| window | S2 dd +- SE | S3 dd +- SE |")
for w, v in out["dd"].items():
    print(f"| w{w:02d} | {v['S2']['dd']:+.2f} +- {v['S2']['se']:.2f} | {v['S3']['dd']:+.3f} +- {v['S3']['se']:.3f} |")
