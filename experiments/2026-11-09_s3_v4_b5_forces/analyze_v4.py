"""Summarize V4: stretch/approach force offsets and energy scatter of S2 and S3 vs ENV20 (and EW20).

Projections as b5_analyze.py: stretch = 0.5 u.(F_C1 - F_Cl1), u = unit(C1 - Cl1);
approach = 0.5 v.(F_C1 - F_O), v = unit(C1 - O), O = the Asp124 oxygen nearer C1 (kJ/mol/nm).
S2, EW20: B5 analytic gradients (experiments/2026-10-29_b5_scope/pyscf_e1); S3: analytic gradient (out/);
ENV20: central differences of energies along the projections (out/, lam from the records).
Controls on frame 0: EW20 central difference vs B5 EW20 analytic; S2chk vs B5 S2.
Writes v4_summary.json for the frames that are complete.  Run with any python that has numpy.
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
EH, FG = units.HARTREE_IN_KJ_MOL, units.HARTREE_PER_BOHR_IN_KJ_MOL_NM

B5 = ROOT / "experiments/2026-10-29_b5_scope"
req = np.load(B5 / "requests_t09_A.npz")
names = [str(n) for n in req["qm_names"]]
n_real = len(names)
k1, kc = names.index("C1"), names.index("CL1")


def load(f, job):
    p = HERE / "out" / f"frame_{f:02d}_{job}.json"
    return json.loads(p.read_text()) if p.exists() else None


def geometry(f):
    pos = req[f"S2_qm_pos_{f}"][:n_real]
    ko = min((names.index(n) for n in ("OD1", "OD2")), key=lambda i: np.linalg.norm(pos[k1] - pos[i]))
    u = pos[k1] - pos[kc]
    v = pos[k1] - pos[ko]
    return u / np.linalg.norm(u), v / np.linalg.norm(v), ko


def proj_from_grad(grad, f):
    F = -np.asarray(grad)[:n_real] * FG
    u, v, ko = geometry(f)
    return np.array([0.5 * u @ (F[k1] - F[kc]), 0.5 * v @ (F[k1] - F[ko])])


def proj_from_fd(f, ref):
    out = []
    for p in ("stretch", "approach"):
        rp, rm = load(f, f"{ref}_{p}_p"), load(f, f"{ref}_{p}_m")
        if rp is None or rm is None:
            return None
        lam_nm = rp["lam_angstrom"] / 10.0
        out.append(-(rp["energy_eh"] - rm["energy_eh"]) * EH / (2 * lam_nm))
    return np.array(out)


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), rms=float(np.sqrt((x ** 2).mean())), std=float(x.std()),
                min=float(x.min()), max=float(x.max()), n=int(x.size))


rows = []
for f in range(20):
    s3, env = load(f, "S3"), load(f, "ENV20")
    fd = {r: proj_from_fd(f, r) for r in ("ENV20", "S2", "S3")}
    if s3 is None or env is None or any(v is None for v in fd.values()):
        continue
    b5 = json.loads((B5 / "pyscf_e1" / f"frame_{f:02d}.json").read_text())
    rows.append(dict(frame=f,
                     E={"S2": b5["S2"]["energy_eh"] * EH, "EW20": b5["EW20"]["energy_eh"] * EH,
                        "S3": s3["energy_eh"] * EH, "ENV20": env["energy_eh"] * EH},
                     P_fd={r: v.tolist() for r, v in fd.items()},
                     P_an={"S2": proj_from_grad(b5["S2"]["qm_grad_eh_bohr"], f).tolist(),
                           "EW20": proj_from_grad(b5["EW20"]["qm_grad_eh_bohr"], f).tolist(),
                           "S3": proj_from_grad(s3["qm_grad_eh_bohr"], f).tolist()}))

summary = dict(frames=[r["frame"] for r in rows], comparisons={},
               note="fd: energy differences along the projections; an: analytic gradients. Mixed fd/an "
                    "differences are not used (common fd-an offset ~16.7 kJ/mol/nm in stretch).")
for kind, a, b in (("fd", "S2", "ENV20"), ("fd", "S3", "ENV20"), ("an", "S2", "EW20"), ("an", "S3", "EW20"),
                   ("an", "S2", "S3"), ("fd", "S2", "S3")):
    key = "P_fd" if kind == "fd" else "P_an"
    dP = np.array([np.subtract(r[key][a], r[key][b]) for r in rows])
    eb = "ENV20" if b in ("ENV20",) else b
    dE = np.array([r["E"][a] - r["E"][eb] for r in rows])
    summary["comparisons"][f"{kind}:{a}-{b}"] = dict(stretch=stats(dP[:, 0]), approach=stats(dP[:, 1]),
                                                     dE_mean=float(dE.mean()), dE_frame_std=float(dE.std()))
fd_an = np.array([np.subtract(r["P_fd"]["S3"], r["P_an"]["S3"]) for r in rows])
summary["S3_fd_minus_an"] = dict(stretch=stats(fd_an[:, 0]), approach=stats(fd_an[:, 1]))
fd_an2 = np.array([np.subtract(r["P_fd"]["S2"], r["P_an"]["S2"]) for r in rows])
summary["S2_fd_minus_an"] = dict(stretch=stats(fd_an2[:, 0]), approach=stats(fd_an2[:, 1]))
controls = {}
ew_fd = proj_from_fd(0, "EW20")
if ew_fd is not None:
    b5 = json.loads((B5 / "pyscf_e1" / "frame_00.json").read_text())
    ew_an = proj_from_grad(b5["EW20"]["qm_grad_eh_bohr"], 0)
    controls["EW20_fd_minus_analytic_frame0"] = (ew_fd - ew_an).tolist()
    controls["EW20_analytic_frame0"] = ew_an.tolist()
s3_fd = proj_from_fd(0, "S3")
if s3_fd is not None and load(0, "S3") is not None:
    s3_an = proj_from_grad(load(0, "S3")["qm_grad_eh_bohr"], 0)
    controls["S3_fd_minus_analytic_frame0"] = (s3_fd - s3_an).tolist()
    controls["S3_analytic_frame0"] = s3_an.tolist()
chk = load(0, "S2chk")
if chk is not None:
    b5 = json.loads((B5 / "pyscf_e1" / "frame_00.json").read_text())
    controls["S2chk_energy_minus_B5_eh"] = chk["energy_eh"] - b5["S2"]["energy_eh"]
    controls["S2chk_grad_max_abs_diff_eh_bohr"] = float(np.abs(np.subtract(chk["qm_grad_eh_bohr"],
                                                                         b5["S2"]["qm_grad_eh_bohr"])).max())
summary["controls"] = controls
summary["rows"] = rows
(HERE / "v4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("frames", summary["frames"])
print(json.dumps(controls, indent=1))
print("S3 fd-an", summary["S3_fd_minus_an"]); print("S2 fd-an", summary["S2_fd_minus_an"])
print("| comparison | stretch mean / RMS / SD | approach mean / RMS / SD | dE frame SD |")
for k, c in summary["comparisons"].items():
    s, p = c["stretch"], c["approach"]
    print(f"| {k} | {s['mean']:+.2f} / {s['rms']:.2f} / {s['std']:.2f} | {p['mean']:+.2f} / {p['rms']:.2f} / "
          f"{p['std']:.2f} | {c['dE_frame_std']:.3f} |")
