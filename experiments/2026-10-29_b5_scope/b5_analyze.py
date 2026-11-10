"""B5 E3: summarize E1 (PySCF S2/S0 vs periodic Ewald, + ORCA bridge) and E2 (region A vs C under S2).

Conventions (as B4): stretch = 0.5 u·(F_C1 − F_Cl1), u = unit(C1 − Cl1), positive = stretching C1–Cl1;
approach = 0.5 v·(F_C1 − F_O), v = unit(C1 − O), O = the Asp124 oxygen nearer C1, positive = pushing apart.
Forces in kJ/mol/nm, energies in kJ/mol. Writes b5_results.json; prints the tables.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP.parent))
from openmmorca.qmmm.imaging import minimum_image  # noqa: E402
from openmmorca.units import HARTREE_IN_KJ_MOL, HARTREE_PER_BOHR_IN_KJ_MOL_NM  # noqa: E402

out = {}


def unit_vec(v):
    return v / np.linalg.norm(v)


def coords(pos, names, box):
    """Unit vectors u (Cl1→C1) and v (O→C1) from minimum-image differences."""
    c1, cl = pos[names.index("C1")], pos[names.index("CL1")]
    o = min((pos[names.index(n)] for n in ("OD1", "OD2")),
            key=lambda p: np.linalg.norm(minimum_image(c1 - p, box)))
    return unit_vec(minimum_image(c1 - cl, box)), unit_vec(minimum_image(c1 - o, box)), o


def rc_forces(f, names, u, v, o_pos, pos):
    k1, kc = names.index("C1"), names.index("CL1")
    ko = [i for i, n in enumerate(names) if n in ("OD1", "OD2")]
    ko = min(ko, key=lambda i: np.linalg.norm(pos[i] - o_pos))
    return 0.5 * u @ (f[k1] - f[kc]), 0.5 * v @ (f[k1] - f[ko])


def stats(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "rms": float(np.sqrt((x ** 2).mean())), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()), "n": int(x.size)}


# ---------------- E1: PySCF ----------------
req = np.load(HERE / "requests_t09_A.npz")
names = [str(n) for n in req["qm_names"]]
n_real = len(names)
pys = {}
for p in sorted((HERE / "pyscf_e1").glob("frame_*.json")):
    pys[int(p.stem.split("_")[1])] = json.loads(p.read_text())
frames_e1 = sorted(f for f, d in pys.items() if all(s in d for s in ("S0", "S2", "EW20")))
e1 = {"frames": frames_e1, "method": pys[frames_e1[0]]["S2"]["xc"] + "/" + pys[frames_e1[0]]["S2"]["basis"]}
F = HARTREE_PER_BOHR_IN_KJ_MOL_NM
rows = []
for f in frames_e1:
    box = req["box_nm"][f]
    pos = req[f"S2_qm_pos_{f}"][:n_real]
    u, v, o = coords(pos, names, box)
    r = {"frame": f}
    for s, d in pys[f].items():
        force = -np.array(d["qm_grad_eh_bohr"])[:n_real] * F
        r[s] = {"E": d["energy_eh"] * HARTREE_IN_KJ_MOL, "F": force, "rc": rc_forces(force, names, u, v, o, pos),
                "t": d["t_scf_s"] + d["t_grad_s"]}
    rows.append(r)


def compare(rows, a, b):
    dF = np.array([r[a]["F"] - r[b]["F"] for r in rows if a in r and b in r])
    dE = np.array([r[a]["E"] - r[b]["E"] for r in rows if a in r and b in r])
    ds = [r[a]["rc"][0] - r[b]["rc"][0] for r in rows if a in r and b in r]
    dp = [r[a]["rc"][1] - r[b]["rc"][1] for r in rows if a in r and b in r]
    return {"force_rms": float(np.sqrt((dF ** 2).sum(-1).mean())), "force_max": float(np.abs(dF).max()),
            "stretch": stats(ds), "approach": stats(dp), "dE_frame_std": float(dE.std()), "n": len(dE)}


e1["S2_vs_EW20"] = compare(rows, "S2", "EW20")
e1["S0_vs_EW20"] = compare(rows, "S0", "EW20")
e1["EW30_vs_EW20"] = compare(rows, "EW30", "EW20")
e1["t_s"] = {s: float(np.mean([r[s]["t"] for r in rows if s in r])) for s in ("S0", "S2", "EW20")}

# Bridge (E1c): S0 − S2 stretch shift, ORCA r2SCAN-3c vs PySCF.
t09 = EXP / "2026-10-14_t09_cutoff_scan"
s2dir = EXP / "2026-10-17_b4_embedding" / "s2_t09"
qm_atoms = list(req["qm_atoms"])
s2f = np.load(s2dir / "qm_forces.npz")
assert list(s2f["qm_atoms"]) == qm_atoms
s2e = {r["frame"]: r["energy_kj_mol"] for r in json.loads((s2dir / "stage2.json").read_text()) if r["scheme"] == "S2"}
with np.load(t09 / "scan" / "c1.200.npz") as d:
    s0 = {int(f): (float(d[f"energy_kj_mol_{f}"]), d[f"forces_{f}"][qm_atoms]) for f in d["frames_done"]}
bridge = []
for r in rows:
    f = r["frame"]
    if f not in s0 or f"S2_f{f}" not in s2f.files:
        continue
    box = req["box_nm"][f]
    pos = req[f"S2_qm_pos_{f}"][:n_real]
    u, v, o = coords(pos, names, box)
    orca_s0 = rc_forces(s0[f][1], names, u, v, o, pos)
    orca_s2 = rc_forces(s2f[f"S2_f{f}"], names, u, v, o, pos)
    bridge.append({"frame": f, "orca_dstretch": orca_s0[0] - orca_s2[0], "pyscf_dstretch": r["S0"]["rc"][0] - r["S2"]["rc"][0],
                   "orca_dE": s0[f][0] - s2e[f], "pyscf_dE": r["S0"]["E"] - r["S2"]["E"]})
bo = np.array([b["orca_dstretch"] for b in bridge])
bp = np.array([b["pyscf_dstretch"] for b in bridge])
eo = np.array([b["orca_dE"] for b in bridge])
ep = np.array([b["pyscf_dE"] for b in bridge])
slope = float(bo @ bp / (bp @ bp))
e1["bridge"] = {"n": len(bridge), "stretch_orca": stats(bo), "stretch_pyscf": stats(bp),
                "stretch_corr": float(np.corrcoef(bo, bp)[0, 1]), "stretch_slope_orca_per_pyscf": slope,
                "stretch_resid_rms": float(np.sqrt(((bo - slope * bp) ** 2).mean())),
                "dE_frame_std_orca": float(eo.std()), "dE_frame_std_pyscf": float(ep.std()),
                "dE_corr": float(np.corrcoef(eo, ep)[0, 1]), "dE_diff_frame_std": float((eo - ep).std())}
out["E1"] = e1

# ---------------- E2: region A vs C under S2 ----------------
e2 = {}
for label, sub, frames_name in (("F_A", "sp_fa", "frames_a_s2.npz"), ("F_C", "sp_fc", "frames_c_s0.npz")):
    fr = np.load(HERE / frames_name)
    A, C = np.load(HERE / sub / "A.npz"), np.load(HERE / sub / "C.npz")
    meta_a = json.loads((HERE / sub / "A.meta.json").read_text())
    atoms = list(A["atoms"])
    common = meta_a["qm_atoms"]
    idx = [atoms.index(i) for i in common]
    done = sorted(set(A["frames_done"].tolist()) & set(C["frames_done"].tolist()))
    dF, net, ds, dp, EA, EC, tA, tC = [], [], [], [], [], [], [], []
    for f in done:
        fa = (A[f"f_qm_{f}"] + A[f"f_mm_{f}"])[idx]
        fc = (C[f"f_qm_{f}"] + C[f"f_mm_{f}"])[idx]
        box = fr["box_nm"][f]
        pos = fr["positions_nm"][f][common]
        u, v, o = coords(pos, names, box)
        ra, rc = rc_forces(fa, names, u, v, o, pos), rc_forces(fc, names, u, v, o, pos)
        dF.append(fc - fa)
        net.append((fc - fa).sum(0))
        ds.append(rc[0] - ra[0])
        dp.append(rc[1] - ra[1])
        EA.append(float(A[f"e_qm_{f}"] + A[f"e_mm_{f}"]))
        EC.append(float(C[f"e_qm_{f}"] + C[f"e_mm_{f}"]))
        tA.append(float(A[f"t_orca_{f}"]))
        tC.append(float(C[f"t_orca_{f}"]))
    dF, net, EA, EC = np.array(dF), np.array(net), np.array(EA), np.array(EC)
    per_atom = np.sqrt((dF ** 2).sum(-1).mean(0))
    e2[label] = {"frames": done, "force_rms": float(np.sqrt((dF ** 2).sum(-1).mean())),
                 "force_max": float(np.abs(dF).max()), "net_force_rms": float(np.sqrt((net ** 2).sum(-1).mean())),
                 "stretch": stats(ds), "approach": stats(dp), "dE_frame_std": float((EC - EA).std()),
                 "E_A_frame_std": float(EA.std()), "E_corr": float(np.corrcoef(EA, EC)[0, 1]),
                 "per_atom_rms": {names[k]: float(per_atom[k]) for k in range(len(common))},
                 "t_orca_A": float(np.mean(tA)), "t_orca_C": float(np.mean(tC))}
out["E2"] = e2

(HERE / "b5_results.json").write_text(json.dumps(out, indent=1) + "\n")


def line(name, c):
    return (f"| {name} | {c['n']} | {c['force_rms']:.1f} / {c['force_max']:.1f} | {c['stretch']['mean']:+.1f} / {c['stretch']['rms']:.1f} "
            f"| {c['approach']['mean']:+.1f} / {c['approach']['rms']:.1f} | {c['dE_frame_std']:.2f} |")


print(f"E1 PySCF {e1['method']}, T09 frames {frames_e1}")
print("| comparison | frames | QM force diff RMS / max | stretch diff mean / RMS | approach diff mean / RMS | ΔE frame std |")
for k in ("S2_vs_EW20", "S0_vs_EW20", "EW30_vs_EW20"):
    print(line(k, e1[k]))
print("mean wall per frame (s):", {k: round(v) for k, v in e1["t_s"].items()})
b = e1["bridge"]
print(f"bridge S0−S2 stretch: ORCA mean {b['stretch_orca']['mean']:+.1f} RMS {b['stretch_orca']['rms']:.1f}; "
      f"PySCF mean {b['stretch_pyscf']['mean']:+.1f} RMS {b['stretch_pyscf']['rms']:.1f}; corr {b['stretch_corr']:.3f}; "
      f"slope ORCA/PySCF {b['stretch_slope_orca_per_pyscf']:.3f}; resid RMS {b['stretch_resid_rms']:.1f}")
print(f"bridge S0−S2 ΔE frame std: ORCA {b['dE_frame_std_orca']:.2f}, PySCF {b['dE_frame_std_pyscf']:.2f}, corr {b['dE_corr']:.4f}, "
      f"std of difference {b['dE_diff_frame_std']:.2f}")
print("\nE2 region C − region A (S2, total MM+QM forces on the 14 common atoms)")
print("| frames | n | ΔF RMS / max | 14-atom net ΔF RMS | stretch Δ mean / RMS | approach Δ mean / RMS | ΔE frame std (E_A std, corr) | t_orca A / C (s) |")
for k, c in e2.items():
    print(f"| {k} | {len(c['frames'])} | {c['force_rms']:.1f} / {c['force_max']:.1f} | {c['net_force_rms']:.1f} | "
          f"{c['stretch']['mean']:+.1f} / {c['stretch']['rms']:.1f} | {c['approach']['mean']:+.1f} / {c['approach']['rms']:.1f} | "
          f"{c['dE_frame_std']:.1f} ({c['E_A_frame_std']:.1f}, {c['E_corr']:.3f}) | {c['t_orca_A']:.1f} / {c['t_orca_C']:.1f} |")
    print("  per-atom ΔF RMS:", ", ".join(f"{n} {x:.0f}" for n, x in c["per_atom_rms"].items()))
