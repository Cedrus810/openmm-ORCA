"""Roadmap B1 follow-up: where the QM-atom force difference sits, and the replicate noise floor.

Needs b1_arrays.npz from b1_embedding_charge.py.
* Noise floor: T09 smoke (frame 0, np32) vs the scan (frame 0, np16) at 1.2 and 2.5 nm: same geometry
  and same selection, both fresh SCF; differences are run-to-run noise.
* Per QM atom: RMS over frames of ΔF = F(c) − F(2.5) and of the residual ΔF + F_cl(shell).
* Net force on the QM region (ΣΔF vs −ΣF_cl) and the best scale α in ΔF ≈ −α F_cl.

Usage: b1_force_check.py T09_DIR B1_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import openmm.app as app

t09, b1 = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.loads((t09 / "scan" / "meta.json").read_text())
qm = meta["qm_atoms"]
atoms = list(app.PDBFile(str(t09 / meta["run_files"] / "prepared.pdb")).topology.atoms())
names = [f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}" for i in qm]
arr = np.load(b1 / "b1_arrays.npz")
res = json.loads((b1 / "b1_results.json").read_text())
frames = res["frames"]


def npz_frame(path, f):
    with np.load(path) as d:
        return float(d[f"energy_kj_mol_{f}"]), d[f"forces_{f}"][qm], int(d[f"n_embed_groups_{f}"])


print("Noise floor (smoke np32 vs scan np16, frame 0):")
for c in ("1.200", "2.500"):
    e1, f1, n1 = npz_frame(t09 / "smoke" / f"c{c}.npz", 0)
    e2, f2, n2 = npz_frame(t09 / "scan" / f"c{c}.npz", 0)
    d = f1 - f2
    print(f"  {c} nm: groups {n1}/{n2}, |ΔE| {abs(e1 - e2):.2e} kJ/mol, QM RMS|ΔF| "
          f"{np.sqrt((d ** 2).sum(1).mean()):.2e}, max|ΔF| {np.linalg.norm(d, axis=1).max():.2e} kJ/mol/nm")

for s in res["cutoffs"][:-1]:
    c = f"{s['cutoff_nm']:.3f}"
    dF = np.stack([arr[f"dF_c{c}_f{f}"] for f in frames])
    fcl = np.stack([arr[f"FclC_c{c}_f{f}"] + arr[f"FclN_c{c}_f{f}"] for f in frames])
    pred = -fcl
    resid = dF - pred
    alpha = float((dF * pred).sum() / (pred * pred).sum())
    net_d, net_p = dF.sum(1), pred.sum(1)
    print(f"\ncutoff {c} nm: α = {alpha:.2f}; net force on QM region RMS: ΔF {np.sqrt((net_d ** 2).sum(1).mean()):.1f}, "
          f"pred {np.sqrt((net_p ** 2).sum(1).mean()):.1f}, resid {np.sqrt(((net_d - net_p) ** 2).sum(1).mean()):.1f}")
    print("  atom | RMS ΔF | RMS pred | RMS resid")
    for k, n in enumerate(names):
        r = lambda x: np.sqrt((x[:, k] ** 2).sum(-1).mean())  # noqa: E731
        print(f"  {n:<14} | {r(dF):7.1f} | {r(pred):7.1f} | {r(resid):7.1f}")

# Residual along the C1–Cl1 bond (the SN2 reaction bond): bond-stretch component of the residual,
# (r_C1 − r_Cl1)·(res_C1 − res_Cl1)/2, compared with the residual's full size on the two atoms.
fr = np.load(t09 / meta["frames_npz"])
k_c1, k_cl1 = names.index("DCE600:C1"), names.index("DCE600:CL1")
print("\nC1–Cl1 residual: stretch component vs total (RMS over frames, kJ/mol/nm)")
for s in res["cutoffs"][:-1]:
    c = f"{s['cutoff_nm']:.3f}"
    stretch, total, along_pred = [], [], []
    for f in frames:
        pos = fr["positions_nm"][f][qm]
        u = pos[k_c1] - pos[k_cl1]
        u /= np.linalg.norm(u)
        d = arr[f"dF_c{c}_f{f}"]
        p = -(arr[f"FclC_c{c}_f{f}"] + arr[f"FclN_c{c}_f{f}"])
        r = d - p
        stretch.append(0.5 * u @ (r[k_c1] - r[k_cl1]))
        along_pred.append(0.5 * u @ (p[k_c1] - p[k_cl1]))
        total.append(np.sqrt(0.5 * ((r[k_c1] ** 2).sum() + (r[k_cl1] ** 2).sum())))
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
    print(f"  {c} nm: stretch resid {rms(stretch):6.1f}, classical stretch {rms(along_pred):5.1f}, "
          f"per-atom resid {rms(total):6.1f}")

# Is the C1–Cl1 stretch change driven by the shell's electric field along the bond? The classical
# shell force on Cl1 is q_Cl1·E_shell(Cl1), so u·F_cl(Cl1) is the field component along the bond up
# to the constant factor q_Cl1; the sign of the correlation is therefore not meaningful.
print("\ncorrelation over frames: C1–Cl1 stretch ΔF vs shell field along the bond at Cl1")
for s in res["cutoffs"][:-1]:
    c = f"{s['cutoff_nm']:.3f}"
    x, y = [], []
    for f in frames:
        pos = fr["positions_nm"][f][qm]
        u = pos[k_c1] - pos[k_cl1]
        u /= np.linalg.norm(u)
        d = arr[f"dF_c{c}_f{f}"]
        fcl = arr[f"FclC_c{c}_f{f}"] + arr[f"FclN_c{c}_f{f}"]
        y.append(0.5 * u @ (d[k_c1] - d[k_cl1]))
        x.append(u @ fcl[k_cl1])
    print(f"  {c} nm: r = {np.corrcoef(x, y)[0, 1]:+.3f}")
