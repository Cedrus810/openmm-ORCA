"""Roadmap B3-1: region A vs region C on the 20 T09 frames (cutoff 1.2 nm, r2SCAN-3c).

Three single-point sets on the same geometries:
  A(S_A)  region A, its own selection      T09 scan/c1.200.npz (183, np16)
  A(S_C)  region A, region C's residues     sp_a_setC/sp_w*.npz (B3-1b)
  C(S_C)  region C, its own selection      sp_c/w*/c1.200.npz (B3-1a)
A(S_A) → A(S_C) is the embedded-set change, A(S_C) → C(S_C) the QM-region change.

Region C moves the DCE/Asp124–Trp/His interactions (electrostatics and Lennard-Jones) from MM into
QM, so QM-group forces alone are not comparable. Every comparison uses total forces and energies:
the MM part of each mixed System (all force groups except the QM group, OpenMM CPU platform, no
ORCA) plus the stored QM part. The MM part does not depend on the embedded set.

Compared on the 14 atoms that are QM in both regions (DCE + Asp124 side chain): total force
difference (RMS/max), the C1–Cl1 stretch component 0.5·u·(F_C1 − F_Cl1), the net force on the 14
atoms, and the frame-to-frame part of the total energy difference (std over frames; the constant
offset between different QM regions is not an accuracy measure).

Usage: b3_analyze_sp.py B3_DIR T09_DIR
Writes B3_DIR/b3_sp_results.json and prints Markdown tables.
"""
import json
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import enzyme_qmmm as enzyme  # noqa: E402

from openmmorca import ORCAPotential  # noqa: E402

QM_GROUP = 1
KJ = unit.kilojoule_per_mole


def load_scan(paths):
    out = {}
    for p in paths:
        with np.load(p) as d:
            for f in d["frames_done"]:
                f = int(f)
                out[f] = (float(d[f"energy_kj_mol_{f}"]), d[f"forces_{f}"])
    return out


def load_sparse(directory, n_atoms):
    out = {}
    for p in sorted(directory.glob("sp_w*.npz")):
        if p.stem.endswith(".tmp"):
            continue
        with np.load(p) as d:
            for j in d["jobs_done"]:
                j = int(j)
                f = np.zeros((n_atoms, 3))
                f[d[f"force_atoms_{j}"]] = d[f"forces_{j}"]
                out[j] = (float(d[f"energy_kj_mol_{j}"]), f)
    return out


def mm_part(topology, system, region, frames):
    """Energy and forces of every force group except the QM group, per frame."""
    qm, pairs, *_ = enzyme.qm_selection(topology, region)
    potential = ORCAPotential(method="r2SCAN-3c", charge=-1, multiplicity=1, nprocs=1)
    try:
        mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=QM_GROUP,
                                            embeddingCutoff=1.2 * unit.nanometer)
        groups = {f.getForceGroup() for f in mixed.getForces()} - {QM_GROUP}
        context = mm.Context(mixed, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("CPU"))
        out = []
        for row in range(len(frames["step"])):
            context.setPeriodicBoxVectors(*frames["box_nm"][row])
            context.setPositions(frames["positions_nm"][row])
            s = context.getState(getEnergy=True, getForces=True, groups=groups)
            out.append((s.getPotentialEnergy().value_in_unit(KJ),
                        s.getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer)))
        return qm, out
    finally:
        potential.close()


def main():
    b3, t09 = Path(sys.argv[1]), Path(sys.argv[2])
    frames = np.load(t09 / "frames.npz")
    run_files = t09 / "run_files"
    pdb = app.PDBFile(str(run_files / "prepared.pdb"))
    topology = pdb.topology
    state = mm.XmlSerializer.deserialize((run_files / "equilibrated.xml").read_text())
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    n_atoms = system.getNumParticles()

    qm_a, mm_a = mm_part(topology, system, "A", frames)
    qm_c, mm_c = mm_part(topology, system, "C", frames)
    common = list(qm_a)
    assert set(common) <= set(qm_c)
    atoms = list(topology.atoms())
    names = [f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}" for i in common]
    k_c1, k_cl1 = names.index("DCE600:C1"), names.index("DCE600:CL1")

    qm_sets = {
        "A(S_A)": load_scan([t09 / "scan" / "c1.200.npz"]),
        "A(S_C)": load_sparse(b3 / "sp_a_setC", n_atoms),
        "C(S_C)": load_scan(sorted((b3 / "sp_c").glob("w*/c1.200.npz"))),
    }
    mm_of = {"A(S_A)": mm_a, "A(S_C)": mm_a, "C(S_C)": mm_c}
    frame_ids = sorted(set.intersection(*(set(v) for v in qm_sets.values())))
    print(f"frames with all three sets: {len(frame_ids)}/20")

    total = {}
    for name, qm_part in qm_sets.items():
        total[name] = {f: (mm_of[name][f][0] + qm_part[f][0], mm_of[name][f][1] + qm_part[f][1]) for f in frame_ids}

    def stretch(f, F):
        pos = frames["positions_nm"][f][common]
        u = pos[k_c1] - pos[k_cl1]
        u /= np.linalg.norm(u)
        return 0.5 * float(u @ (F[k_c1] - F[k_cl1]))

    def compare(a, b):
        dF = np.stack([total[b][f][1][common] - total[a][f][1][common] for f in frame_ids])
        dE = np.array([total[b][f][0] - total[a][f][0] for f in frame_ids])
        ea = np.array([total[a][f][0] for f in frame_ids])
        eb = np.array([total[b][f][0] for f in frame_ids])
        st = np.array([stretch(f, total[b][f][1][common]) - stretch(f, total[a][f][1][common]) for f in frame_ids])
        sa = np.array([stretch(f, total[a][f][1][common]) for f in frame_ids])
        return {
            "pair": f"{a} → {b}",
            "dF_rms": float(np.sqrt((dF ** 2).sum(-1).mean())),
            "dF_max": float(np.linalg.norm(dF, axis=-1).max()),
            "dF_rms_per_atom": {n: float(np.sqrt((dF[:, k] ** 2).sum(-1).mean())) for k, n in enumerate(names)},
            "net_dF_rms": float(np.sqrt((dF.sum(1) ** 2).sum(-1).mean())),
            "stretch_d_rms": float(np.sqrt(np.mean(st ** 2))), "stretch_d_mean": float(st.mean()),
            "stretch_a_rms": float(np.sqrt(np.mean(sa ** 2))),
            "dE_offset": float(dE.mean()), "dE_frame_std": float(dE.std()),
            "E_frame_std_a": float(ea.std()), "corr_E": float(np.corrcoef(ea, eb)[0, 1]),
        }

    pairs = [compare("A(S_A)", "A(S_C)"), compare("A(S_C)", "C(S_C)"), compare("A(S_A)", "C(S_C)")]
    fmag = np.concatenate([np.linalg.norm(total["C(S_C)"][f][1][common], axis=-1) for f in frame_ids])
    print(f"total force on the 14 common atoms in C(S_C): median {np.median(fmag):.0f}, max {fmag.max():.0f} kJ/mol/nm\n")
    print("| comparison | common-atom ΔF RMS | ΔF max | net ΔF RMS | C1–Cl1 stretch Δ RMS (mean) "
          "| ΔE frame-to-frame std | E std of first | corr E |")
    print("|---|---|---|---|---|---|---|---|")
    for p in pairs:
        print(f"| {p['pair']} | {p['dF_rms']:.1f} | {p['dF_max']:.1f} | {p['net_dF_rms']:.1f} | "
              f"{p['stretch_d_rms']:.1f} ({p['stretch_d_mean']:+.1f}) | {p['dE_frame_std']:.1f} | "
              f"{p['E_frame_std_a']:.1f} | {p['corr_E']:.3f} |")
    print("\nper-atom total-force difference RMS (kJ/mol/nm)\n")
    print("| atom | " + " | ".join(p["pair"] for p in pairs) + " |")
    print("|---|" + "---|" * len(pairs))
    for n in names:
        print(f"| {n} | " + " | ".join(f"{p['dF_rms_per_atom'][n]:.1f}" for p in pairs) + " |")
    (b3 / "b3_sp_results.json").write_text(json.dumps({"frames": frame_ids, "pairs": pairs}, indent=1) + "\n")


if __name__ == "__main__":
    main()
