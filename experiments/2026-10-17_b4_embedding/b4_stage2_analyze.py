"""Roadmap B4: QM single points of S0 (1.2 nm), S1 (2.5 nm) and S2 (whole box) on the 20 T09 frames,
compared with the classical model of stage 1 (QM region A stood in for by force-field charges).

QM data: S0, S1 from the T09 scan (scan/c1.200.npz, c2.500.npz; 183, np16); S2 from s2_t09/ (181, np18).
For X in {S0, S1}:
  QM:        ΔE_X = E_X − E_S2,   Δs_X = stretch_X − stretch_S2   (stretch = 0.5·u·(F_C1 − F_Cl1))
  classical: ΔU_X = U_X − U_S2,   Δb_X = bond-axis field_X − field_S2
The frame-to-frame std of ΔE_X − ΔU_X tests whether the classical model also explains S0/S1 → S2.
β = least-squares slope of Δs on Δb (S0 and S1 pooled) is the QM stretch response per unit classical
bond-axis field. The remaining S2 error is then estimated from the classical S2 → Ewald difference:
energy by the frame std of U_Ewald − U_S2, stretch by β·(b_Ewald − b_S2). This assumes the linear
response measured between S0/S1 and S2 also holds beyond S2; it is an estimate, not a measurement.

Usage: b4_stage2_analyze.py T09_DIR S2_DIR RUN_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b4_stage1_classical import coulomb, ewald, placement, setup  # noqa: E402

from openmmorca.qmmm.imaging import make_qm_whole  # noqa: E402
from openmmorca.qmmm.linkatoms import LinkAtomManager  # noqa: E402


def main():
    t09, s2dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    frames = np.load(t09 / "frames.npz")
    topology, state, system, qm, pairs_idx, pairs, parts, charges, groups, graph = setup(run_dir, "A")
    qm_atoms = list(parts.qm_atoms)
    names = [a.name for a in topology.atoms()]
    k_c1 = qm_atoms.index(next(i for i in qm_atoms if names[i] == "C1"))
    k_cl1 = qm_atoms.index(next(i for i in qm_atoms if names[i] == "CL1"))
    links = LinkAtomManager(pairs)
    qm_q_ff = np.array([parts.removed_qm_charges[i] for i in qm_atoms])
    qm_q = np.concatenate([qm_q_ff, np.full(links.n_links, (parts.qm_formal_charge - qm_q_ff.sum()) / links.n_links)])
    atoms = np.array([i for g in groups for i in g])
    group_of = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    starts = np.concatenate([[0], np.cumsum([len(g) for g in groups])[:-1]]).astype(int)
    q_atoms = charges[atoms]

    s2 = {r["frame"]: r for r in json.loads((s2dir / "stage2.json").read_text()) if r["scheme"] == "S2"}
    s2f = np.load(s2dir / "qm_forces.npz")
    assert list(s2f["qm_atoms"]) == qm_atoms

    def scan(c):
        with np.load(t09 / "scan" / f"c{c}.npz") as d:
            return {int(f): (float(d[f"energy_kj_mol_{f}"]), d[f"forces_{f}"][qm_atoms]) for f in d["frames_done"]}

    qmE = {"S0": scan("1.200"), "S1": scan("2.500")}
    qmE["S2"] = {f: (s2[f]["energy_kj_mol"], s2f[f"S2_f{f}"]) for f in s2}
    fids = sorted(set(qmE["S0"]) & set(qmE["S1"]) & set(qmE["S2"]))

    rows = []
    for f in fids:
        box = frames["box_nm"][f]
        pos = make_qm_whole(frames["positions_nm"][f], parts.qm_atoms, graph, box)
        qm_pos = pos[qm_atoms]
        sites = np.vstack([qm_pos, links.link_positions(pos)])
        u = qm_pos[k_c1] - qm_pos[k_cl1]
        u /= np.linalg.norm(u)
        placed, gd = placement(qm_pos, pos, atoms, group_of, starts, box)
        inc = {"S0": gd < 1.2, "S1": gd < 2.5, "S2": np.ones(len(gd), bool)}
        cl = {}
        for s, m in inc.items():
            mask = m[group_of]
            phi, field = coulomb(sites, placed[mask], q_atoms[mask])
            cl[s] = ((qm_q * phi).sum(), u @ (field[k_c1] + field[k_cl1]) / 2)
        phi, field = ewald(sites, pos[atoms], q_atoms, box, 1.2, 14)
        cl["Ewald"] = ((qm_q * phi).sum(), u @ (field[k_c1] + field[k_cl1]) / 2)
        st = {s: 0.5 * float(u @ (qmE[s][f][1][k_c1] - qmE[s][f][1][k_cl1])) for s in ("S0", "S1", "S2")}
        rows.append({"frame": f, "E": {s: qmE[s][f][0] for s in ("S0", "S1", "S2")}, "stretch": st,
                     "U": {s: float(v[0]) for s, v in cl.items()}, "b": {s: float(v[1]) for s, v in cl.items()}})

    def arr(key, s):
        return np.array([r[key][s] for r in rows])

    print(f"frames: {len(rows)}\n")
    print("| X | QM ΔE_X frame std (kJ/mol) | classical ΔU_X frame std | std(ΔE − ΔU) | corr | QM Δstretch RMS (mean) (kJ/mol/nm) | classical Δ bond field RMS (mean) |")
    print("|---|---|---|---|---|---|---|")
    ds_all, db_all = [], []
    for X in ("S0", "S1"):
        dE = arr("E", X) - arr("E", "S2")
        dU = arr("U", X) - arr("U", "S2")
        ds = arr("stretch", X) - arr("stretch", "S2")
        db = arr("b", X) - arr("b", "S2")
        ds_all.append(ds)
        db_all.append(db)
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
        print(f"| {X} | {dE.std():.1f} | {dU.std():.1f} | {(dE - dU).std():.1f} | {np.corrcoef(dE, dU)[0, 1]:.3f} | "
              f"{rms(ds):.1f} ({ds.mean():+.1f}) | {rms(db):.1f} ({db.mean():+.1f}) |")
    ds_all, db_all = np.concatenate(ds_all), np.concatenate(db_all)
    beta = float((ds_all * db_all).sum() / (db_all ** 2).sum())
    resid = ds_all - beta * db_all
    print(f"\nβ (QM stretch per classical bond-axis field, S0/S1 → S2 pooled): {beta:.2f}; "
          f"corr {np.corrcoef(ds_all, db_all)[0, 1]:+.3f}; residual RMS {np.sqrt(np.mean(resid ** 2)):.1f} kJ/mol/nm")
    dU_rem = arr("U", "Ewald") - arr("U", "S2")
    db_rem = arr("b", "Ewald") - arr("b", "S2")
    print(f"remaining S2 error (estimate): energy frame std {dU_rem.std():.1f} kJ/mol; "
          f"bond-axis field {np.sqrt(np.mean(db_rem ** 2)):.1f} (mean {db_rem.mean():+.1f}); "
          f"stretch β·Δb RMS {abs(beta) * np.sqrt(np.mean(db_rem ** 2)):.1f} (mean {beta * db_rem.mean():+.1f}) kJ/mol/nm")
    t = [s2[f]["t_orca"] for f in fids]
    print(f"S2 t_orca: mean {np.mean(t):.1f} s, range {min(t):.1f}–{max(t):.1f} s (181, np18)")
    out = {"rows": rows, "beta": beta}
    (s2dir / "b4_stage2_analysis.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
