"""Roadmap B4 stage 1: classical comparison of embedding schemes (no ORCA).

QM region A stood in for by its force-field charges plus the remainder on the link H (as in B1/B2).
For every frame of the 2026-10-05 run (x_0 + 200 DCD frames) the electrostatic potential and field at
the 15 QM sites are computed for

  S0   groups with any atom within 1.2 nm of a QM atom (current)
  S1   same, 2.5 nm
  S2   every group, placed at its minimum image relative to the QM region (no cutoff)
  S5   every charged group (|q| > 0.5 e) as in S2, neutral groups as in S0

Group placement is the one CutoffEmbedding.select uses (whole group, at the image of its atom nearest
the QM region), applied without a cutoff. Reference: periodic Ewald sum over all embeddable MM charges
(minimum-image real space; reciprocal sum on the full k grid via separable phase factors; uniform
background for the +1 e net MM charge, which only shifts the potential by a constant). The Ewald sum is
checked against itself with two (alpha, kmax) pairs on a few frames.

Accuracy (vs Ewald, all frames): RMS over sites/frames of the field error times the site charge
(a force on the stand-in charges, kJ/mol/nm); the bond-axis field at the C1–Cl1 midpoint,
u·(E_C1 + E_Cl1)/2 (kJ/mol/nm per e); frame-to-frame std of the interaction-energy error
U = Σ q_i φ_i (kJ/mol; the constant offset is not meaningful).

Continuity (consecutive frames k → k+1, at the geometry of k+1): the jump caused by updating the
inclusion and image choice from frame k to frame k+1. The "old" choice keeps each group's frame-k
inclusion and the image nearest its frame-k placement. Reported: number of events, |ΔU|, max site
force jump, bond-field jump.

Usage: b4_stage1_classical.py RUN_DIR OUTDIR
"""
import json
import sys
import time
from pathlib import Path

import mdtraj
import numpy as np
from openmm import unit
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-15_b2_crossing_events"))
from b2_events import setup  # noqa: E402

from openmmorca.qmmm.imaging import make_qm_whole, minimum_image  # noqa: E402
from openmmorca.qmmm.linkatoms import LinkAtomManager  # noqa: E402
from openmmorca.units import COULOMB_KJ_MOL_NM as KE  # noqa: E402

CHARGED = 0.5
SCHEMES = ("S0", "S1", "S2", "S5")


def placement(qm_pos, pos_all, atoms, group_of, starts, box):
    """Whole-group minimum-image placement relative to the QM region, and per-group min distance."""
    anchor = qm_pos[0]
    p = pos_all[atoms]
    near = anchor + minimum_image(p - anchor, box)
    d = np.min(np.linalg.norm(near[:, None, :] - qm_pos[None, :, :], axis=-1), axis=1)
    internal = minimum_image(p - p[starts[group_of]], box)
    n_groups = len(starts)
    best = np.full(n_groups, -1)
    gd = np.full(n_groups, np.inf)
    order = np.argsort(d)[::-1]  # so that the smallest distance is written last
    best[group_of[order]] = order
    np.minimum.at(gd, group_of, d)
    placed = near[best[group_of]] + internal - internal[best[group_of]]
    return placed, gd


def coulomb(sites, pos, q):
    d = sites[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    phi = KE * (q[None, :] / r).sum(1)
    field = KE * ((q[None, :] / r ** 3)[:, :, None] * d).sum(1)
    return phi, field


def ewald(sites, pos, q, box, alpha, nmax):
    """Potential and field at sites from periodic point charges (cubic box), kJ/mol/e and kJ/mol/nm/e."""
    L = float(box[0, 0])
    assert np.allclose(box, np.diag([L, L, L]))
    V = L ** 3
    d = minimum_image(sites[:, None, :] - pos[None, :, :], box)
    r = np.linalg.norm(d, axis=-1)
    phi = (q[None, :] * erfc(alpha * r) / r).sum(1)
    g = erfc(alpha * r) / r ** 2 + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * r) ** 2) / r
    field = ((q[None, :] * g / r)[:, :, None] * d).sum(1)
    n = np.arange(-nmax, nmax + 1)
    kk = 2 * np.pi * n / L
    ph = [np.exp(-1j * np.outer(pos[:, a], kk)) for a in range(3)]  # (N, 2n+1)
    S = np.empty((len(n),) * 3, dtype=complex)
    for ia in range(len(n)):
        w = q * ph[0][:, ia]
        S[ia] = (ph[1] * w[:, None]).T @ ph[2]
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    f = np.zeros_like(k2)
    nz = k2 > 0
    f[nz] = np.exp(-k2[nz] / (4 * alpha ** 2)) / k2[nz]
    pref = 4 * np.pi / V
    for i, s in enumerate(sites):
        e = np.exp(1j * (KX * s[0] + KY * s[1] + KZ * s[2])) * S
        phi[i] += pref * (f * e.real).sum()
        field[i] += pref * np.array([(f * KX * e.imag).sum(), (f * KY * e.imag).sum(), (f * KZ * e.imag).sum()])
    phi -= np.pi * q.sum() / (V * alpha ** 2)
    return KE * phi, KE * field


def main():
    run_dir, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    topology, state, system, qm, pairs_idx, pairs, parts, charges, groups, graph = setup(run_dir, "A")
    links = LinkAtomManager(pairs)
    pair0 = parts.boundary_pairs[0]
    qm_atoms = list(parts.qm_atoms)
    names = [a.name for a in topology.atoms()]
    k_c1 = qm_atoms.index(next(i for i in qm_atoms if names[i] == "C1"))
    k_cl1 = qm_atoms.index(next(i for i in qm_atoms if names[i] == "CL1"))
    qm_q_ff = np.array([parts.removed_qm_charges[i] for i in qm_atoms])
    qm_q = np.concatenate([qm_q_ff, np.full(links.n_links, (parts.qm_formal_charge - qm_q_ff.sum()) / links.n_links)])

    atoms = np.array([i for g in groups for i in g])
    group_of = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    starts = np.concatenate([[0], np.cumsum([len(g) for g in groups])[:-1]]).astype(int)
    q_atoms = charges[atoms]
    group_q = np.array([charges[list(g)].sum() for g in groups])
    charged = np.abs(group_q) > CHARGED

    traj = mdtraj.load(str(run_dir / "trajectory.dcd"), top=str(run_dir / "prepared.pdb"))
    x0 = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box0 = np.array([v.value_in_unit(unit.nanometer) for v in state.getPeriodicBoxVectors()])
    frames = [(np.asarray(x0, float), box0)] + [(traj.xyz[i].astype(float), traj.unitcell_vectors[i].astype(float))
                                                 for i in range(traj.n_frames)]

    def include(gd):
        return {"S0": gd < 1.2, "S1": gd < 2.5, "S2": np.ones(len(gd), bool), "S5": charged | (gd < 1.2)}

    def evaluate(sites, placed, inc):
        mask = inc[group_of]
        return coulomb(sites, placed[mask], q_atoms[mask])

    def summarize(phi, field, sites):
        u = sites[k_c1] - sites[k_cl1]
        u /= np.linalg.norm(u)
        return {"U": float((qm_q * phi).sum()), "F": qm_q[:, None] * field,
                "bond": float(u @ (field[k_c1] + field[k_cl1]) / 2)}

    t0 = time.perf_counter()
    acc = {s: {"dU": [], "dF": [], "dbond": []} for s in SCHEMES}
    jumps = {s: [] for s in SCHEMES}
    ewald_check = []
    prev = None
    for k, (pos_raw, box) in enumerate(frames):
        pos = make_qm_whole(pos_raw, parts.qm_atoms, graph, box)
        qm_pos = pos[qm_atoms]
        sites = np.vstack([qm_pos, links.link_positions(pos)])
        placed, gd = placement(qm_pos, pos, atoms, group_of, starts, box)
        inc = include(gd)
        phi_e, field_e = ewald(sites, pos[atoms], q_atoms, box, 1.2, 14)
        ref = summarize(phi_e, field_e, sites)
        if k in (0, 100, 200):
            phi_2, field_2 = ewald(sites, pos[atoms], q_atoms, box, 1.6, 20)
            ewald_check.append({"frame": k, "max_dphi": float(np.abs(phi_e - phi_2 - (phi_e - phi_2).mean()).max()),
                                "max_dfield": float(np.abs(field_e - field_2).max()),
                                "field_scale": float(np.abs(field_e).max())})
        for s in SCHEMES:
            cur = summarize(*evaluate(sites, placed, inc[s]), sites)
            acc[s]["dU"].append(cur["U"] - ref["U"])
            acc[s]["dF"].append(cur["F"] - ref["F"])
            acc[s]["dbond"].append(cur["bond"] - ref["bond"])
            if prev is not None:
                # old choice: frame-k inclusion, image nearest the frame-k placement
                # relative to the QM anchor: frames may be wrapped differently (x_0 from equilibrated.xml)
                shift = np.round(((prev["placed"] - prev["anchor"]) - (placed - sites[0])) / box[0, 0]) * box[0, 0]
                gshift = shift[starts]  # whole groups move together
                old_placed = placed + gshift[group_of]
                old = summarize(*evaluate(sites, old_placed, prev["inc"][s]), sites)
                n_changed = int((prev["inc"][s] != inc[s]).sum() + ((np.abs(gshift).sum(1) > 0) & inc[s] & prev["inc"][s]).sum())
                if n_changed:
                    inc_changed = np.nonzero(prev["inc"][s] != inc[s])[0]
                    img_changed = np.nonzero((np.abs(gshift).sum(1) > 0) & inc[s] & prev["inc"][s])[0]
                    jumps[s].append({"k": k, "n": n_changed, "inc_changed": inc_changed.tolist(),
                                     "img_changed": img_changed.tolist(), "dU": cur["U"] - old["U"],
                                     "dF_max": float(np.linalg.norm(cur["F"] - old["F"], axis=1).max()),
                                     "dbond": cur["bond"] - old["bond"]})
        prev = {"placed": placed, "anchor": sites[0], "inc": inc}
        if k % 50 == 0:
            print(f"frame {k}: {time.perf_counter() - t0:.0f} s", flush=True)

    print(f"\nEwald self-check (alpha 1.2/n14 vs 1.6/n20): {ewald_check}")
    n_ps = (len(frames) - 1) * 10 * 0.0005
    print(f"\n### Accuracy vs Ewald ({len(frames)} frames)\n")
    print("| scheme | mean groups | site force error RMS (kJ/mol/nm) | max | C1–Cl1 bond-axis field error RMS (kJ/mol/nm/e) | mean | U error frame-to-frame std (kJ/mol) |")
    print("|---|---|---|---|---|---|---|")
    out = {"ewald_check": ewald_check, "schemes": {}}
    for s in SCHEMES:
        dF = np.stack(acc[s]["dF"])
        db = np.array(acc[s]["dbond"])
        dU = np.array(acc[s]["dU"])
        row = {"site_F_err_rms": float(np.sqrt((dF ** 2).sum(-1).mean())),
               "site_F_err_max": float(np.linalg.norm(dF, axis=-1).max()),
               "bond_field_err_rms": float(np.sqrt((db ** 2).mean())), "bond_field_err_mean": float(db.mean()),
               "U_err_frame_std": float(dU.std())}
        js = jumps[s]
        row["jumps"] = {"intervals": len(js), "per_ps": len(js) / n_ps, "n_changes": int(sum(j["n"] for j in js))}
        if js:
            for key in ("dU", "dF_max", "dbond"):
                a = np.abs([j[key] for j in js])
                row["jumps"][key] = {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}
        out["schemes"][s] = row
        print(f"| {s} | — | {row['site_F_err_rms']:.1f} | {row['site_F_err_max']:.1f} | {row['bond_field_err_rms']:.1f} | "
              f"{row['bond_field_err_mean']:+.1f} | {row['U_err_frame_std']:.1f} |")
    print(f"\n### Jumps between consecutive frames (fixed geometry; {n_ps:.1f} ps)\n")
    print("| scheme | intervals with a change | group changes | |ΔU| median / p90 / max (kJ/mol) | site force jump max: median / p90 / max | |Δ bond field| median / p90 / max |")
    print("|---|---|---|---|---|---|")
    for s in SCHEMES:
        j = out["schemes"][s]["jumps"]
        if j["intervals"] == 0:
            print(f"| {s} | 0 | 0 | — | — | — |")
            continue
        fmt = lambda d: f"{d['median']:.2f} / {d['p90']:.2f} / {d['max']:.2f}"  # noqa: E731
        print(f"| {s} | {j['intervals']} | {j['n_changes']} | {fmt(j['dU'])} | {fmt(j['dF_max'])} | {fmt(j['dbond'])} |")
    out["jump_lists"] = jumps
    (outdir / "b4_stage1.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
