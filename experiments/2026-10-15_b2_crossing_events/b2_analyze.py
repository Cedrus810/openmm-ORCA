"""Roadmap B2, step 3: size of real embedding crossings, split from the geometry change.

Needs events.json, jobs.json, frames.npz (b2_events.py) and sp/sp_w*.npz (b2_singlepoints.py).

For every interval (k, k+1) of consecutive frames (10 MD steps = 5 fs) with a set change:
  set part at fixed geometry   x_k:     E(x_k, S_{k+1}) − E(x_k, S_k)
                                x_{k+1}: E(x_{k+1}, S_{k+1}) − E(x_{k+1}, S_k)
  geometry part at fixed set   S_k:     E(x_{k+1}, S_k) − E(x_k, S_k)
                                S_{k+1}: E(x_{k+1}, S_{k+1}) − E(x_k, S_{k+1})
and the same for forces on the QM atoms. Single-group toggles (one group crossing, or each group of a
multi-group interval) are compared with the classical model of B1: the QM region stood in for by its
force-field charges plus the remainder on the link H, interacting with the group's charges.

The force a crossing group itself feels jumps from 0 to the QM-region force on it; that is the MM side
of the discontinuity and is reported from the job in which the group is embedded.

Replicate noise: smoke/ (same jobs, run earlier in another process) vs sp/.

Usage: b2_analyze.py B2_DIR RUN_DIR
Writes B2_DIR/b2_results.json and prints Markdown tables.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_events import setup  # noqa: E402

from openmmorca.qmmm.embedding import CutoffEmbedding  # noqa: E402
from openmmorca.qmmm.imaging import make_qm_whole  # noqa: E402
from openmmorca.qmmm.linkatoms import LinkAtomManager  # noqa: E402
from openmmorca.units import COULOMB_KJ_MOL_NM  # noqa: E402

CHARGED = 0.5  # e; groups are whole residues/ions/waters, net charges are integers


def load_sp(directory):
    out = {}
    for p in sorted(directory.glob("sp_w*.npz")):
        if p.stem.endswith(".tmp"):
            continue
        with np.load(p) as d:
            for j in d["jobs_done"]:
                j = int(j)
                atoms = d[f"force_atoms_{j}"]
                out[j] = {"E": float(d[f"energy_kj_mol_{j}"]),
                          "F": dict(zip(atoms.tolist(), d[f"forces_{j}"])),
                          "t_orca": float(d[f"t_orca_{j}"])}
    return out


def coulomb(qm_pos, qm_q, pos, q):
    d = qm_pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    pair = COULOMB_KJ_MOL_NM * qm_q[:, None] * q[None, :] / r
    return float(pair.sum()), ((pair / r ** 2)[:, :, None] * d).sum(axis=1)


def rms_atoms(x):
    return float(np.sqrt((np.asarray(x) ** 2).sum(-1).mean()))


def main():
    b2, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    events = json.loads((b2 / "events.json").read_text())
    jobs = json.loads((b2 / "jobs.json").read_text())["jobs"]
    frames = np.load(b2 / "frames.npz")
    row_of = {int(f): i for i, f in enumerate(frames["frame"])}
    sp = load_sp(b2 / "sp")
    missing = [j["id"] for j in jobs if j["id"] not in sp]

    topology, state, system, qm, pairs_idx, pairs, parts, charges, groups, graph = setup(run_dir)
    qm_atoms = list(parts.qm_atoms)
    atoms = list(topology.atoms())
    names = [f"{atoms[i].residue.name}{atoms[i].residue.id}:{atoms[i].name}" for i in qm_atoms]
    k_c1, k_cl1 = names.index("DCE600:C1"), names.index("DCE600:CL1")
    links = LinkAtomManager(pairs)
    pair0 = parts.boundary_pairs[0]
    q1_row = qm_atoms.index(pair0.q1)
    qm_q_ff = np.array([parts.removed_qm_charges[i] for i in qm_atoms])
    qm_q = np.concatenate([qm_q_ff, np.full(links.n_links, (parts.qm_formal_charge - qm_q_ff.sum()) / links.n_links)])
    group_q = np.array([charges[list(g)].sum() for g in groups])
    reach = CutoffEmbedding(groups, charges, 2.0)
    atom_group = {i: k for k, g in enumerate(groups) for i in g}

    geo_cache = {}

    def geometry(frame):
        if frame not in geo_cache:
            r = row_of[frame]
            box = frames["box_nm"][r]
            pos = make_qm_whole(frames["positions_nm"][r], parts.qm_atoms, graph, box)
            sites = np.vstack([pos[qm_atoms], links.link_positions(pos)])
            sel_atoms, placed, _ = reach.select(pos[qm_atoms], pos, box)
            by_group = {}
            for a, p in zip(sel_atoms.tolist(), placed):
                by_group.setdefault(atom_group[a], []).append((a, p))
            u = pos[qm_atoms[k_c1]] - pos[qm_atoms[k_cl1]]
            geo_cache[frame] = (sites, by_group, u / np.linalg.norm(u))
        return geo_cache[frame]

    def classical(frame, g):
        """Classical energy and QM-row forces of group g with the QM stand-in at this frame."""
        sites, by_group, _ = geometry(frame)
        ids = [a for a, _ in by_group[g]]
        e, fs = coulomb(sites, qm_q, np.array([p for _, p in by_group[g]]), charges[ids])
        fq = fs[: len(qm_atoms)].copy()
        fq[q1_row] += (1.0 - pair0.g) * fs[len(qm_atoms)]
        return e, fq

    def qmF(j):
        return np.array([sp[j]["F"].get(i, np.zeros(3)) for i in qm_atoms])

    def stretch(frame, f):
        u = geometry(frame)[2]
        return 0.5 * float(u @ (f[k_c1] - f[k_cl1]))

    by_key = {(j["kind"], j["frame"], j["set_from"], tuple(j["groups"])): j["id"] for j in jobs}
    base = {j["frame"]: j["id"] for j in jobs if j["kind"] == "base"}

    intervals, singles = [], []
    for ev in events["events"]:
        if not ev["crossed"]:
            continue
        k = ev["k"]
        swap_k = next(j["id"] for j in jobs if j["kind"] == "swap" and j["frame"] == k and j["set_from"] == k + 1)
        swap_k1 = next(j["id"] for j in jobs if j["kind"] == "swap" and j["frame"] == k + 1 and j["set_from"] == k)
        ids = (base[k], base[k + 1], swap_k, swap_k1)
        if any(i not in sp for i in ids):
            continue
        b0, b1, s0, s1 = ids  # (x_k,S_k), (x_k1,S_k1), (x_k,S_k1), (x_k1,S_k)
        E = {i: sp[i]["E"] for i in ids}
        F = {i: qmF(i) for i in ids}
        crossed = ev["crossed"]
        charged = any(abs(c["q"]) > CHARGED for c in crossed)
        intervals.append({
            "k": k, "n_crossed": len(crossed), "charged": charged,
            "labels": [f"{c['label']}{'+' if c['direction'] == 'in' else '-'}" for c in crossed],
            "dE_set_xk": E[s0] - E[b0], "dE_set_xk1": E[b1] - E[s1],
            "dE_geo_Sk": E[s1] - E[b0], "dE_geo_Sk1": E[b1] - E[s0],
            "dF_set_xk": F[s0] - F[b0], "dF_set_xk1": F[b1] - F[s1],
            "dF_geo_Sk": F[s1] - F[b0],
        })
        # single-group toggles at x_k
        for c in crossed:
            g = c["group"]
            if len(crossed) == 1:
                j_with, j_without = (s0, b0) if c["direction"] == "in" else (b0, s0)
            else:
                base_set = set(next(j for j in jobs if j["id"] == b0)["groups"])
                j_tog = by_key.get(("one", k, k, tuple(sorted(base_set ^ {g}))))
                if j_tog is None or j_tog not in sp:
                    continue
                j_with, j_without = (j_tog, b0) if c["direction"] == "in" else (b0, j_tog)
            dE = sp[j_with]["E"] - sp[j_without]["E"]
            dF = qmF(j_with) - qmF(j_without)
            e_cl, f_cl = classical(k, g)
            grp_atoms = list(groups[g])
            f_grp = np.array([sp[j_with]["F"].get(a, np.zeros(3)) for a in grp_atoms])
            singles.append({
                "k": k, "group": g, "label": c["label"], "q": c["q"], "charged": abs(c["q"]) > CHARGED,
                "d_nm": c["d_k_nm"], "dE": dE, "dE_cl": e_cl,
                "dF_rms": rms_atoms(dF), "dF_max": float(np.linalg.norm(dF, axis=1).max()),
                "dF_resid_rms": rms_atoms(dF - f_cl), "dF_cl_rms": rms_atoms(f_cl),
                "stretch": stretch(k, dF), "stretch_cl": stretch(k, f_cl),
                "field_cl1": float(geometry(k)[2] @ f_cl[k_cl1]),
                "F_group_max": float(np.linalg.norm(f_grp, axis=1).max()),
            })

    # replicate noise
    noise = []
    smoke = load_sp(b2 / "smoke") if (b2 / "smoke").is_dir() else {}
    for j, r in smoke.items():
        if j in sp:
            fa = np.array([r["F"].get(i, np.zeros(3)) for i in qm_atoms])
            noise.append({"job": j, "dE": abs(r["E"] - sp[j]["E"]), "dF_max": float(np.linalg.norm(fa - qmF(j), axis=1).max())})

    def stats(x):
        x = np.abs(np.asarray(x, dtype=float))
        if len(x) == 0:
            return "—"
        return f"{np.median(x):.2f} / {np.percentile(x, 90):.2f} / {x.max():.2f}"

    print(f"jobs done {len(sp)}/{len(jobs)}; missing {len(missing)}")
    print(f"replicate noise (smoke vs sp): {noise}")
    t = [r["t_orca"] for r in sp.values()]
    print(f"t_orca mean {np.mean(t):.1f} s, median {np.median(t):.1f} s\n")

    print("### Intervals (10 steps = 5 fs) with a set change; |value| median / 90th pct / max\n")
    print("| intervals | n | ΔE set @x_k (kJ/mol) | ΔE geometry @S_k (kJ/mol) | QM ΔF set RMS | QM ΔF set max "
          "| QM ΔF geometry RMS (10 steps) | set part path dependence |ΔE_set(x_k)−ΔE_set(x_k+1)| |")
    print("|---|---|---|---|---|---|---|---|")
    for name, sub in (("all", intervals), ("with a charged group", [r for r in intervals if r["charged"]]),
                      ("neutral only", [r for r in intervals if not r["charged"]])):
        print(f"| {name} | {len(sub)} | {stats([r['dE_set_xk'] for r in sub])} | "
              f"{stats([r['dE_geo_Sk'] for r in sub])} | {stats([rms_atoms(r['dF_set_xk']) for r in sub])} | "
              f"{stats([np.linalg.norm(r['dF_set_xk'], axis=1).max() for r in sub])} | "
              f"{stats([rms_atoms(r['dF_geo_Sk']) for r in sub])} | "
              f"{stats([r['dE_set_xk'] - r['dE_set_xk1'] for r in sub])} |")
    total_jump = float(np.sum([r["dE_set_xk"] for r in intervals]))
    print(f"\nsum of set jumps over the run: {total_jump:+.1f} kJ/mol "
          f"(RMS per interval {np.sqrt(np.mean(np.square([r['dE_set_xk'] for r in intervals]))):.2f})")

    print("\n### Single-group toggles at fixed geometry (x_k); |value| median / 90th pct / max\n")
    print("| groups | n | ΔE (kJ/mol) | ΔE − classical | QM ΔF RMS | QM ΔF max | QM ΔF − classical RMS "
          "| C1–Cl1 stretch ΔF | force on the group itself max |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, sub in (("charged", [s for s in singles if s["charged"]]),
                      ("neutral", [s for s in singles if not s["charged"]])):
        print(f"| {name} | {len(sub)} | {stats([s['dE'] for s in sub])} | {stats([s['dE'] - s['dE_cl'] for s in sub])} | "
              f"{stats([s['dF_rms'] for s in sub])} | {stats([s['dF_max'] for s in sub])} | "
              f"{stats([s['dF_resid_rms'] for s in sub])} | {stats([s['stretch'] for s in sub])} | "
              f"{stats([s['F_group_max'] for s in sub])} |")
    for name, sub in (("charged", [s for s in singles if s["charged"]]),
                      ("neutral", [s for s in singles if not s["charged"]]), ("all", singles)):
        if len(sub) > 2:
            ce = np.corrcoef([s["dE"] for s in sub], [s["dE_cl"] for s in sub])[0, 1]
            cs = np.corrcoef([s["stretch"] for s in sub], [s["field_cl1"] for s in sub])[0, 1]
            print(f"{name}: corr(ΔE, classical) {ce:+.3f}; corr(C1–Cl1 stretch ΔF, classical bond field at Cl1) {cs:+.3f}")

    print("\n### Charged-group toggles\n")
    print("| interval k | group | q | d (nm) | ΔE | classical | QM ΔF RMS | max | stretch | group force max |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted((s for s in singles if s["charged"]), key=lambda s: s["k"]):
        print(f"| {s['k']} | {s['label']} | {s['q']:+.0f} | {s['d_nm']:.3f} | {s['dE']:+.2f} | {s['dE_cl']:+.2f} | "
              f"{s['dF_rms']:.1f} | {s['dF_max']:.1f} | {s['stretch']:+.1f} | {s['F_group_max']:.1f} |")

    out = {"missing_jobs": missing, "noise": noise, "sum_set_jumps_kj_mol": total_jump,
           "intervals": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in intervals],
           "singles": singles}
    (b2 / "b2_results.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
