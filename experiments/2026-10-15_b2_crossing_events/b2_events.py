"""Roadmap B2, step 1: locate real embedding crossings in the 2026-10-05 DhlA run (no ORCA).

The run (QM region A, r2SCAN-3c, cutoff 1.2 nm) wrote a DCD frame every 10 steps. This script
rebuilds the embedding exactly as ORCAPotential.createMixedSystem does (same path as B1), re-runs
CutoffEmbedding.select on x_0 (equilibrated.xml) and on all DCD frames, and lists for every pair of
consecutive frames (k, k+1) the groups that entered or left (S_k Δ S_{k+1}), with each group's net
charge and its minimum distance to the QM atoms on both frames.

Consistency with the run: steps.csv row n holds the selection made when the forces of step n were
evaluated, i.e. at the positions before step n (x_{n-1}). So |S(x_{10k})| must equal n_embed_groups
of row 10k + 1, and the changes between frames k and k + 1 must be among the embed_changed counts of
rows 10k + 2 ... 10k + 11. Both alignments (row 10k and row 10k + 1) are checked and reported.

It also writes the single-point job list for b2_singlepoints.py. For every interval with a change:
  base  (x_k, S_k), (x_{k+1}, S_{k+1})       every frame once
  swap  (x_k, S_{k+1}), (x_{k+1}, S_k)       set change at fixed geometry, both sides
  one   (x_k, S_k ⊕ g) for each crossing g    only when more than one group crossed
so that E(x_{k+1}, S_{k+1}) − E(x_k, S_k) splits exactly into a fixed-set geometry part and a
fixed-geometry set part.

Usage: b2_events.py RUN_DIR OUTDIR [--cutoff 1.2]
Writes OUTDIR/events.json, OUTDIR/frames.npz (positions/box of frames used by a job), OUTDIR/jobs.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import mdtraj
import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import enzyme_qmmm as enzyme  # noqa: E402

from openmmorca.qmmm import build_mixed_system  # noqa: E402
from openmmorca.qmmm.embedding import CutoffEmbedding, groups_from_topology  # noqa: E402
from openmmorca.qmmm.imaging import make_qm_whole, qm_bond_graph, minimum_image  # noqa: E402
from openmmorca.qmmm.linkatoms import make_boundary_pairs  # noqa: E402

DCD_INTERVAL = 10


def setup(run_dir, region="A"):
    pdb = app.PDBFile(str(run_dir / "prepared.pdb"))
    topology = pdb.topology
    state = mm.XmlSerializer.deserialize((run_dir / "equilibrated.xml").read_text())
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    qm, pairs_idx, *_ = enzyme.qm_selection(topology, region)
    pairs = make_boundary_pairs(topology, pairs_idx, None)
    parts = build_mixed_system(topology, system, qm, boundary_pairs=pairs)
    charges = np.zeros(system.getNumParticles())
    charges[list(parts.mm_atoms)] = parts.mm_charges_e
    groups = groups_from_topology(topology, parts.mm_atoms)
    graph = qm_bond_graph(topology, parts.qm_atoms, pairs)
    return topology, state, system, qm, pairs_idx, pairs, parts, charges, groups, graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--cutoff", type=float, default=1.2)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    run_json = json.loads((args.run_dir / "run.json").read_text())
    ident = run_json["identity"]
    assert ident["qm"]["cutoff_nm"] == args.cutoff and ident["recipe"]["dcd_interval"] == DCD_INTERVAL
    topology, state, system, qm, pairs_idx, pairs, parts, charges, groups, graph = setup(args.run_dir,
                                                                                        ident["qm"]["region"])
    assert qm == ident["qm"]["qm_atoms"] and [list(p) for p in pairs_idx] == ident["qm"]["boundary_pairs"]

    traj = mdtraj.load(str(args.run_dir / "trajectory.dcd"), top=str(args.run_dir / "prepared.pdb"))
    x0 = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box0 = np.array([v.value_in_unit(unit.nanometer) for v in state.getPeriodicBoxVectors()])
    positions = [np.asarray(x0, dtype=float)] + [traj.xyz[i].astype(float) for i in range(traj.n_frames)]
    boxes = [box0] + [traj.unitcell_vectors[i].astype(float) for i in range(traj.n_frames)]
    steps = [0] + [DCD_INTERVAL * (i + 1) for i in range(traj.n_frames)]

    residues = list(topology.residues())
    atom_res = {a.index: r for r in residues for a in r.atoms()}
    group_q = np.array([charges[list(g)].sum() for g in groups])
    group_label = [f"{atom_res[g[0]].name}{atom_res[g[0]].id}" for g in groups]
    atom_group = {i: k for k, g in enumerate(groups) for i in g}
    group_atoms = np.array([i for g in groups for i in g])
    group_of = np.repeat(np.arange(len(groups)), [len(g) for g in groups])

    emb = CutoffEmbedding(groups, charges, args.cutoff)
    sel, min_dist = [], []
    for pos, box in zip(positions, boxes):
        pos = make_qm_whole(pos, parts.qm_atoms, graph, box)
        qm_pos = pos[list(parts.qm_atoms)]
        atoms, _, _ = emb.select(qm_pos, pos, box)
        sel.append(sorted({atom_group[int(a)] for a in atoms}))
        # per-group minimum distance to any QM atom (same rule as select)
        near = qm_pos[0] + minimum_image(pos[group_atoms] - qm_pos[0], box)
        d = np.min(np.linalg.norm(near[:, None, :] - qm_pos[None, :, :], axis=-1), axis=1)
        gd = np.full(len(groups), np.inf)
        np.minimum.at(gd, group_of, d)
        min_dist.append(gd)

    rows = list(csv.DictReader(open(args.run_dir / "steps.csv")))
    n_rows = {int(r["step"]): int(r["n_embed_groups"]) for r in rows}
    changed_rows = {int(r["step"]): int(r["embed_changed"]) for r in rows}
    align = {}
    for offset in (0, 1):
        hits = [len(sel[k]) == n_rows[steps[k] + offset] for k in range(len(steps)) if steps[k] + offset in n_rows]
        align[f"row = frame step + {offset}"] = f"{sum(hits)}/{len(hits)} group counts match"

    events, used_frames, jobs = [], set(), []

    def job(kind, frame, set_from, groups_set, note=""):
        jobs.append({"id": len(jobs), "kind": kind, "frame": frame, "step": steps[frame], "set_from": set_from,
                     "groups": sorted(int(g) for g in groups_set), "note": note})
        used_frames.add(frame)

    based = set()
    for k in range(len(steps) - 1):
        a, b = set(sel[k]), set(sel[k + 1])
        diff = sorted(a ^ b)
        window = range(steps[k] + 2, steps[k + 1] + 2)
        csv_changes = sum(changed_rows.get(n, 0) for n in window)
        if not diff and csv_changes == 0:
            continue
        events.append({
            "k": k, "step_k": steps[k], "step_k1": steps[k + 1], "csv_changes_in_window": csv_changes,
            "crossed": [{"group": g, "label": group_label[g], "q": float(group_q[g]),
                         "direction": "in" if g in b else "out",
                         "d_k_nm": float(min_dist[k][g]), "d_k1_nm": float(min_dist[k + 1][g])} for g in diff],
        })
        if not diff:
            continue
        for f in (k, k + 1):
            if f not in based:
                job("base", f, f, sel[f])
                based.add(f)
        job("swap", k, k + 1, b, f"interval {k}")
        job("swap", k + 1, k, a, f"interval {k}")
        if len(diff) > 1:
            for g in diff:
                job("one", k, k, a ^ {g}, f"interval {k} group {g}")

    used = sorted(used_frames)
    np.savez_compressed(args.outdir / "frames.npz", frame=np.array(used), step=np.array([steps[f] for f in used]),
                        positions_nm=np.array([positions[f] for f in used]),
                        box_nm=np.array([boxes[f] for f in used]))
    report = {"run_dir": str(args.run_dir), "cutoff_nm": args.cutoff, "n_frames": len(steps),
              "frame_steps": "frame 0 = equilibrated.xml (step 0, float64); frame k >= 1 = DCD frame k-1 (step 10k, float32)",
              "alignment_check": align,
              "steps_csv_crossing_steps": sum(1 for r in rows if int(r["step"]) > 1 and int(r["embed_changed"]) > 0),
              "steps_csv_group_changes": sum(int(r["embed_changed"]) for r in rows if int(r["step"]) > 1),
              "n_intervals_with_change": sum(1 for e in events if e["crossed"]),
              "n_frame_pair_changes": sum(len(e["crossed"]) for e in events),
              "events": events}
    (args.outdir / "events.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.outdir / "jobs.json").write_text(json.dumps({"cutoff_nm": args.cutoff, "jobs": jobs}, indent=1) + "\n")

    print(f"alignment: {align}")
    print(f"steps.csv: {report['steps_csv_crossing_steps']} steps with crossings, "
          f"{report['steps_csv_group_changes']} group changes (step 1 excluded)")
    print(f"DCD frame pairs: {report['n_intervals_with_change']} intervals with a set change, "
          f"{report['n_frame_pair_changes']} net group changes")
    missed = [e for e in events if e["csv_changes_in_window"] != len(e["crossed"])]
    print(f"intervals where steps.csv count != net frame-pair change (back-and-forth inside 10 steps): {len(missed)}")
    qs = np.array([c["q"] for e in events for c in e["crossed"]])
    print(f"crossing groups: {np.sum(np.abs(qs) > 0.5)} charged, {np.sum(np.abs(qs) <= 0.5)} neutral; "
          f"labels: {sorted({c['label'] for e in events for c in e['crossed']})}")
    print(f"jobs: {len(jobs)} ({', '.join(f'{k} {sum(j['kind'] == k for j in jobs)}' for k in ('base', 'swap', 'one'))}),"
          f" frames used {len(used)}")


if __name__ == "__main__":
    main()
