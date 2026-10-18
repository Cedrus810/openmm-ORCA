"""B3-2: which groups cross the 1.2 nm cutoff in a 1 ps run, by charge (same counting as B2).

Re-runs CutoffEmbedding.select on x_0 and every DCD frame (every 10 steps; all trajectory*.dcd files
of the run in step order) and counts the net group changes between consecutive frames. Changes that
reverse within 10 steps are not seen.

Usage: b3_md_crossings.py RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from pathlib import Path

import mdtraj
import numpy as np
from openmm import unit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-15_b2_crossing_events"))
from b2_events import setup  # noqa: E402

from openmmorca.qmmm.embedding import CutoffEmbedding  # noqa: E402
from openmmorca.qmmm.imaging import make_qm_whole  # noqa: E402

CHARGED = 0.5


def main():
    for run_dir in map(Path, sys.argv[1:]):
        record = json.loads((run_dir / "run.json").read_text())
        region = record["identity"]["qm"]["region"]
        topology, state, system, qm, pairs_idx, pairs, parts, charges, groups, graph = setup(run_dir, region)
        res = {a.index: a.residue for a in topology.atoms()}
        group_q = np.array([charges[list(g)].sum() for g in groups])
        label = [f"{res[g[0]].name}{res[g[0]].id}" for g in groups]
        emb = CutoffEmbedding(groups, charges, record["identity"]["qm"]["cutoff_nm"])
        atom_group = {i: k for k, g in enumerate(groups) for i in g}

        x0 = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box0 = np.array([v.value_in_unit(unit.nanometer) for v in state.getPeriodicBoxVectors()])
        frames = [(np.asarray(x0, float), box0)]
        for attempt in record["attempts"]:
            t = mdtraj.load(str(run_dir / attempt["trajectory"]), top=str(run_dir / "prepared.pdb"))
            frames += [(t.xyz[i].astype(float), t.unitcell_vectors[i].astype(float)) for i in range(t.n_frames)]

        sel = []
        for pos, box in frames:
            pos = make_qm_whole(pos, parts.qm_atoms, graph, box)
            atoms, _, _ = emb.select(pos[list(parts.qm_atoms)], pos, box)
            sel.append({atom_group[int(a)] for a in atoms})
        changes = [g for a, b in zip(sel, sel[1:]) for g in (a ^ b)]
        charged = [g for g in changes if abs(group_q[g]) > CHARGED]
        by_label = {}
        for g in charged:
            by_label[label[g]] = by_label.get(label[g], 0) + 1
        print(f"{run_dir.name} (region {region}): {len(frames)} frames, "
              f"{sum(1 for a, b in zip(sel, sel[1:]) if a ^ b)} intervals with a change, "
              f"{len(changes)} net group changes, {len(charged)} charged; "
              f"charged by residue {dict(sorted(by_label.items()))}; mean groups {np.mean([len(s) for s in sel]):.0f}")


if __name__ == "__main__":
    main()
