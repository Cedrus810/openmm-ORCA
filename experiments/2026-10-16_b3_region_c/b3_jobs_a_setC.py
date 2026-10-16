"""Roadmap B3-1b: job list for region A embedded in the environment region C selects.

For each T09 frame, run the region C cutoff selection (1.2 nm) and take the residues of the selected
groups plus the residues that are QM in C but MM in A (Trp125, Trp175, His289 side chains). The job
embeds, in region A's grouping, every group whose residue is in that set. Region A with this set
differs from region C only in which atoms are treated by QM, so A(S_A) → A(S_C) isolates the
embedded-set change and A(S_C) → C(S_C) the QM-region change.

Output is in the b2_singlepoints.py format (kind "setC"); run it with
  b2_singlepoints.py jobs_a_setC.json T09/frames.npz T09/run_files sp_a_setC --region A

Usage: b3_jobs_a_setC.py T09_DIR OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-15_b2_crossing_events"))
from b2_events import setup  # noqa: E402

from openmmorca.qmmm.embedding import CutoffEmbedding  # noqa: E402
from openmmorca.qmmm.imaging import make_qm_whole  # noqa: E402

CUTOFF = 1.2


def main():
    t09, out = Path(sys.argv[1]), Path(sys.argv[2])
    run_files = t09 / "run_files"
    frames = np.load(t09 / "frames.npz")
    regions = {r: setup(run_files, r) for r in ("A", "C")}
    topology = regions["A"][0]
    atom_res = {a.index: a.residue.index for a in topology.atoms()}
    res_id = {r.index: int(r.id) for r in topology.residues()}

    def group_res(parts_groups):
        return [atom_res[g[0]] for g in parts_groups]

    _, _, _, qm_a, _, _, parts_a, charges_a, groups_a, graph_a = regions["A"]
    _, _, _, qm_c, _, _, parts_c, charges_c, groups_c, graph_c = regions["C"]
    res_a, res_c = group_res(groups_a), group_res(groups_c)
    # residues that are QM (side chain) in C but MM in A
    extra = {atom_res[i] for i in set(qm_c) - set(qm_a)}
    assert sorted(res_id[r] for r in extra) == [125, 175, 289], extra
    emb_c = CutoffEmbedding(groups_c, charges_c, CUTOFF)

    jobs = []
    for row, frame in enumerate(frames["frame"]):
        box = frames["box_nm"][row]
        pos = make_qm_whole(frames["positions_nm"][row], parts_c.qm_atoms, graph_c, box)
        atoms, _, _ = emb_c.select(pos[list(parts_c.qm_atoms)], pos, box)
        sel_res = {atom_res[int(a)] for a in atoms} | extra
        want = [k for k, r in enumerate(res_a) if r in sel_res]
        jobs.append({"id": row, "kind": "setC", "frame": int(frame), "step": int(frames["step"][row]),
                     "set_from": int(frame), "groups": want,
                     "note": f"region C selection {emb_c.last_n_groups} groups"})
        print(f"frame {int(frame)}: C selects {emb_c.last_n_groups} groups; A with C's residues: {len(want)} groups")
    out.write_text(json.dumps({"cutoff_nm": CUTOFF, "jobs": jobs}, indent=1) + "\n")


if __name__ == "__main__":
    main()
