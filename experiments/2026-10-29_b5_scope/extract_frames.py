"""B5: fixed snapshots every STRIDE_STEPS MD steps from an enzyme_qmmm.py run (all attempts, via run.json).

Usage: extract_frames.py RUN_DIR OUT.npz [--stride-steps 100]
DCDReporter interval 10: attempt frames are steps start_step + 10, + 20, ... (DCD stores float32).
"""
import argparse
import json
from pathlib import Path

import mdtraj
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--stride-steps", type=int, default=100)
args = parser.parse_args()
record = json.loads((args.run_dir / "run.json").read_text())
pos, box, steps = [], [], []
for attempt in record["attempts"]:
    if attempt["valid_through_step"] <= attempt["start_step"]:
        continue
    traj = mdtraj.load(str(args.run_dir / attempt["trajectory"]), top=str(args.run_dir / "prepared.pdb"))
    frame_steps = attempt["start_step"] + 10 * (np.arange(traj.n_frames) + 1)
    for i, s in enumerate(frame_steps):
        if s <= attempt["valid_through_step"] and s % args.stride_steps == 0:
            pos.append(traj.xyz[i].astype(np.float64))
            box.append(traj.unitcell_vectors[i].astype(np.float64))
            steps.append(int(s))
assert len(set(steps)) == len(steps)
np.savez_compressed(args.out, positions_nm=np.array(pos), box_nm=np.array(box), step=np.array(steps))
json.dump({"run_dir": str(args.run_dir), "stride_steps": args.stride_steps, "steps": steps,
           "note": "DCD stores float32 positions"}, open(args.out.with_suffix(".json"), "w"), indent=2)
print(f"{len(steps)} frames ({steps[0]}..{steps[-1]}) -> {args.out}")
