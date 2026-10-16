"""T09: copy fixed snapshots of the 2026-10-05 DhlA run (kasuga01 local disk) to frames.npz.

Usage: extract_frames.py RUN_DIR OUT.npz STRIDE
DCDReporter interval 10: frame i (0-based) is MD step 10 * (i + 1).
"""
import json
import sys
from pathlib import Path

import mdtraj
import numpy as np

run, out, stride = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
traj = mdtraj.load(str(run / "trajectory.dcd"), top=str(run / "prepared.pdb"))
picked = np.arange(stride - 1, traj.n_frames, stride)
np.savez_compressed(out, positions_nm=traj.xyz[picked].astype(np.float64),
                    box_nm=traj.unitcell_vectors[picked].astype(np.float64),
                    frame=picked, step=10 * (picked + 1))
json.dump({"run_dir": str(run), "n_frames_total": traj.n_frames, "stride": stride,
           "frames": picked.tolist(), "steps": (10 * (picked + 1)).tolist(),
           "note": "DCD stores float32 positions"}, open(out.with_suffix(".json"), "w"), indent=2)
print(f"{len(picked)} frames -> {out}")
