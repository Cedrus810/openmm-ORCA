"""B4 S2 implementation check: new vectorized CutoffEmbedding.select vs the pre-change loop version.

Both are run on the T09 frames (DhlA 2DHC, regions A and C) for cutoff 1.2 nm (S0) and the whole box
(S2; the old code reproduces S2 as the B4 prototype did: cutoff 100 nm with the box check disabled).
Reports exact equality of selected atoms / positions / charges, and select() wall time per call.

Usage: check_placement.py T09_DIR [--regions A C]
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "experiments" / "2026-10-15_b2_crossing_events"))
from b2_events import setup  # noqa: E402

from openmmorca.qmmm.embedding import CutoffEmbedding  # noqa: E402
from openmmorca.qmmm.imaging import make_qm_whole  # noqa: E402

spec = importlib.util.spec_from_file_location("embedding_before", HERE / "embedding_before.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.check_cutoff_against_box = lambda *a, **k: None  # S2 prototype: no sphere/box bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("t09", type=Path)
    parser.add_argument("--regions", nargs="+", default=["A", "C"])
    args = parser.parse_args()
    frames = np.load(args.t09 / "frames.npz")
    for region in args.regions:
        topology, state, system, qm, pairs_idx, pairs, parts, charges, groups, graph = setup(
            args.t09 / "run_files", region)
        qm_atoms = list(parts.qm_atoms)
        for label, new_cut, old_cut in (("S0", 1.2, 1.2), ("S2", None, 100.0)):
            new_emb = CutoffEmbedding(groups, charges, new_cut)
            old_emb = old.CutoffEmbedding(groups, charges, old_cut)
            t_new, t_old, n_equal, n_atoms, n_changed = [], [], 0, [], []
            for f in range(len(frames["step"])):
                box = frames["box_nm"][f]
                pos = make_qm_whole(frames["positions_nm"][f], qm_atoms, graph, box)
                t0 = time.perf_counter()
                a = new_emb.select(pos[qm_atoms], pos, box)
                t1 = time.perf_counter()
                b = old_emb.select(pos[qm_atoms], pos, box)
                t2 = time.perf_counter()
                t_new.append(t1 - t0)
                t_old.append(t2 - t1)
                n_equal += all(np.array_equal(x, y) for x, y in zip(a, b))
                n_atoms.append(len(a[0]))
                n_changed.append((new_emb.last_changed, new_emb.last_reimaged, old_emb.last_changed))
            print(f"region {region} {label}: frames identical {n_equal}/{len(t_new)}; "
                  f"embedded atoms {min(n_atoms)}-{max(n_atoms)}; total charge {a[2].sum():+.6f} e; "
                  f"select new median {np.median(t_new):.3f} s, old median {np.median(t_old):.3f} s; "
                  f"(changed, reimaged, old changed) frames 1-3: {n_changed[1:4]}", flush=True)


if __name__ == "__main__":
    main()
