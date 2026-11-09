"""ORCA r2SCAN-3c S2/S3 energies on the V5 window frames (same requests and pseudo charges as the PySCF V5 run).

Backend as the pilot bridge: ORCAOPIBackend, nprocs 1, fresh SCF, default SCF settings; scratch /dev/shm.
Output out/wXX_fYY_<S2|S3>.json; existing files skipped.
Usage: orca_bridge.py --worker I --nworkers N [--windows ...] [--frames ...]   (openmm_dev; OPI_ORCA set)
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
V5 = ROOT / "experiments/2026-11-09_s3_v5_c1_windows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--nworkers", type=int, required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 5, 6, 7, 15])
    ap.add_argument("--frames", type=int, nargs="+", default=list(range(20)))
    a = ap.parse_args()
    (HERE / "out").mkdir(exist_ok=True)
    jobs = [(w, f, s) for w in a.windows for f in a.frames for s in ("S2", "S3")]
    mine = [j for k, j in enumerate(jobs) if k % a.nworkers == a.worker
            and not (HERE / "out" / f"w{j[0]:02d}_f{j[1]:02d}_{j[2]}.json").exists()]
    if not mine:
        return
    from openmmorca.backend.base import QMRequest
    from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend

    backend = ORCAOPIBackend(ORCAConfig(method="r2SCAN-3c", charge=-1, multiplicity=1, nprocs=1, restart=False,
                                        scratch_root=f"/dev/shm/openmmorca-bridge-w{a.worker}", timeout_s=1800))
    data = {}
    try:
        for w, f, s in mine:
            if w not in data:
                data[w] = np.load(V5 / "requests" / f"w{w:02d}.npz")
            z = data[w]
            mm, q = z["mm_pos"][f], z["mm_q"][f]
            if s == "S3":
                mm, q = np.vstack([mm, z["sites"][f]]), np.concatenate([q, z["Q"][f]])
            request = QMRequest(qm_elements=tuple(str(e) for e in z["qm_elements"]), qm_positions_nm=z["qm_pos"][f],
                                mm_positions_nm=mm, mm_charges_e=q, step=f)
            t = time.perf_counter()
            result = backend.evaluate(request)
            rec = dict(window=w, frame=f, scheme=s, energy_kj_mol=result.energy_kj_mol, step=int(z["step"][f]),
                       rc_nm=float(z["rc"][f]), n_mm=request.n_mm, wall_s=time.perf_counter() - t,
                       timings_s=result.timings_s, host=platform.node(), method="r2SCAN-3c", nprocs=1)
            path = HERE / "out" / f"w{w:02d}_f{f:02d}_{s}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec) + "\n")
            tmp.replace(path)
            print(f"DONE w{w:02d} f{f:02d} {s}: E={result.energy_kj_mol:.6f} {rec['wall_s']:.0f} s", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
