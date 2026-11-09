"""Task 0, item 2: does orca_vpot give dE/dQ_k at the pseudo-charge sites?  (ORCA r2SCAN-3c, frame 0)

Request: pilot frame 0, S2 charges + the V2 pseudo charges (K=50).  Modes:
  base          single point; keep qm.gbw/qm.densities in vpot/; run orca_vpot at the 3 probe sites
  fd K SIGN     single point with Q_K -> Q_K + SIGN*H (H = 0.05 e); energy only
Probe sites: pseudo indices 0, 17, 33.  VeryTightSCF, nprocs 1, fresh SCF, scratch in /dev/shm.
Usage: check_vpot.py base | check_vpot.py fd K +1|-1   (writes vpot/*.json)
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from openmmorca.backend.base import QMRequest  # noqa: E402
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend  # noqa: E402
from openmmorca.units import BOHR_IN_ANGSTROM, HARTREE_IN_KJ_MOL  # noqa: E402

PROBES, H, F = (0, 17, 33), 0.05, 0


def request(dq=None):
    req = np.load(ROOT / "experiments/2026-11-06_c2_matched_pilot/inputs/requests_A.npz")
    ps = np.load(ROOT / "experiments/2026-11-07_s3_v2_orca/pseudo_K50_R2.0_lam1e-4.npz")
    Q = ps[f"Q_{F}"].copy()
    if dq is not None:
        Q[dq[0]] += dq[1]
    pos = np.vstack([req[f"S2_mm_pos_{F}"], ps[f"sites_{F}"]])
    return QMRequest(qm_elements=tuple(req["qm_elements"].tolist()), qm_positions_nm=req[f"S2_qm_pos_{F}"],
                     mm_positions_nm=pos, mm_charges_e=np.concatenate([req[f"S2_mm_q_{F}"], Q]), step=0), ps


def main():
    mode = sys.argv[1]
    outdir = HERE / "vpot"
    outdir.mkdir(exist_ok=True)
    dq = None if mode == "base" else (int(sys.argv[2]), float(sys.argv[3]) * H)
    tag = "base" if mode == "base" else f"fd_{dq[0]}_{'p' if dq[1] > 0 else 'm'}"
    cfg = ORCAConfig(method="r2SCAN-3c", charge=-1, multiplicity=1, nprocs=1, restart=False,
                     extra_keywords=("VeryTightSCF",), scratch_root=f"/dev/shm/openmmorca-s3-vpot-{tag}",
                     timeout_s=1800)
    backend = ORCAOPIBackend(cfg)
    try:
        r, ps = request(dq)
        t = time.perf_counter()
        res = backend.evaluate(r)
        rec = dict(mode=mode, dq=dq, energy_kj_mol=res.energy_kj_mol, wall_s=time.perf_counter() - t)
        if mode == "base":
            cur = backend.scratch.current
            rec["current_files"] = sorted(p.name for p in cur.iterdir())
            keep = outdir / "base_files"
            keep.mkdir(exist_ok=True)
            for name in ("qm.gbw", "qm.densities", "qm.densitiesinfo", "qm.inp", "qm.out"):
                if (cur / name).exists():
                    shutil.copy2(cur / name, keep / name)
            sites_bohr = ps[f"sites_{F}"][list(PROBES)] * 10 / BOHR_IN_ANGSTROM
            (keep / "points.xyz").write_text(f"{len(PROBES)}\n" + "".join(
                f"{x:.10f} {y:.10f} {z:.10f}\n" for x, y, z in sites_bohr))
            vpot = Path(os.environ["OPI_ORCA"]) / "orca_vpot"
            t = time.perf_counter()
            pr = subprocess.run([str(vpot), "qm.gbw", "scfp", "points.xyz", "pot.out"], cwd=keep,
                                capture_output=True, text=True)
            rec["vpot_wall_s"] = time.perf_counter() - t
            rec["vpot_rc"] = pr.returncode
            rec["vpot_stdout_tail"] = pr.stdout[-2000:]
            rec["vpot_stderr_tail"] = pr.stderr[-2000:]
            if (keep / "pot.out").exists():
                rec["pot_out"] = (keep / "pot.out").read_text()
        (outdir / f"{tag}.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "vpot_stdout_tail"}, indent=1), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
