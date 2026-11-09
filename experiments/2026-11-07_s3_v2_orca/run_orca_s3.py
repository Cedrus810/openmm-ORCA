"""S3 step V2: ORCA r2SCAN-3c single points, S2 and S3_P2 (S2 charges + V1 pseudo charges), one frame.

Settings as the pilot ORCA bridge (nprocs 1, fresh SCF); scratch in /dev/shm.
Usage: run_orca_s3.py FRAME   (writes out/candidate_XX_{S2,S3P2}.json)
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from openmmorca.backend.base import QMRequest  # noqa: E402
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend  # noqa: E402


def main():
    f = int(sys.argv[1])
    outdir = HERE / "out"
    outdir.mkdir(exist_ok=True)
    req = np.load(ROOT / "experiments/2026-11-06_c2_matched_pilot/inputs/requests_A.npz")
    ps = np.load(HERE / "pseudo_K50_R2.0_lam1e-4.npz")
    qm, mm, q = req[f"S2_qm_pos_{f}"], req[f"S2_mm_pos_{f}"], req[f"S2_mm_q_{f}"]
    schemes = {"S2": (mm, q),
               "S3P2": (np.vstack([mm, ps[f"sites_{f}"]]), np.concatenate([q, ps[f"Q_{f}"]]))}
    config = ORCAConfig(method="r2SCAN-3c", charge=-1, multiplicity=1, nprocs=1, restart=False,
                        scratch_root="/dev/shm/openmmorca-s3-v2", timeout_s=900)
    backend = ORCAOPIBackend(config)
    try:
        for scheme, (pos, charges) in schemes.items():
            request = QMRequest(qm_elements=tuple(req["qm_elements"].tolist()), qm_positions_nm=qm,
                                mm_positions_nm=pos, mm_charges_e=charges, step=f)
            t = time.perf_counter()
            result = backend.evaluate(request)
            rec = dict(candidate=f, scheme=scheme, energy_kj_mol=result.energy_kj_mol,
                       qm_forces_kj_mol_nm=result.qm_forces_kj_mol_nm.tolist(),
                       pseudo_forces_kj_mol_nm=result.mm_forces_kj_mol_nm[len(q):].tolist() if scheme == "S3P2" else None,
                       n_mm=request.n_mm, mm_charge_e=float(charges.sum()), method="r2SCAN-3c", nprocs=1,
                       restart=False, host=platform.node(), wall_s=time.perf_counter() - t,
                       timings_s=result.timings_s, argv=sys.argv)
            (outdir / f"candidate_{f:02d}_{scheme}.json").write_text(json.dumps(rec, indent=2) + "\n")
            print(f"DONE {f} {scheme}: E={result.energy_kj_mol:.6f} kJ/mol, {rec['wall_s']:.1f} s", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
