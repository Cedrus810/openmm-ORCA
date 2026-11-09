"""ORCA r2SCAN-3c S0/S2 bridge on the three captured region-A pilot inputs."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openmmorca.backend.base import QMRequest
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("requests", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 4, 8])
    p.add_argument("--schemes", nargs="+", choices=["S0", "S2"], default=["S2", "S0"])
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = ORCAConfig(method="r2SCAN-3c", charge=-1, multiplicity=1, nprocs=1,
                        restart=False, scratch_root="/tmp/openmmorca-c2-pilot", timeout_s=600)
    backend = ORCAOPIBackend(config)
    try:
        with np.load(args.requests, allow_pickle=False) as data:
            for f in args.frames:
                for scheme in args.schemes:
                    path = args.outdir / f"candidate_{f:02d}_{scheme}.json"
                    if path.exists():
                        raise FileExistsError(f"preserve existing result: {path}")
                    request = QMRequest(qm_elements=tuple(data["qm_elements"].tolist()),
                                        qm_positions_nm=data[f"{scheme}_qm_pos_{f}"],
                                        mm_positions_nm=data[f"{scheme}_mm_pos_{f}"],
                                        mm_charges_e=data[f"{scheme}_mm_q_{f}"], step=f)
                    print(f"START candidate {f} {scheme}", flush=True)
                    started = time.perf_counter()
                    result = backend.evaluate(request)
                    record = dict(candidate=f, scheme=scheme, energy_kj_mol=result.energy_kj_mol,
                                  qm_forces_kj_mol_nm=result.qm_forces_kj_mol_nm.tolist(),
                                  qm_positions_nm=request.qm_positions_nm.tolist(),
                                  qm_atoms=data["qm_atoms"].tolist(),
                                  n_mm=request.n_mm, mm_charge_e=float(request.mm_charges_e.sum()),
                                  request_file=str(args.requests.resolve()), method="r2SCAN-3c",
                                  nprocs=1, restart=False, host=platform.node(),
                                  wall_s=time.perf_counter()-started, timings_s=result.timings_s,
                                  box_nm=data["box_nm"][f].tolist(), argv=sys.argv)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(record, indent=2) + "\n")
                    tmp.replace(path)
                    print(f"DONE candidate {f} {scheme}: E={result.energy_kj_mol:.6f} kJ/mol, "
                          f"{record['wall_s']:.1f} s", flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
