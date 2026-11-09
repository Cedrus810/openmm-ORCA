"""V5 (C1 §4 with S3): energies E_S2, E_S3, E_ENV20 on every captured frame, PySCF r2SCAN/def2-SVP.

S2: molecular embedding of the S2 charges; S3: S2 charges + pseudo charges; ENV20: periodic MM field without
QM self images (C2 pilot periodic_reference.py), rcut_hcore 20 A.  conv_tol 1e-10, as the pilot/B5.
Output out/wXX_fYY_<scheme>.json; existing files skipped.  Energies only.
Usage: v5_pyscf.py --worker I --nworkers N --threads T [--windows ...] [--frames ...]   (pyscf-env)
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib, qmmm
from pyscf.qmmm.pbc import itrf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/2026-11-06_c2_matched_pilot"))
from periodic_reference import remove_qm_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--nworkers", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 5, 6, 7, 15])
    ap.add_argument("--frames", type=int, nargs="+", default=list(range(20)))
    a = ap.parse_args()
    lib.num_threads(a.threads)
    (HERE / "out").mkdir(exist_ok=True)
    jobs = [(w, f, s) for w in a.windows for f in a.frames for s in ("ENV20", "S2", "S3")]
    data = {}
    for k, (w, f, s) in enumerate(jobs):
        if k % a.nworkers != a.worker:
            continue
        path = HERE / "out" / f"w{w:02d}_f{f:02d}_{s}.json"
        if path.exists():
            continue
        if w not in data:
            data[w] = np.load(HERE / "requests" / f"w{w:02d}.npz")
        z = data[w]
        qm, mm, q = z["qm_pos"][f] * 10.0, z["mm_pos"][f] * 10.0, z["mm_q"][f]
        mol = gto.M(atom=list(zip([str(e) for e in z["qm_elements"]], qm.tolist())), basis="def2-svp",
                    charge=-1, spin=0, unit="Angstrom", verbose=3)
        mf = dft.RKS(mol)
        mf.xc = "r2scan"
        if s == "S2":
            mf = qmmm.mm_charge(mf, mm, q, unit="Angstrom")
        elif s == "S3":
            mf = qmmm.mm_charge(mf, np.vstack([mm, z["sites"][f] * 10.0]), np.concatenate([q, z["Q"][f]]),
                                unit="Angstrom")
        else:
            mf = remove_qm_images(itrf.add_mm_charges(mf, mm, z["box"][f] * 10.0, q, rcut_hcore=20.0,
                                                      unit="Angstrom"))
        mf.conv_tol = 1e-10
        t0 = time.perf_counter()
        e = mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"w{w} f{f} {s}: SCF not converged")
        rec = dict(window=w, frame=f, scheme=s, energy_eh=float(e), step=int(z["step"][f]), rc_nm=float(z["rc"][f]),
                   wall_s=time.perf_counter() - t0, threads=lib.num_threads(), host=platform.node())
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec) + "\n")
        tmp.replace(path)
        print(f"DONE w{w:02d} f{f:02d} {s}: E={e:.10f} {rec['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
