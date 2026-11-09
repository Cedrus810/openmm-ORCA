"""Bounded C2 energy pilot: identical requests under S2 and periodic MM fields.

ENV20/ENV30 remove QM self images during SCF and energy evaluation.
EW20 retains the original B5 reference for a separately labelled diagnostic.
Results are written per (candidate, scheme); existing results are preserved.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, lib, qmmm
from pyscf.qmmm.pbc import itrf

from periodic_reference import remove_qm_images


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("requests", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 4, 8])
    p.add_argument("--schemes", nargs="+", choices=["S0", "S2", "ENV20", "ENV30", "EW20"],
                   default=["S2", "ENV20"])
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    lib.num_threads(args.threads)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with np.load(args.requests, allow_pickle=False) as data:
        for f in args.frames:
            for scheme in args.schemes:
                path = args.outdir / f"candidate_{f:02d}_{scheme}.json"
                if path.exists():
                    raise FileExistsError(f"preserve existing result: {path}")
                base = "S0" if scheme == "S0" else "S2"
                qm = data[f"{base}_qm_pos_{f}"]
                mm = data[f"{base}_mm_pos_{f}"]
                charges = data[f"{base}_mm_q_{f}"]
                elements = data["qm_elements"].tolist()
                mol = gto.M(atom=list(zip(elements, (qm*10).tolist())), basis="def2-svp",
                            charge=-1, spin=0, unit="Angstrom", verbose=3)
                mol.max_memory = 3000
                mf = dft.RKS(mol)
                mf.xc = "r2scan"
                if scheme in ("S0", "S2"):
                    mf = qmmm.mm_charge(mf, mm*10, charges, unit="Angstrom")
                else:
                    rh = 30. if scheme == "ENV30" else 20.
                    mf = itrf.add_mm_charges(mf, mm*10, data["box_nm"][f]*10,
                                             charges, rcut_hcore=rh, unit="Angstrom")
                    if scheme.startswith("ENV"):
                        mf = remove_qm_images(mf)
                mf.conv_tol = 1e-10
                mf.max_cycle = 100
                started = time.perf_counter()
                print(f"START candidate {f} {scheme}", flush=True)
                energy = mf.kernel()
                if not mf.converged or not np.isfinite(energy):
                    raise RuntimeError(f"candidate {f} {scheme}: invalid/unconverged SCF")
                gradient = None
                if scheme in ("S0", "S2"):
                    gradient = np.asarray(mf.nuc_grad_method().kernel())
                    if not np.isfinite(gradient).all():
                        raise RuntimeError("nonfinite molecular-embedding gradient")
                elapsed = time.perf_counter() - started
                record = dict(candidate=f, scheme=scheme, energy_eh=float(energy),
                              converged=True, xc="r2scan", basis="def2-svp", conv_tol=1e-10,
                              pyscf=pyscf.__version__, threads=lib.num_threads(), host=platform.node(),
                              wall_s=elapsed, n_mm=len(charges), mm_charge_e=float(charges.sum()),
                              request_file=str(args.requests.resolve()), argv=sys.argv,
                              qm_positions_nm=qm.tolist(), box_nm=data["box_nm"][f].tolist(),
                              qm_self_images="removed in SCF and energy" if scheme.startswith("ENV")
                              else ("included" if scheme == "EW20" else "absent"),
                              qm_grad_eh_bohr=None if gradient is None else gradient.tolist(),
                              gradients="molecular embedding only" if gradient is not None
                              else "not evaluated", pid=os.getpid())
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(record, indent=2) + "\n")
                tmp.replace(path)
                print(f"DONE candidate {f} {scheme}: E={energy:.12f} Eh, {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
