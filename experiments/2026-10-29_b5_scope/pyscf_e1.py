"""B5 E1 (183, pyscf-env): S2 vs periodic Ewald QM/MM in PySCF on the dumped ORCA inputs (region A).

Schemes per frame (same QM method, same point charges as ORCA):
  S2       non-periodic pyscf.qmmm.mm_charge with the whole-box minimum-image charges
  S0       same with the 1.2 nm charges (for the ORCA bridge check, E1c)
  EW<rh>   pyscf.qmmm.pbc.itrf: charges within rcut_hcore (rh, Angstrom) of the QM centre exact,
           the rest by Ewald with QM multipoles; lattice = frame box
Writes OUTDIR/frame_<f>.json (energy Eh, QM gradient Eh/bohr incl. link H, wall s); skips existing files.

Usage: pyscf_e1.py REQUESTS.npz OUTDIR --frames 0 [--schemes S2 EW20 EW30] [--xc r2scan|hf]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, qmmm, scf
from pyscf.qmmm.pbc import itrf as pbc_itrf

parser = argparse.ArgumentParser()
parser.add_argument("requests", type=Path)
parser.add_argument("outdir", type=Path)
parser.add_argument("--frames", type=int, nargs="+", required=True)
parser.add_argument("--schemes", nargs="+", default=["S2", "EW20", "EW30"])
parser.add_argument("--xc", default="r2scan")
parser.add_argument("--basis", default="def2-svp")
args = parser.parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)
data = np.load(args.requests)
elements = [str(e) for e in data["qm_elements"]]


def make_mf(mol):
    if args.xc == "hf":
        return scf.RHF(mol)
    mf = dft.RKS(mol)
    mf.xc = args.xc
    return mf


for f in args.frames:
    path = args.outdir / f"frame_{f:02d}.json"
    out = json.loads(path.read_text()) if path.is_file() else {}
    for scheme in args.schemes:
        if scheme in out:
            continue
        base = "S0" if scheme == "S0" else "S2"
        qm = data[f"{base}_qm_pos_{f}"] * 10.0
        mm_pos = data[f"{base}_mm_pos_{f}"] * 10.0
        mm_q = data[f"{base}_mm_q_{f}"]
        mol = gto.M(atom=list(zip(elements, qm.tolist())), basis=args.basis, charge=-1, spin=0,
                    unit="Angstrom", verbose=3)
        t0 = time.perf_counter()
        if scheme.startswith("EW"):
            box = data["box_nm"][f] * 10.0
            mf = pbc_itrf.add_mm_charges(make_mf(mol), mm_pos, box, mm_q,
                                         rcut_hcore=float(scheme[2:]), unit="Angstrom")
        else:
            mf = qmmm.mm_charge(make_mf(mol), mm_pos, mm_q, unit="Angstrom")
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged, f"frame {f} {scheme}: SCF not converged"
        t1 = time.perf_counter()
        grad = mf.nuc_grad_method().kernel()
        t2 = time.perf_counter()
        out[scheme] = {"energy_eh": float(energy), "qm_grad_eh_bohr": np.asarray(grad).tolist(),
                       "t_scf_s": t1 - t0, "t_grad_s": t2 - t1, "n_mm": int(mm_q.size),
                       "xc": args.xc, "basis": args.basis}
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(f"frame {f} {scheme}: E {energy:.10f} Eh, SCF {t1 - t0:.0f} s, grad {t2 - t1:.0f} s", flush=True)
