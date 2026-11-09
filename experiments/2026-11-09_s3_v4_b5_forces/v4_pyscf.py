"""V4 (S3 proposal §4): stretch/approach force offsets of S2 and S3 vs the periodic MM reference, B5 frames.

PySCF r2SCAN/def2-SVP, conv_tol 1e-10, same requests and settings as B5 E1 (pyscf_e1.py).
Jobs per frame f (out/frame_ff_<job>.json, existing files skipped -> resumable):
  S3              S2 charges + pseudo charges (pseudo_b5.npz): energy + analytic QM gradient
  ENV20           periodic MM field without QM self images (C2 pilot periodic_reference.py), energy
  ENV20_<p>_<s>   ENV20 energy with the pair displaced along projection p (stretch: C1 +lam/2 u, Cl1 -lam/2 u;
                  approach: C1 +lam/2 v, O -lam/2 v), s = p|m, lam = 0.002 Angstrom
frame 0 only (controls): S2chk (S2 energy + gradient, vs B5), EW20_<p>_<s> (vs B5 EW20 analytic gradient).
--s3fd: only S3_<p>_<s> on frame 0 (S3 energy differences vs the S3 analytic gradient: checks the
difference procedure on a path whose analytic gradient is validated; added after the smoke, 2026-11-09).
S2_<p>_<s>, S3_<p>_<s>: the same differences for S2 and S3 (added 2026-11-09 after the smoke: the analytic
r2SCAN gradients differ from energy differences by a common ~16.7 kJ/mol/nm in the stretch projection, so all
comparisons are made difference-to-difference).
Projection force = -(E+ - E-)/(2 lam); definitions as b5_analyze.py.
Usage: v4_pyscf.py --worker I --nworkers N [--frames ...] [--threads T]   (pyscf-env)
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

LAM = 0.002  # Angstrom


def displaced(qm_ang, names, proj, sign):
    pos = qm_ang.copy()
    k1, kc = names.index("C1"), names.index("CL1")
    ko = min((names.index(n) for n in ("OD1", "OD2")), key=lambda i: np.linalg.norm(pos[k1] - pos[i]))
    other = kc if proj == "stretch" else ko
    u = pos[k1] - pos[other]
    u /= np.linalg.norm(u)
    pos[k1] += sign * LAM / 2 * u
    pos[other] -= sign * LAM / 2 * u
    return pos


def jobs(frames):
    out = []
    for f in frames:
        out += [(f, "S3"), (f, "ENV20")] + [(f, f"{r}_{p}_{s}") for r in ("ENV20", "S2", "S3")
                                             for p in ("stretch", "approach") for s in "pm"]
        if f == 0:
            out += [(0, "S2chk")] + [(0, f"EW20_{p}_{s}") for p in ("stretch", "approach") for s in "pm"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--nworkers", type=int, required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--s3fd", action="store_true")
    a = ap.parse_args()
    lib.num_threads(a.threads)
    req = np.load(ROOT / "experiments/2026-10-29_b5_scope/requests_t09_A.npz")
    ps = np.load(HERE / "pseudo_b5.npz")
    names = [str(n) for n in req["qm_names"]]
    elements = [str(e) for e in req["qm_elements"]]
    (HERE / "out").mkdir(exist_ok=True)
    todo = [(0, f"S3_{p}_{s}") for p in ("stretch", "approach") for s in "pm"] if a.s3fd else jobs(a.frames)
    for k, (f, job) in enumerate(todo):
        if k % a.nworkers != a.worker:
            continue
        path = HERE / "out" / f"frame_{f:02d}_{job}.json"
        if path.exists():
            continue
        qm = req[f"S2_qm_pos_{f}"] * 10.0
        mm, q = req[f"S2_mm_pos_{f}"] * 10.0, req[f"S2_mm_q_{f}"]
        if "_" in job:
            ref, proj, s = job.split("_")
            qm = displaced(qm, names, proj, 1 if s == "p" else -1)
        else:
            ref = job
        mol = gto.M(atom=list(zip(elements, qm.tolist())), basis="def2-svp", charge=-1, spin=0,
                    unit="Angstrom", verbose=3)
        mf = dft.RKS(mol)
        mf.xc = "r2scan"
        if ref == "S3":
            mf = qmmm.mm_charge(mf, np.vstack([mm, ps[f"sites_{f}"] * 10.0]),
                                np.concatenate([q, ps[f"Q_{f}"]]), unit="Angstrom")
        elif ref in ("S2chk", "S2"):
            mf = qmmm.mm_charge(mf, mm, q, unit="Angstrom")
        else:
            mf = itrf.add_mm_charges(mf, mm, req["box_nm"][f] * 10.0, q, rcut_hcore=20.0, unit="Angstrom")
            if ref == "ENV20":
                mf = remove_qm_images(mf)
        mf.conv_tol = 1e-10
        t0 = time.perf_counter()
        e = mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"frame {f} {job}: SCF not converged")
        t1 = time.perf_counter()
        grad = mf.nuc_grad_method().kernel().tolist() if job in ("S3", "S2chk") else None
        rec = dict(frame=f, job=job, energy_eh=float(e), qm_grad_eh_bohr=grad, lam_angstrom=LAM,
                   t_scf_s=t1 - t0, t_grad_s=time.perf_counter() - t1, host=platform.node(),
                   threads=lib.num_threads())
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec) + "\n")
        tmp.replace(path)
        print(f"DONE frame {f} {job}: E={e:.10f} SCF {t1 - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
