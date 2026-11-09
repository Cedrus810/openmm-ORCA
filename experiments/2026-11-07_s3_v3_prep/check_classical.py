"""Task 0, items 0 and 3: correctness and cost of the analytic Taylor prototype (taylor.py), pilot frames.

Checks (frame 0 unless noted): c vs finite differences of the exact dV (classical.py) at rc; Taylor
polynomial vs exact dV on points within 0.4 nm of the QM atoms (all three frames, compared with the
V1 order-2 fit); MM correction forces vs finite differences of m.c on a few charges; Ewald
convergence and wall time vs (alpha, kmax).  Writes check_classical.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-11-07_c2_error_decomp"))
from classical import direct, ewald, monomials  # noqa: E402
from taylor import EXPS2, taylor  # noqa: E402

req = np.load(HERE.parent / "2026-11-06_c2_matched_pilot/inputs/requests_A.npz")
V1 = HERE.parent / "2026-11-07_s3_v1_pyscf/out"
out = {"frames": {}}


def dV(pts, mm, q, L):
    return ewald(pts, mm, q, L) - direct(pts, mm, q)


for f in (0, 4, 8):
    qm, mm, q = (req[f"S2_{k}_{f}"].astype(float) for k in ("qm_pos", "mm_pos", "mm_q"))
    L = float(req["box_nm"][f][0, 0])
    rc = qm.mean(0)
    t = time.perf_counter()
    c, _ = taylor(rc, mm, q, L)
    t_c = time.perf_counter() - t
    rng = np.random.default_rng(99 + f)
    d = rng.normal(size=(300, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    pts = qm[rng.integers(len(qm), size=300)] + d * (0.4 * rng.random(300) ** (1 / 3))[:, None]
    exact = dV(pts, mm, q, L)
    A, _ = monomials(pts - rc, 2)
    v1 = np.array(json.loads((V1 / f"candidate_{f:02d}.json").read_text())["fits"]["2"]["coef"])
    rec = {"t_coeff_s": t_c, "c": c.tolist(), "c_v1_fit": v1.tolist(),
           "max_dist_qm_from_rc_nm": float(np.linalg.norm(qm - rc, axis=1).max()),
           "taylor_vs_exact_rms": float(np.sqrt(((A @ c - exact) ** 2).mean())),
           "taylor_vs_exact_max": float(np.abs(A @ c - exact).max()),
           "v1fit_vs_exact_rms": float(np.sqrt(((A @ v1 - exact) ** 2).mean())),
           "v1fit_vs_exact_max": float(np.abs(A @ v1 - exact).max())}
    if f == 0:
        # c vs FD of exact dV at rc (h = 1e-3 nm, 2nd order central stencils)
        h = 1e-3
        e = np.eye(3)
        pts_fd = [rc] + [rc + s * h * e[a] for a in range(3) for s in (1, -1)] + \
                 [rc + s1 * h * e[a] + s2 * h * e[b] for a in range(3) for b in range(a + 1, 3)
                  for s1 in (1, -1) for s2 in (1, -1)]
        v = dV(np.array(pts_fd), mm, q, L)
        v0 = v[0]
        g = [(v[1 + 2 * a] - v[2 + 2 * a]) / (2 * h) for a in range(3)]
        hd = [(v[1 + 2 * a] + v[2 + 2 * a] - 2 * v0) / h ** 2 for a in range(3)]
        off, k = {}, 7
        for a in range(3):
            for b in range(a + 1, 3):
                pp, pm, mp, mm_ = v[k:k + 4]
                off[(a, b)] = (pp - pm - mp + mm_) / (4 * h * h)
                k += 4
        fd = {}
        for i, ex in enumerate(EXPS2):
            n = sum(ex)
            axes = [a for a in range(3) for _ in range(ex[a])]
            fd[i] = v0 if n == 0 else (g[axes[0]] if n == 1 else (hd[axes[0]] / 2 if axes[0] == axes[1]
                                                                   else off[tuple(axes)]))
        fdv = np.array([fd[i] for i in range(len(EXPS2))])
        rec["c_minus_fd"] = (c - fdv).tolist()
        # forces: m = arbitrary moments; compare F_j with FD of m.c wrt r_j (charges near, far, at L/2)
        m = np.array([-1.0, 0.25, -0.2, -0.18, 0.02, 0.01, -0.01, 0.03, 0.005, -0.02])
        t = time.perf_counter()
        c2, F = taylor(rc, mm, q, L, want_force_m=m)
        rec["t_coeff_and_force_s"] = time.perf_counter() - t
        dist = np.linalg.norm(mm - rc, axis=1)
        order = np.argsort(dist)
        probe = [int(order[0]), int(order[len(order) // 2]), int(order[-1]), int(order[-2])]
        hf = 1e-5
        fdF = []
        for j in probe:
            row = []
            for a in range(3):
                p1 = mm.copy(); p1[j, a] += hf
                p2 = mm.copy(); p2[j, a] -= hf
                e1 = m @ taylor(rc, p1, q, L)[0]
                e2 = m @ taylor(rc, p2, q, L)[0]
                row.append(-(e1 - e2) / (2 * hf))
            fdF.append(row)
        rec["force_probe"] = {"atoms": probe, "dist_nm": dist[probe].tolist(), "analytic": F[probe].tolist(),
                              "fd": fdF, "max_abs_diff": float(np.abs(F[probe] - np.array(fdF)).max()),
                              "max_abs_force": float(np.abs(F).max()),
                              "sum_F_plus_rc_term_note": "sum_j F_j = -(m.dc/drc); checked in Task 1"}
        # Ewald settings: convergence and cost
        ref_c, ref_F = taylor(rc, mm, q, L, alpha=1.6, kmax=22, want_force_m=m)
        conv = []
        for alpha, kmax in ((1.2, 14), (1.2, 12), (1.2, 10), (1.0, 10), (1.0, 8), (1.0, 12)):
            t = time.perf_counter()
            cc, FF = taylor(rc, mm, q, L, alpha=alpha, kmax=kmax, want_force_m=m)
            conv.append({"alpha": alpha, "kmax": kmax, "wall_s": time.perf_counter() - t,
                         "max_dc_nonconst": float(np.abs((cc - ref_c)[1:]).max()),
                         "dc0": float(cc[0] - ref_c[0]),
                         "max_dF": float(np.abs(FF - ref_F).max()),
                         "rms_dF": float(np.sqrt(((FF - ref_F) ** 2).mean()))})
        rec["ewald_convergence_vs_alpha1.6_k22"] = conv
    out["frames"][f] = rec
    print(f, json.dumps({k: v for k, v in rec.items() if k not in ("c", "c_v1_fit")}, indent=1), flush=True)
(HERE / "check_classical.json").write_text(json.dumps(out, indent=2) + "\n")
