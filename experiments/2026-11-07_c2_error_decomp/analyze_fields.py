"""Summarize fields/wXX.npz: statistics of the classical S2-vs-Ewald field difference over 320 frames.

Quantities (kJ/mol/e, kJ/mol/nm/e, e nm): c0 = dV at the QM centroid, g = grad dV, M = S2 cluster
dipole about the centroid.  With QM charge -1 and the first-order model of decompose.py,
D ~ const + c0 - mu.g - (quadrupole term); here
  D_mono = c0                          (exact classical, no QM input)
  D_dip  = -mu(rc).g                   (mu interpolated in RC from the three pilot S2 densities;
                                        an estimate, not a QM result)
Window errors relative to w00: Delta(W) = <D>_W - <D>_w00, SE from frame SDs with n = 20
(frames are correlated; SE is a lower bound).
Writes field_stats.json and prints tables.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    W = [np.load(HERE / "fields" / f"w{w:02d}.npz") for w in range(16)]
    L = float(W[0]["box_L"])
    KE = float(W[0]["KE"])
    k_surf = KE * 4 * np.pi / (3 * L ** 3)
    cat = {k: np.concatenate([w[k] for w in W]) for k in ("rc", "rc_saved", "c0", "g", "M", "test_rms", "test_max",
                                                           "mm_charge", "n_mm")}
    win = np.repeat(np.arange(16), [len(w["rc"]) for w in W])
    out = {"n_frames": int(len(win)), "box_nm": L, "surface_coeff_kj_mol_nm_e_per_e_nm": k_surf}

    # checks
    out["rc_recomputed_minus_saved_max_nm"] = float(np.abs(cat["rc"] - cat["rc_saved"]).max())
    out["fit_test_rms_max"] = float(cat["test_rms"].max())
    out["fit_test_max_max"] = float(cat["test_max"].max())
    out["mm_charge_range"] = [float(cat["mm_charge"].min()), float(cat["mm_charge"].max())]
    req = np.load(HERE.parent / "2026-11-06_c2_matched_pilot/inputs/requests_A.npz")
    pilot = {0: (0, 0), 4: (6, 15), 8: (15, 19)}
    out["pilot_identity"] = {}
    mus, rcs = {}, {}
    for cand, (w, fr) in pilot.items():
        d = json.loads((HERE / "out" / f"candidate_{cand:02d}.json").read_text())
        same = bool(np.array_equal(W[w]["qm_pos"][fr], req[f"S2_qm_pos_{cand}"]))
        out["pilot_identity"][cand] = {"qm_pos_identical": same,
                                       "dc0": float(W[w]["c0"][fr] - d["c0_kj_mol_e"]),
                                       "dg_max": float(np.abs(W[w]["g"][fr] - d["field_grad_kj_mol_nm_e"]).max())}
        mus[cand] = np.array(d["dipole_e_nm"])
        rcs[cand] = float(W[w]["rc"][fr])

    # g vs M
    g, M = cat["g"], cat["M"]
    cos = -(g * M).sum(1) / np.linalg.norm(g, axis=1) / np.linalg.norm(M, axis=1)
    ratio = np.linalg.norm(g, axis=1) / (k_surf * np.linalg.norm(M, axis=1))
    # least squares g = A M + b (3x3 A), and scalar fit g = -s k M
    X = np.hstack([M, np.ones((len(M), 1))])
    coef, *_ = np.linalg.lstsq(X, g, rcond=None)
    resid = g - X @ coef
    s = -float((g * M).sum() / (k_surf * (M * M).sum()))
    resid_s = g + s * k_surf * M
    Mhat = M / np.linalg.norm(M, axis=1)[:, None]
    out["g_vs_M"] = {
        "cos_antiparallel_min_median": [float(cos.min()), float(np.median(cos))],
        "ratio_|g|/|kM|_min_median_max": [float(ratio.min()), float(np.median(ratio)), float(ratio.max())],
        "scalar_fit_s": s, "scalar_fit_resid_rms": float(np.sqrt((resid_s ** 2).sum(1).mean())),
        "linear_fit_resid_rms": float(np.sqrt((resid ** 2).sum(1).mean())),
        "g_rms": float(np.sqrt((g ** 2).sum(1).mean())),
        "g_sd_about_mean_rms": float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean())),
        "M_direction_spread_min_cos_to_mean": float((Mhat @ (Mhat.mean(0) / np.linalg.norm(Mhat.mean(0)))).min()),
        "M_norm_range": [float(np.linalg.norm(M, axis=1).min()), float(np.linalg.norm(M, axis=1).max())],
    }

    # energy-error estimate
    order = sorted(rcs, key=rcs.get)
    rc_k = np.array([rcs[c] for c in order])
    mu_k = np.array([mus[c] for c in order])
    mu = np.stack([np.interp(cat["rc"], rc_k, mu_k[:, a]) for a in range(3)], 1)
    D_mono = cat["c0"]
    D_dip = -(mu * g).sum(1)
    mu_R = mus[0]
    D_dip_fixmu = -(mu_R * g).sum(1)          # same QM dipole everywhere: environment-only part of dipole term
    rows = []
    for w in range(16):
        m = win == w
        r = {"window": w, "rc_mean": float(cat["rc"][m].mean()),
             "g_mean": g[m].mean(0).tolist(), "g_sd": g[m].std(0, ddof=1).tolist(),
             "abs_g_mean": float(np.linalg.norm(g[m], axis=1).mean()),
             "M_mean": M[m].mean(0).tolist()}
        for name, arr in (("mono", D_mono), ("dip_est", D_dip), ("dip_fixmu", D_dip_fixmu),
                          ("total_est", D_mono + D_dip)):
            r[f"{name}_mean"] = float(arr[m].mean())
            r[f"{name}_sd"] = float(arr[m].std(ddof=1))
        rows.append(r)
    R = rows[0]
    for r in rows:
        for name in ("mono", "dip_est", "dip_fixmu", "total_est"):
            r[f"{name}_rel"] = r[f"{name}_mean"] - R[f"{name}_mean"]
            r[f"{name}_rel_se"] = float(np.sqrt((r[f"{name}_sd"] ** 2 + R[f"{name}_sd"] ** 2) / 20))
    out["windows"] = rows
    out["mu_interp"] = {"rc_nm": rc_k.tolist(), "mu_e_nm": mu_k.tolist(),
                        "note": "piecewise-linear in RC from the three pilot S2 densities; clamped outside"}
    # lag-1 autocorrelation of g within windows
    ac = []
    for w in W:
        x = w["g"] - w["g"].mean(0)
        ac.append(float((x[1:] * x[:-1]).sum() / (x * x).sum()))
    out["g_lag1_autocorr_per_window"] = ac
    (HERE / "field_stats.json").write_text(json.dumps(out, indent=2) + "\n")

    print(json.dumps({k: v for k, v in out.items() if k not in ("windows", "g_lag1_autocorr_per_window")}, indent=1))
    print("lag1 autocorr g:", np.round(ac, 2))
    print("| W | RC | g mean | g SD | |g| | M mean | mono rel±SE | dip_est rel±SE | dip_fixmu rel | total_est rel±SE |")
    for r in rows:
        f = lambda v: "(" + ", ".join(f"{x:.1f}" for x in v) + ")"
        print(f"| {r['window']:02d} | {r['rc_mean']:+.3f} | {f(r['g_mean'])} | {f(r['g_sd'])} | {r['abs_g_mean']:.1f} | "
              f"{f(r['M_mean'])} | {r['mono_rel']:+.2f}±{r['mono_rel_se']:.2f} | {r['dip_est_rel']:+.2f}±{r['dip_est_rel_se']:.2f} | "
              f"{r['dip_fixmu_rel']:+.2f} | {r['total_est_rel']:+.2f}±{r['total_est_rel_se']:.2f} |")


if __name__ == "__main__":
    main()
