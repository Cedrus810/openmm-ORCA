"""T09 item 1 analysis: differences against the largest cutoff on matched snapshots.

Reported per cutoff (frames present in both files):
* QM atoms: max and RMS |ΔF| (kJ/mol/nm) against the reference cutoff;
* key MM atoms (boundary M1 and MM atoms within --near nm of any QM atom in that frame): max |ΔF|;
* relative energies E(frame) − E(first frame) against the reference, so the constant truncation
  offset cancels; absolute ΔE is listed separately and is the truncated QM–MM interaction, not
  an accuracy measure;
* embedded groups and ORCA time.
Usage: analyze_cutoff_scan.py OUTDIR [--near 0.5]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cutoff_scan import load
from openmmorca.units import HARTREE_IN_KJ_MOL

parser = argparse.ArgumentParser()
parser.add_argument("outdir", type=Path)
parser.add_argument("--near", type=float, default=0.5)
args = parser.parse_args()
meta = json.loads((args.outdir / "meta.json").read_text())
frames = np.load(meta["frames_npz"])
qm = np.array(meta["qm_atoms"])
m1 = np.array([m for _, m in meta["boundary_pairs"]], dtype=int)
scans = {float(p.stem[1:]): load(p) for p in sorted(args.outdir.glob("c*.npz")) if not p.stem.endswith(".tmp")}
ref_cut = max(scans)
ref = scans[ref_cut]


def key_mm(f):
    pos, box = frames["positions_nm"][f], np.diag(frames["box_nm"][f])
    d = pos[None, :, :] - pos[qm][:, None, :]
    d -= box * np.round(d / box)  # rectangular box (DhlA run is cubic)
    near = np.where(np.linalg.norm(d, axis=2).min(axis=0) < args.near)[0]
    return np.setdiff1d(np.union1d(near, m1), qm)


report = {"reference_cutoff_nm": ref_cut, "near_nm": args.near, "cutoffs": []}
print(f"reference cutoff {ref_cut} nm; key MM = boundary M1 + MM atoms within {args.near} nm of QM\n")
print("| cutoff (nm) | frames | groups | t_orca (s) | QM max|ΔF| | QM RMS|ΔF| | key-MM max|ΔF| | rel. E RMS dev (kJ/mol) | mean abs ΔE (kJ/mol) |")
print("|---|---|---|---|---|---|---|---|---|")
for cut in sorted(scans):
    scan = scans[cut]
    common = sorted(set(scan) & set(ref))
    if not common:
        continue
    dq = np.concatenate([scan[f]["forces"][qm] - ref[f]["forces"][qm] for f in common])
    dk = max(np.abs(scan[f]["forces"][key_mm(f)] - ref[f]["forces"][key_mm(f)]).max() for f in common)
    e = np.array([float(scan[f]["energy_kj_mol"]) for f in common])
    er = np.array([float(ref[f]["energy_kj_mol"]) for f in common])
    rel = (e - e[0]) - (er - er[0])
    row = {"cutoff_nm": cut, "frames": common,
           "n_groups_mean": float(np.mean([scan[f]["n_embed_groups"] for f in common])),
           "t_orca_mean_s": float(np.mean([scan[f]["t_orca"] for f in common])),
           "qm_max_dF": float(np.linalg.norm(dq, axis=1).max()),
           "qm_rms_dF": float(np.sqrt((dq ** 2).sum(axis=1).mean())),
           "key_mm_max_dF": float(dk), "rel_E_rms_dev": float(np.sqrt((rel[1:] ** 2).mean())) if len(rel) > 1 else None,
           "abs_dE_mean": float(np.mean(np.abs(e - er))), "abs_dE_eh": ((e - er) / HARTREE_IN_KJ_MOL).tolist()}
    report["cutoffs"].append(row)
    rel_text = "—" if row["rel_E_rms_dev"] is None else f"{row['rel_E_rms_dev']:.3f}"
    print(f"| {cut:.3f} | {len(common)} | {row['n_groups_mean']:.0f} | {row['t_orca_mean_s']:.1f} | "
          f"{row['qm_max_dF']:.2f} | {row['qm_rms_dF']:.2f} | {row['key_mm_max_dF']:.2f} | {rel_text} | {row['abs_dE_mean']:.2f} |")
(args.outdir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
