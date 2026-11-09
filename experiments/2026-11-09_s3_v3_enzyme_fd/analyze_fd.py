"""Summarize the S3 enzyme finite-difference check (out/*.json) -> fd_summary.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = HERE / "out"
base = json.loads((out / "base.json").read_text())
H = base["h_nm"]
rows = []
for label in base["atoms"]:
    for axis in (0, 1):
        p = json.loads((out / f"fd_{label}_{axis}_p.json").read_text())
        m = json.loads((out / f"fd_{label}_{axis}_m.json").read_text())
        fd = -(p["energy_kj_mol"] - m["energy_kj_mol"]) / (2 * H)
        an = base["forces"][label][axis]
        rows.append(dict(atom=label, index=base["atoms"][label], axis=axis, analytic=an, fd=fd, diff=an - fd,
                         second_diff=(p["energy_kj_mol"] + m["energy_kj_mol"] - 2 * base["energy_kj_mol"])))
summary = dict(frame=base["frame"], h_nm=H, tolerance_kj_mol_nm=0.5, rows=rows,
               max_abs_diff=max(abs(r["diff"]) for r in rows),
               force_sum=base["force_sum"], force_absmax=base["force_absmax"],
               min_face_margin_nm=base["min_face_margin_nm"], far_water_o_dist_nm=base["far_water_o_dist_nm"],
               base_timings=base["timings_row"], host=base["host"])
(HERE / "fd_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"{'atom':5s} {'ax':2s} {'analytic':>12s} {'fd':>12s} {'diff':>9s}")
for r in rows:
    print(f"{r['atom']:5s} {r['axis']:2d} {r['analytic']:12.4f} {r['fd']:12.4f} {r['diff']:+9.4f}")
print({k: summary[k] for k in ("max_abs_diff", "force_sum", "force_absmax", "min_face_margin_nm",
                               "far_water_o_dist_nm")})
print(summary["base_timings"])
