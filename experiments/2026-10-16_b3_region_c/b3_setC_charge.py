"""B3-1: net charge and charged groups that region C's 1.2 nm selection adds to region A's (A grouping)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-15_b2_crossing_events"))
from b2_events import setup  # noqa: E402

t09 = Path("../2026-10-14_t09_cutoff_scan")
topology, *_, parts, charges, groups, graph = setup(t09 / "run_files", "A")
res = {a.index: a.residue for a in topology.atoms()}
gq = np.array([charges[list(g)].sum() for g in groups])
label = [f"{res[g[0]].name}{res[g[0]].id}" for g in groups]
arr = np.load("../2026-10-15_b1_embedding_charge/b1_arrays.npz")
jobs = json.load(open("jobs_a_setC.json"))["jobs"]
extra_q, extra_n, charged = [], [], {}
for f, j in enumerate(jobs):
    s_a = set(arr[f"c1.200_f{f}"].tolist())
    s_c = set(j["groups"])
    assert s_a <= s_c, f
    add = sorted(s_c - s_a)
    extra_q.append(gq[add].sum())
    extra_n.append(len(add))
    for g in add:
        if abs(gq[g]) > 0.5:
            charged[label[g]] = charged.get(label[g], 0) + 1
print(f"groups added: mean {np.mean(extra_n):.0f}; net charge added: mean {np.mean(extra_q):+.2f} e, range [{min(extra_q):+.0f}, {max(extra_q):+.0f}]")
print(f"charged groups added (frames out of 20): {dict(sorted(charged.items()))}")
