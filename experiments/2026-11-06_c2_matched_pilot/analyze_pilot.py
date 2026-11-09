"""Summarize completed C2 pilot results without inferring TS or ensemble acceptance."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openmmorca.units import HARTREE_IN_KJ_MOL, HARTREE_PER_BOHR_IN_KJ_MOL_NM

HERE = Path(__file__).resolve().parent


def projection(force, positions, qm_atoms, atoms):
    indices = {key: qm_atoms.index(value) for key, value in atoms.items()}
    c, cl, o = (indices[key] for key in ("C1", "Cl1", "O"))
    stretch = positions[c]-positions[cl]
    approach = positions[c]-positions[o]
    return {"stretch": float(.5*(force[c]-force[cl]) @ stretch / np.linalg.norm(stretch)),
            "approach": float(.5*(force[c]-force[o]) @ approach / np.linalg.norm(approach))}


def main():
    manifest = json.loads((HERE/"inputs/selection.json").read_text())
    candidates = manifest["pilot_indices"]
    result = dict(scope="three fixed geometries; no TS, ensemble, or barrier acceptance",
                  delta_e_target_kj_mol=5., frames=[], missing=[], cutoff_checks=[], native_reference=[])
    for f in candidates:
        row = dict(candidate=f, label=manifest["frames"][f]["label"],
                   rc_nm=manifest["frames"][f]["rc_nm"], pyscf={}, orca={})
        for engine, schemes in (("pyscf", ("S2", "ENV20", "S0", "ENV30", "EW20")),
                                ("orca", ("S2", "S0"))):
            for scheme in schemes:
                path = HERE/engine/f"candidate_{f:02d}_{scheme}.json"
                if not path.is_file():
                    if scheme not in ("ENV30", "EW20"):
                        result["missing"].append(str(path.relative_to(HERE)))
                    continue
                data = json.loads(path.read_text())
                row[engine][scheme] = data
        py = row["pyscf"]
        if "S2" in py and "ENV20" in py:
            if py["S2"]["qm_positions_nm"] != py["ENV20"]["qm_positions_nm"]:
                raise ValueError("mismatched QM geometries in embedding comparison")
            row["s2_minus_env20_kj_mol"] = (py["S2"]["energy_eh"]-py["ENV20"]["energy_eh"])*HARTREE_IN_KJ_MOL
        if "ENV20" in py and "ENV30" in py:
            result["cutoff_checks"].append(dict(candidate=f, env30_minus_env20_kj_mol=
                (py["ENV30"]["energy_eh"]-py["ENV20"]["energy_eh"])*HARTREE_IN_KJ_MOL))
        if "EW20" in py and "ENV20" in py:
            result["native_reference"].append(dict(candidate=f, native_minus_no_qm_images_kj_mol=
                (py["EW20"]["energy_eh"]-py["ENV20"]["energy_eh"])*HARTREE_IN_KJ_MOL))
        if all(s in row[engine] for engine in ("orca", "pyscf") for s in ("S0", "S2")):
            bridge = {}
            for engine in ("orca", "pyscf"):
                values = {}
                for scheme in ("S0", "S2"):
                    d = row[engine][scheme]
                    force = (np.array(d["qm_forces_kj_mol_nm"]) if engine == "orca" else
                             -np.array(d["qm_grad_eh_bohr"])*HARTREE_PER_BOHR_IN_KJ_MOL_NM)
                    values[scheme] = projection(force, np.array(d["qm_positions_nm"]),
                                                manifest["regions"]["A"]["qm_atoms"], manifest["atoms"])
                bridge[engine] = {key: values["S0"][key]-values["S2"][key] for key in ("stretch", "approach")}
                bridge[engine]["energy_shift_kj_mol"] = (row[engine]["S0"]["energy_kj_mol"]-
                    row[engine]["S2"]["energy_kj_mol"] if engine == "orca" else
                    (row[engine]["S0"]["energy_eh"]-row[engine]["S2"]["energy_eh"])*HARTREE_IN_KJ_MOL)
            row["bridge_s0_minus_s2"] = bridge
        result["frames"].append(row)
    baseline = next((r for r in result["frames"] if r["candidate"] == 0), None)
    if baseline and "s2_minus_env20_kj_mol" in baseline:
        reference = baseline["s2_minus_env20_kj_mol"]
        for row in result["frames"]:
            if "s2_minus_env20_kj_mol" in row:
                difference = row["s2_minus_env20_kj_mol"]-reference
                row["relative_embedding_shift_from_candidate_0_kj_mol"] = difference
                row["within_5_for_this_pair"] = abs(difference) <= 5.
                print(f"candidate {row['candidate']} RC={row['rc_nm']:+.4f}: "
                      f"relative embedding shift {difference:+.4f} kJ/mol")
        for row in result["frames"]:
            if "bridge_s0_minus_s2" in row and "bridge_s0_minus_s2" in baseline:
                b, r = row["bridge_s0_minus_s2"], baseline["bridge_s0_minus_s2"]
                row["relative_bridge_energy_residual_kj_mol"] = (
                    b["orca"]["energy_shift_kj_mol"]-r["orca"]["energy_shift_kj_mol"]-
                    b["pyscf"]["energy_shift_kj_mol"]+r["pyscf"]["energy_shift_kj_mol"])
    (HERE/"pilot_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("missing required results:", len(result["missing"]))
    for item in result["cutoff_checks"]:
        print("cutoff check", item)


if __name__ == "__main__":
    main()
