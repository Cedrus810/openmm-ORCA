"""Select nine geometry-based C2 candidates and capture three matched QM requests.

No QM or dynamics is run. Output directories must be new. Frame indices are
zero-based; the three pilot indices are 0 (R), 4 (crossing), and 8 (P-like).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


def image(vector, box):
    vector = np.array(vector, dtype=float)
    for axis in (2, 1, 0):
        vector -= np.round(vector[..., axis] / box[axis, axis])[..., None] * box[axis]
    return vector


def distance(pos, box, i, j):
    return float(np.linalg.norm(image(pos[i] - pos[j], box)))


def select_frames(scan):
    """Bracket three rare central-RC frames without labelling any frame a TS."""
    with np.load(scan / "w05.npz", allow_pickle=False) as w5:
        pre = int(np.argmax(w5["rc"]))
    with np.load(scan / "w07.npz", allow_pickle=False) as w7:
        post = int(np.argmin(w7["rc"]))
    with np.load(scan / "w06.npz", allow_pickle=False) as w6:
        central = []
        for low, high in ((0, .02), (.02, .04), (.04, .06)):
            hits = np.flatnonzero((w6["rc"] >= low) & (w6["rc"] < high))
            if len(hits) != 1:
                raise ValueError(f"expected one saved central frame in [{low}, {high}), got {len(hits)}")
            central.append(int(hits[0]))
    return [(0, 0, "R_first"), (0, 19, "R_last"), (5, pre, "pre_crossing"),
            (6, central[0], "crossing_low"), (6, central[1], "crossing_mid"),
            (6, central[2], "crossing_high"), (7, post, "post_crossing"),
            (15, 0, "P_like_first"), (15, 19, "P_like_last")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=ROOT / "experiments/2026-11-01_c2_rc_scan/scan")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"preserve existing output: {args.outdir}")

    import openmm as mm
    import openmm.app as app
    from openmm import unit
    import enzyme_qmmm as enzyme
    from openmmorca import ORCAPotential
    from openmmorca.backend.base import QMResult

    class Capture:
        request = None

        def evaluate(self, request):
            self.request = request
            return QMResult(0., np.zeros((request.n_qm, 3)), np.zeros((request.n_mm, 3)))

        def close(self):
            pass

    meta = json.loads((args.scan / "meta.json").read_text())
    run = Path(meta["run_dir"])
    pdb = app.PDBFile(str(run / "prepared.pdb"))
    topology = pdb.topology
    atoms = list(topology.atoms())
    o, c, cl = (meta["atoms"][k] for k in ("O", "C1", "Cl1"))
    if [atoms[i].name for i in (o, c, cl)] != ["OD2", "C1", "CL1"]:
        raise ValueError("scan atom names do not match topology")
    lookup = {(int(a.residue.id), a.name): a.index for a in atoms if a.residue.chain.index == 0}
    contacts = {f"Cl1_TRP{r}_{name}_nm": lookup[r, name]
                for r in (125, 175) for name in ("NE1", "HE1")}
    selections = select_frames(args.scan)
    arrays = {k: [] for k in ("positions_nm", "box_nm", "step")}
    rows = []
    for candidate, (window, frame, label) in enumerate(selections):
        source = args.scan / f"w{window:02d}.npz"
        with np.load(source, allow_pickle=False) as data:
            pos = data["positions_nm"][frame].astype(float)
            box = data["box_nm"][frame]
            if pos.shape != (len(atoms), 3) or not np.isfinite(pos).all():
                raise ValueError(f"invalid coordinates: {source}:{frame}")
            oc, ccl = distance(pos, box, o, c), distance(pos, box, c, cl)
            if abs(ccl - oc - float(data["rc"][frame])) > 1e-5:
                raise ValueError("recomputed RC disagrees with saved geometry")
            v1, v2 = image(pos[o] - pos[c], box), image(pos[cl] - pos[c], box)
            angle = float(np.degrees(np.arccos(np.clip(v1 @ v2 / (oc * ccl), -1, 1))))
            row = dict(candidate=candidate, label=label, source=str(source.resolve()),
                       window=window, frame=frame, step=int(data["step"][frame]),
                       rc_nm=ccl-oc, d_oc_nm=oc, d_ccl_nm=ccl, angle_deg=angle,
                       d_C1_OD1_nm=distance(pos, box, c, lookup[124, "OD1"]),
                       d_AspCG_OD2_nm=distance(pos, box, lookup[124, "CG"], o))
            row.update({key: distance(pos, box, cl, atom) for key, atom in contacts.items()})
            rows.append(row)
            arrays["positions_nm"].append(pos)
            arrays["box_nm"].append(box)
            arrays["step"].append(row["step"])
    arrays = {key: np.array(value) for key, value in arrays.items()}
    regions = {}
    for region in ("A", "C"):
        qm, pairs, *_ = enzyme.qm_selection(topology, region)
        regions[region] = dict(qm_atoms=qm, qm_names=[atoms[i].name for i in qm],
                               qm_elements=[atoms[i].element.symbol for i in qm], boundary_pairs=pairs)
    if not set(regions["A"]["qm_atoms"]).issubset(regions["C"]["qm_atoms"]):
        raise ValueError("A/C common atom mapping mismatch")
    manifest = dict(scan=str(args.scan.resolve()), run_files=str(run), n_atoms=len(atoms),
                    selection="geometry only; no TS or ensemble designation", pilot_indices=[0, 4, 8],
                    atoms=meta["atoms"], regions=regions, frames=rows)
    args.outdir.mkdir(parents=True)
    (args.outdir / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(args.outdir / "selected_frames.npz", **arrays)
    print("selected 9 frames; pilot = candidates 0, 4, 8", flush=True)

    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    box = arrays["box_nm"][0]
    system.setDefaultPeriodicBoxVectors(*box)
    topology.setPeriodicBoxVectors(box * unit.nanometer)
    qm, pairs, *_ = enzyme.qm_selection(topology, "A")
    requests = dict(box_nm=arrays["box_nm"], step=arrays["step"],
                    qm_atoms=np.array(qm), qm_names=np.array(regions["A"]["qm_names"]))
    for scheme, cutoff in (("S0", 1.2 * unit.nanometer), ("S2", None)):
        capture = Capture()
        with_potential = ORCAPotential(method="r2SCAN-3c", charge=-1, backend_factory=lambda: capture)
        try:
            mixed = with_potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs,
                                                     forceGroup=31, embeddingCutoff=cutoff)
            integrator = mm.VerletIntegrator(.0005)
            context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName("CPU"))
            for f in manifest["pilot_indices"]:
                context.setPeriodicBoxVectors(*arrays["box_nm"][f])
                context.setPositions(arrays["positions_nm"][f])
                context.getState(getEnergy=True, groups={31})
                r = capture.request
                if r is None or not np.isfinite(r.mm_positions_nm).all():
                    raise ValueError("invalid captured request")
                requests[f"{scheme}_qm_pos_{f}"] = r.qm_positions_nm
                requests[f"{scheme}_mm_pos_{f}"] = r.mm_positions_nm
                requests[f"{scheme}_mm_q_{f}"] = r.mm_charges_e
                requests["qm_elements"] = np.array(r.qm_elements)
                print(f"captured {scheme} candidate {f}: QM={r.n_qm}, MM={r.n_mm}, "
                      f"MM charge={r.mm_charges_e.sum():+.6f}", flush=True)
            del context, integrator
        finally:
            with_potential.close()
    for f in manifest["pilot_indices"]:
        if not np.array_equal(requests[f"S0_qm_pos_{f}"], requests[f"S2_qm_pos_{f}"]):
            raise ValueError("S0/S2 QM geometries differ")
    np.savez_compressed(args.outdir / "requests_A.npz", **requests)
    print("request capture complete; no QM evaluated", flush=True)


if __name__ == "__main__":
    main()
