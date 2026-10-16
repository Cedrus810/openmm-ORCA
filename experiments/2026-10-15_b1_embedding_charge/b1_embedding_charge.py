"""Roadmap B1: embedded charge composition of the T09 cutoff scan, and a classical test of the shell.

Reuses the T09 fixed-snapshot scan (experiments/2026-10-14_t09_cutoff_scan); no ORCA is run.

1. Consistency: rebuild the embedding exactly as ORCAPotential.createMixedSystem does (same
   prepared.pdb, force field, QM region A, charge shift) and re-run CutoffEmbedding.select on the
   same frames; the selected group count must equal the n_embed_groups the scan recorded.
2. Composition per (cutoff, frame): net embedded charge, number of charged groups (|q_group| > 0.5 e)
   by sign, Na+ count.
3. Energy/force split against the 2.5 nm reference: constant offset (mean over frames of
   ΔE = E(c) − E(2.5)), frame-varying part (std over frames of ΔE), RMS ΔF on QM atoms.
4. Classical test of the shell between c and 2.5 nm. The QM region is stood in for by its
   force-field charges (DCE + Asp124 side chain, Σ = −0.858 e) plus the remainder to the QM charge
   (−0.142 e) on the link H. If the cutoff sensitivity is far-field electrostatics, then
   ΔE ≈ −E_cl(shell) and ΔF_QM ≈ −F_cl(shell). E_cl and F_cl are split into charged and neutral
   groups to see which carries the frame-to-frame variation.

Usage: b1_embedding_charge.py T09_DIR OUTDIR
Writes OUTDIR/b1_results.json, OUTDIR/b1_arrays.npz (selected groups, QM-atom ΔF and classical shell
forces per cutoff and frame) and
prints Markdown tables.
"""
import json
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import enzyme_qmmm as enzyme  # noqa: E402

from openmmorca.qmmm import build_mixed_system  # noqa: E402
from openmmorca.qmmm.embedding import CutoffEmbedding, groups_from_topology  # noqa: E402
from openmmorca.qmmm.imaging import make_qm_whole, qm_bond_graph  # noqa: E402
from openmmorca.qmmm.linkatoms import LinkAtomManager, make_boundary_pairs  # noqa: E402
from openmmorca.units import COULOMB_KJ_MOL_NM  # noqa: E402

CHARGED = 0.5  # e; groups are whole residues/ions, so net charges are integers here


def load_scan(path):
    with np.load(path) as data:
        return {int(f): {k: data[f"{k}_{f}"] for k in ("energy_kj_mol", "forces", "n_embed_groups")}
                for f in data["frames_done"]}


def coulomb(qm_pos, qm_q, pos, q):
    """Energy (kJ/mol) and forces on the QM sites (kJ/mol/nm) from point charges pos, q."""
    if len(q) == 0:
        return 0.0, np.zeros_like(qm_pos)
    d = qm_pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    pair = COULOMB_KJ_MOL_NM * qm_q[:, None] * q[None, :] / r
    force = (pair / r ** 2)[:, :, None] * d
    return float(pair.sum()), force.sum(axis=1)


def main():
    t09, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((t09 / "scan" / "meta.json").read_text())
    frames = np.load(t09 / meta["frames_npz"])
    run_files = t09 / meta["run_files"]

    pdb = app.PDBFile(str(run_files / "prepared.pdb"))
    topology = pdb.topology
    state = mm.XmlSerializer.deserialize((run_files / "equilibrated.xml").read_text())
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    qm, pairs_idx, *_ = enzyme.qm_selection(topology, meta["region"])
    assert qm == meta["qm_atoms"] and [list(p) for p in pairs_idx] == meta["boundary_pairs"]
    pairs = make_boundary_pairs(topology, pairs_idx, None)
    parts = build_mixed_system(topology, system, qm, boundary_pairs=pairs)
    charges = np.zeros(system.getNumParticles())
    charges[list(parts.mm_atoms)] = parts.mm_charges_e
    groups = groups_from_topology(topology, parts.mm_atoms)
    graph = qm_bond_graph(topology, parts.qm_atoms, pairs)
    links = LinkAtomManager(pairs)

    residues = list(topology.residues())
    atom_res = {a.index: r for r in residues for a in r.atoms()}
    group_q = np.array([charges[list(g)].sum() for g in groups])
    group_label = [f"{atom_res[g[0]].name}{atom_res[g[0]].id}" for g in groups]
    group_is_na = np.array([atom_res[g[0]].name.upper() in ("NA", "NA+") for g in groups])
    atom_group = {i: k for k, g in enumerate(groups) for i in g}

    qm_q_ff = np.array([parts.removed_qm_charges[i] for i in parts.qm_atoms])
    link_q = (parts.qm_formal_charge - qm_q_ff.sum()) / links.n_links
    qm_q = np.concatenate([qm_q_ff, np.full(links.n_links, link_q)])

    scans = {float(p.stem[1:]): load_scan(p) for p in sorted((t09 / "scan").glob("c*.npz"))
             if not p.stem.endswith(".tmp")}
    cutoffs = sorted(scans)
    ref_cut = cutoffs[-1]
    frame_ids = sorted(set.intersection(*(set(s) for s in scans.values())))

    # Selection per (cutoff, frame).
    sel = {}  # (cut, f) -> sorted group indices
    placed = {}  # f -> {atom: position} from the reference selection
    qm_sites = {}
    mismatch = []
    for f in frame_ids:
        box = frames["box_nm"][f]
        pos = make_qm_whole(frames["positions_nm"][f], parts.qm_atoms, graph, box)
        qm_sites[f] = np.vstack([pos[list(parts.qm_atoms)], links.link_positions(pos)])
        for cut in cutoffs:
            emb = CutoffEmbedding(groups, charges, cut)
            atoms, positions, _ = emb.select(pos[list(parts.qm_atoms)], pos, box)
            gids = sorted({atom_group[int(a)] for a in atoms})
            sel[(cut, f)] = gids
            if len(gids) != int(scans[cut][f]["n_embed_groups"]):
                mismatch.append((cut, f, len(gids), int(scans[cut][f]["n_embed_groups"])))
            if cut == ref_cut:
                placed[f] = dict(zip(atoms.tolist(), positions))

    qm_rows = list(range(len(parts.qm_atoms)))
    pair0 = parts.boundary_pairs[0]
    q1_row = parts.qm_atoms.index(pair0.q1)

    report = {"reference_cutoff_nm": ref_cut, "frames": frame_ids, "selection_mismatches": mismatch,
              "qm_standin": {"ff_sum_e": float(qm_q_ff.sum()), "link_q_e": float(link_q),
                             "qm_formal_charge": parts.qm_formal_charge},
              "cutoffs": []}
    sel_arrays = {}
    for cut in cutoffs:
        rows = []
        for f in frame_ids:
            g_in = np.array(sel[(cut, f)], dtype=int)
            g_ref = np.array(sel[(ref_cut, f)], dtype=int)
            sel_arrays[f"c{cut:.3f}_f{f}"] = g_in
            assert set(g_in) <= set(g_ref)
            shell = np.setdiff1d(g_ref, g_in)
            q_in = group_q[g_in]
            e_parts, f_parts = {}, {}
            for name, mask in (("charged", np.abs(group_q[shell]) > CHARGED),
                               ("neutral", np.abs(group_q[shell]) <= CHARGED)):
                atoms = [i for g in shell[mask] for i in groups[g]]
                p = np.array([placed[f][i] for i in atoms]).reshape(-1, 3)
                e, fs = coulomb(qm_sites[f], qm_q, p, charges[atoms])
                # link H force -> Q1 with weight (1 - g); only QM rows are compared
                fq = fs[: len(parts.qm_atoms)].copy()
                fq[q1_row] += (1.0 - pair0.g) * fs[len(parts.qm_atoms)]
                e_parts[name], f_parts[name] = e, fq
            de = float(scans[cut][f]["energy_kj_mol"]) - float(scans[ref_cut][f]["energy_kj_mol"])
            df = scans[cut][f]["forces"][list(parts.qm_atoms)] - scans[ref_cut][f]["forces"][list(parts.qm_atoms)]
            sel_arrays[f"dF_c{cut:.3f}_f{f}"] = df
            sel_arrays[f"FclC_c{cut:.3f}_f{f}"] = f_parts["charged"]
            sel_arrays[f"FclN_c{cut:.3f}_f{f}"] = f_parts["neutral"]
            rows.append({
                "frame": f, "n_groups": len(g_in), "net_q": float(q_in.sum()),
                "n_pos": int((q_in > CHARGED).sum()), "n_neg": int((q_in < -CHARGED).sum()),
                "n_na": int(group_is_na[g_in].sum()),
                "shell_net_q": float(group_q[shell].sum()),
                "shell_charged": [group_label[g] for g in shell if abs(group_q[g]) > CHARGED],
                "dE": de, "Ecl_charged": e_parts["charged"], "Ecl_neutral": e_parts["neutral"],
                "dF": df, "Fcl_charged": f_parts["charged"], "Fcl_neutral": f_parts["neutral"],
            })
        de = np.array([r["dE"] for r in rows])
        pred = -np.array([r["Ecl_charged"] + r["Ecl_neutral"] for r in rows])
        pred_c = -np.array([r["Ecl_charged"] for r in rows])
        pred_n = -np.array([r["Ecl_neutral"] for r in rows])
        dF = np.concatenate([r["dF"] for r in rows])
        fc = -np.concatenate([r["Fcl_charged"] for r in rows])
        fn = -np.concatenate([r["Fcl_neutral"] for r in rows])

        def rms(x):
            return float(np.sqrt((np.asarray(x) ** 2).sum(axis=-1).mean())) if np.ndim(x) == 2 else float(np.sqrt(np.mean(np.square(x))))

        def corr(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None

        summary = {
            "cutoff_nm": cut,
            "n_groups_mean": float(np.mean([r["n_groups"] for r in rows])),
            "net_q_mean": float(np.mean([r["net_q"] for r in rows])),
            "net_q_min": float(min(r["net_q"] for r in rows)),
            "net_q_max": float(max(r["net_q"] for r in rows)),
            "n_pos_mean": float(np.mean([r["n_pos"] for r in rows])),
            "n_neg_mean": float(np.mean([r["n_neg"] for r in rows])),
            "n_na_mean": float(np.mean([r["n_na"] for r in rows])),
            "shell_net_q_mean": float(np.mean([r["shell_net_q"] for r in rows])),
            "dE_offset": float(de.mean()), "dE_frame_std": float(de.std()),
            "pred_offset": float(pred.mean()), "pred_frame_std": float(pred.std()),
            "pred_charged_frame_std": float(pred_c.std()), "pred_neutral_frame_std": float(pred_n.std()),
            "corr_dE_pred": corr(de, pred),
            "resid_frame_std": float((de - pred - (de - pred).mean()).std()),
            "dF_rms": rms(dF), "resid_F_rms": rms(dF - fc - fn),
            "Fcl_charged_rms": rms(fc), "Fcl_neutral_rms": rms(fn),
            "per_frame": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
                           if k not in ("dF", "Fcl_charged", "Fcl_neutral")} for r in rows],
        }
        report["cutoffs"].append(summary)

    np.savez_compressed(outdir / "b1_arrays.npz", **sel_arrays)
    (outdir / "b1_results.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"selection mismatches vs scan n_embed_groups: {len(mismatch)} {mismatch[:5]}")
    print(f"QM stand-in: FF charges sum {qm_q_ff.sum():+.3f} e, link H {link_q:+.3f} e\n")
    print("| cutoff (nm) | groups | net q mean [min, max] (e) | +groups | −groups | Na+ | shell net q (e) |")
    print("|---|---|---|---|---|---|---|")
    for s in report["cutoffs"]:
        print(f"| {s['cutoff_nm']:.2f} | {s['n_groups_mean']:.0f} | {s['net_q_mean']:+.2f} "
              f"[{s['net_q_min']:+.1f}, {s['net_q_max']:+.1f}] | {s['n_pos_mean']:.1f} | {s['n_neg_mean']:.1f} | "
              f"{s['n_na_mean']:.1f} | {s['shell_net_q_mean']:+.2f} |")
    print("\nEnergies kJ/mol, forces kJ/mol/nm; ΔX = X(c) − X(2.5 nm); pred = −classical(shell)\n")
    print("| cutoff | ΔE offset | pred offset | ΔE std | pred std | charged std | neutral std | corr | resid std "
          "| ΔF RMS | resid F RMS | F charged RMS | F neutral RMS |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in report["cutoffs"][:-1]:
        c = "—" if s["corr_dE_pred"] is None else f"{s['corr_dE_pred']:.3f}"
        print(f"| {s['cutoff_nm']:.2f} | {s['dE_offset']:.1f} | {s['pred_offset']:.1f} | {s['dE_frame_std']:.1f} | "
              f"{s['pred_frame_std']:.1f} | {s['pred_charged_frame_std']:.1f} | {s['pred_neutral_frame_std']:.1f} | "
              f"{c} | {s['resid_frame_std']:.1f} | {s['dF_rms']:.1f} | {s['resid_F_rms']:.1f} | "
              f"{s['Fcl_charged_rms']:.1f} | {s['Fcl_neutral_rms']:.1f} |")


if __name__ == "__main__":
    main()
