"""C2 (contract §5): windowed restrained QM/MM MD along RC = d(C1–Cl1) − d(O–C1), region A, S2 embedding.

Generates configurations from reactant to product for matched single points; the path itself is not a
result (contract §5, §7). Restraint: 0.5 k (RC − rc0)², minimum-image distances, O = Asp124 oxygen nearest
C1 in the start state.

--mode pull: windows in sequence, each --pull-steps from the previous window's end; writes
  OUTDIR/w<i>.start.xml (State) per window.
--mode prod --windows I ...: per window, from w<i>.start.xml: --eq-steps, then --prod-steps saving a frame
  every --save-every steps -> OUTDIR/w<i>.npz (positions nm float32, box, step, RC, d_OC, d_CCl, angle;
  rc0, k) and w<i>.end.xml. Independent windows can run in parallel.
Both modes skip finished windows (resumable). --smoke N: N steps in the first window, print timing only.

Usage: rc_scan.py RUN_DIR OUTDIR --mode pull|prod [--windows I ...] [--method XTB] [--nprocs 1]
       [--rc-start -0.10] [--rc-end 0.20] [--rc-step 0.02] [--k 20000] [--pull-steps 200]
       [--eq-steps 400] [--prod-steps 1000] [--save-every 50] [--smoke N]
RUN_DIR holds prepared.pdb and final_state.xml (start positions, velocities, box).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import enzyme_qmmm as enzyme  # noqa: E402

from openmmorca import ORCAPotential  # noqa: E402
from openmmorca.qmmm.imaging import minimum_image  # noqa: E402

QM_GROUP = 31
RESTRAINT_GROUP = 30


def geometry(pos, box, o, c1, cl1):
    v_o = minimum_image(pos[o] - pos[c1], box)
    v_cl = minimum_image(pos[cl1] - pos[c1], box)
    d_oc, d_ccl = np.linalg.norm(v_o), np.linalg.norm(v_cl)
    angle = np.degrees(np.arccos(v_o @ v_cl / (d_oc * d_ccl)))
    return d_ccl - d_oc, d_oc, d_ccl, angle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("--method", default="XTB")
    p.add_argument("--nprocs", type=int, default=1)
    p.add_argument("--rc-start", type=float, default=-0.10)
    p.add_argument("--rc-end", type=float, default=0.20)
    p.add_argument("--rc-step", type=float, default=0.02)
    p.add_argument("--k", type=float, default=20000.0, help="kJ/mol/nm^2")
    p.add_argument("--mode", choices=("pull", "prod"), default="pull")
    p.add_argument("--windows", type=int, nargs="*", default=None)
    p.add_argument("--pull-steps", type=int, default=200)
    p.add_argument("--eq-steps", type=int, default=400)
    p.add_argument("--prod-steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--platform", default="CUDA")
    p.add_argument("--smoke", type=int, default=0)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pdb = app.PDBFile(str(args.run_dir / "prepared.pdb"))
    topology = pdb.topology
    start = mm.XmlSerializer.deserialize((args.run_dir / "final_state.xml").read_text())
    system = enzyme.mm_system(topology, enzyme.forcefield_with_dce())
    system.setDefaultPeriodicBoxVectors(*start.getPeriodicBoxVectors())
    topology.setPeriodicBoxVectors(start.getPeriodicBoxVectors())
    qm, pairs, asp_o, c1, cl1 = enzyme.qm_selection(topology, "A")
    pos0 = start.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box0 = np.array(start.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
    o = min(asp_o, key=lambda i: np.linalg.norm(minimum_image(pos0[i] - pos0[c1], box0)))
    windows = np.round(np.arange(args.rc_start, args.rc_end + 0.5 * args.rc_step, args.rc_step), 6)
    meta = {"run_dir": str(args.run_dir), "method": args.method, "nprocs": args.nprocs, "region": "A",
            "embedding": "whole box (S2)", "atoms": {"O": int(o), "C1": int(c1), "Cl1": int(cl1)},
            "k_kj_mol_nm2": args.k, "windows": windows.tolist(), "pull_steps": args.pull_steps, "eq_steps": args.eq_steps,
            "prod_steps": args.prod_steps, "save_every": args.save_every, "step_fs": enzyme.QMMM_RECIPE["step_fs"],
            "start_geometry": dict(zip(("rc", "d_oc", "d_ccl", "angle"), map(float, geometry(pos0, box0, o, c1, cl1))))}
    if not args.smoke and args.mode == "pull":
        (args.outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["start_geometry"]), "nucleophile", [a for a in topology.atoms()][o], flush=True)

    recipe = enzyme.QMMM_RECIPE
    potential = ORCAPotential(method=args.method, charge=recipe["charge"], multiplicity=recipe["multiplicity"],
                              nprocs=args.nprocs)
    try:
        mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs, forceGroup=QM_GROUP,
                                            embeddingCutoff=None)
        restraint = mm.CustomCompoundBondForce(3, "0.5*k_rc*(distance(p2,p3)-distance(p1,p2)-rc0)^2")
        restraint.addGlobalParameter("k_rc", args.k)
        restraint.addGlobalParameter("rc0", float(windows[0]))
        restraint.addBond([int(o), int(c1), int(cl1)], [])
        restraint.setUsesPeriodicBoundaryConditions(True)
        restraint.setForceGroup(RESTRAINT_GROUP)
        mixed.addForce(restraint)
        integrator = mm.LangevinMiddleIntegrator(recipe["temperature_K"] * unit.kelvin,
                                                 recipe["friction_per_ps"] / unit.picosecond,
                                                 recipe["step_fs"] * unit.femtosecond)
        integrator.setRandomNumberSeed(recipe["integrator_seed"])
        context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName(args.platform))
        # OpenMM 8.5.2: setPositions before the first PythonForce evaluation (spec §2.8), then the State.
        context.setPositions(start.getPositions())
        context.setState(start)

        if args.smoke:
            t0 = time.perf_counter()
            for n in range(1, args.smoke + 1):
                integrator.step(1)
                s = context.getState(getPositions=True)
                g = geometry(s.getPositions(asNumpy=True).value_in_unit(unit.nanometer), box0, o, c1, cl1)
                print(f"smoke step {n}: wall {time.perf_counter() - t0:.2f} s cumulative, RC {g[0]:+.4f}", flush=True)
            rows = potential.backends[0].timings.rows
            print("t_orca per call:", [round(r["t_orca"], 2) for r in rows], flush=True)
            return

        if args.mode == "pull":
            for i, rc0 in enumerate(windows):
                state_file = args.outdir / f"w{i:02d}.start.xml"
                if state_file.is_file():
                    context.setState(mm.XmlSerializer.deserialize(state_file.read_text()))
                    continue
                context.setParameter("rc0", float(rc0))
                t0 = time.perf_counter()
                integrator.step(args.pull_steps)
                s = context.getState(getPositions=True, getVelocities=True)
                g = geometry(s.getPositions(asNumpy=True).value_in_unit(unit.nanometer), box0, o, c1, cl1)
                tmp = state_file.with_suffix(".tmp")
                tmp.write_text(mm.XmlSerializer.serialize(s))
                tmp.replace(state_file)
                print(f"pull window {i} rc0 {rc0:+.3f}: RC {g[0]:+.4f}, d_OC {g[1]:.3f}, d_CCl {g[2]:.3f}, "
                      f"angle {g[3]:.1f}, wall {time.perf_counter() - t0:.0f} s", flush=True)
            return

        for i in (args.windows if args.windows is not None else range(len(windows))):
            rc0 = windows[i]
            out = args.outdir / f"w{i:02d}.npz"
            if out.is_file():
                continue
            context.setState(mm.XmlSerializer.deserialize((args.outdir / f"w{i:02d}.start.xml").read_text()))
            context.setParameter("rc0", float(rc0))
            t0 = time.perf_counter()
            integrator.step(args.eq_steps)
            frames = {"positions_nm": [], "box_nm": [], "step": [], "rc": [], "d_oc": [], "d_ccl": [], "angle": []}
            for n in range(args.save_every, args.prod_steps + 1, args.save_every):
                integrator.step(args.save_every)
                s = context.getState(getPositions=True)
                pos = s.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                box = np.array(s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
                g = geometry(pos, box, o, c1, cl1)
                frames["positions_nm"].append(pos.astype(np.float32))
                frames["box_nm"].append(box)
                frames["step"].append(args.eq_steps + n)
                for key, val in zip(("rc", "d_oc", "d_ccl", "angle"), g):
                    frames[key].append(val)
            final = context.getState(getPositions=True, getVelocities=True)
            (args.outdir / f"w{i:02d}.end.xml").write_text(mm.XmlSerializer.serialize(final))
            np.savez(out.with_suffix(".tmp.npz"), rc0=rc0, k=args.k, **{k: np.array(v) for k, v in frames.items()})
            out.with_suffix(".tmp.npz").replace(out)
            rc = np.array(frames["rc"])
            print(f"window {i} rc0 {rc0:+.3f}: RC {rc.mean():+.4f} ± {rc.std():.4f}, d_OC {np.mean(frames['d_oc']):.3f}, "
                  f"d_CCl {np.mean(frames['d_ccl']):.3f}, angle {np.mean(frames['angle']):.1f}, "
                  f"wall {time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        potential.close()


if __name__ == "__main__":
    main()
