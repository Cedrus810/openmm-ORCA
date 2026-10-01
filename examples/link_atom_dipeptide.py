"""Link-atom QM/MM demo: ACE-ALA-NME with the ALA methyl side chain in ORCA (Task 18).

QM = ALA CB, HB1-3, cut from the backbone at CB–CA; the H link atom turns the
QM region into methane (charge 0, singlet). HF/def2-SVP TightSCF, amber14 for
the rest, vacuum, no constraints. Minimize on the QM/MM surface, then 200
steps of Langevin NVT (300 K, 0.5 fs), checking that the boundary bond stays
intact.

    export OPI_ORCA=/home/ruigengji/ORCA611
    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python examples/link_atom_dipeptide.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

from openmmorca import ORCAPotential

PDB = Path(__file__).resolve().parent.parent / "tests" / "data" / "ace_ala_nme.pdb"


def atom_index(topology, residue_name, atom_name):
    for atom in topology.atoms():
        if atom.residue.name == residue_name and atom.name == atom_name:
            return atom.index
    raise KeyError((residue_name, atom_name))


def run(n_steps: int = 200):
    pdb = app.PDBFile(str(PDB))
    topology = pdb.topology
    system = app.ForceField("amber14-all.xml").createSystem(
        topology, nonbondedMethod=app.NoCutoff, constraints=None
    )
    cb = atom_index(topology, "ALA", "CB")
    ca = atom_index(topology, "ALA", "CA")
    qm = [cb] + [atom_index(topology, "ALA", n) for n in ("HB1", "HB2", "HB3")]

    potential = ORCAPotential(method="HF", basis="def2-SVP", extra_keywords=("TightSCF",))
    mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=[(cb, ca)])

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.5 * unit.femtosecond
    )
    context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    t0 = time.perf_counter()
    mm.LocalEnergyMinimizer.minimize(context)
    print(f"minimization         : {time.perf_counter() - t0:.1f} s")
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)

    distances = []
    step_times = []
    for step in range(n_steps):
        t_step = time.perf_counter()
        integrator.step(1)
        step_times.append(time.perf_counter() - t_step)
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        positions = positions.value_in_unit(unit.nanometer)
        distances.append(float(np.linalg.norm(positions[ca] - positions[cb])))
        if step % 20 == 0:
            print(f"step {step:4d}: CA–CB {distances[-1]:.4f} nm, {step_times[-1] * 1000:.0f} ms")

    distances = np.array(distances)
    step_times = np.array(step_times[1:])  # skip the cold first step
    print(f"steps                : {n_steps}")
    print(f"ms/step (mean, p95)  : {step_times.mean() * 1000:.0f}, {np.percentile(step_times, 95) * 1000:.0f}")
    print(f"CA–CB range          : {distances.min():.4f}–{distances.max():.4f} nm")
    for summary in potential.summarize_timings():
        print(f"  backend {summary['scratch']}: t_orca mean {summary['t_orca']['mean']:.2f} s")
    potential.close()
    intact = bool(np.all((distances > 0.14) & (distances < 0.17)))
    print(f"boundary bond intact : {intact}")
    return intact


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
