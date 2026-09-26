"""Full-QM water NVE demo: OpenMM drives MD, ORCA computes the water molecule.

Run with the openmm_dev environment and OPI_ORCA set:

    export OPI_ORCA=/home/ruigengji/ORCA611
    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python examples/full_qm_water.py
"""

from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

from openmmorca import ORCAPotential


def water_topology_and_positions():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    oxygen = topology.addAtom("O", app.element.oxygen, residue)
    h1 = topology.addAtom("H1", app.element.hydrogen, residue)
    h2 = topology.addAtom("H2", app.element.hydrogen, residue)
    topology.addBond(oxygen, h1)
    topology.addBond(oxygen, h2)

    oh = 0.09572
    angle = 1.82421813418
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [oh, 0.0, 0.0],
            [oh * np.cos(angle), oh * np.sin(angle), 0.0],
        ]
    )
    return topology, positions


def main() -> None:
    topology, positions = water_topology_and_positions()

    potential = ORCAPotential(
        method="HF",
        basis="def2-SVP",
        extra_keywords=("TightSCF",),
    )
    system = potential.createSystem(topology)

    integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 42)

    print("step    pot(Eh)      pot(kJ/mol)   kin(kJ/mol)   total(kJ/mol)")
    for step in range(21):
        state = context.getState(getEnergy=True)
        pot = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kin = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        if step % 5 == 0:
            print(f"{step:3d}   {pot / 2625.4996394799:12.8f}   {pot:12.6f}   {kin:12.6f}   {pot + kin:12.6f}")
        if step < 20:
            integrator.step(1)

    print(f"\nscratch directory: {potential.backends[0].scratch.root}")
    potential.close()


if __name__ == "__main__":
    main()
