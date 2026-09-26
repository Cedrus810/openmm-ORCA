"""Small test systems (implementation plan Task 7).

Water geometry uses the exact TIP3P/TIP4P-Ew force-field equilibrium values
(O-H = 0.09572 nm, H-O-H = 1.82421813418 rad) so that harmonic bond/angle
energies vanish analytically at the helper geometry.
"""

from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.app as app

OH_NM = 0.09572
HOH_ANGLE_RAD = 1.82421813418


def water_topology(n_waters: int) -> app.Topology:
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(n_waters):
        residue = topology.addResidue("HOH", chain)
        oxygen = topology.addAtom("O", app.element.oxygen, residue)
        h1 = topology.addAtom("H1", app.element.hydrogen, residue)
        h2 = topology.addAtom("H2", app.element.hydrogen, residue)
        topology.addBond(oxygen, h1)
        topology.addBond(oxygen, h2)
    return topology


def water_positions(n_waters: int, spacing_nm: float = 0.3) -> np.ndarray:
    """Waters at force-field equilibrium geometry, laid out along x."""
    positions = np.zeros((3 * n_waters, 3))
    offset = np.array([spacing_nm, 0.0, 0.0])
    h1 = np.array([OH_NM, 0.0, 0.0])
    h2 = OH_NM * np.array([np.cos(HOH_ANGLE_RAD), np.sin(HOH_ANGLE_RAD), 0.0])
    for w in range(n_waters):
        base = w * offset
        positions[3 * w + 0] = base
        positions[3 * w + 1] = base + h1
        positions[3 * w + 2] = base + h2
    return positions


def flexible_tip3p_system(topology: app.Topology) -> mm.System:
    """Flexible TIP3P water, no cutoffs, no constraints."""
    forcefield = app.ForceField("tip3p.xml")
    return forcefield.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )


def tip4pew_system(topology: app.Topology, positions: np.ndarray):
    """TIP4P-Ew water with the M virtual sites added; returns (system, topology).

    The returned topology contains the extra particles and must be used for
    anything that indexes particles of the system.
    """
    forcefield = app.ForceField("tip4pew.xml")
    modeller = app.Modeller(topology, positions)
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    return system, modeller.topology
