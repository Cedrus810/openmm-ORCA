"""CUDA vs Reference for the QM/MM force path (T07): link atoms, periodic imaging, virtual sites.

FakeBackend; the PythonForce lives in force group 1, so the comparison of that
group isolates this package's code path (positions handed to the callback,
imaging, link-atom and virtual-site force redistribution) from differences
between the platforms' own MM kernels, which are compared more loosely.
CUDA runs in double precision. Skipped when no CUDA device is usable.
"""

from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
from openmm import unit

import helpers
from openmmorca.backend.fake import FakeBackend
from openmmorca.potential import ORCAPotential
from test_linkatoms import dipeptide, side_chain
from test_periodic_fake import QM_WATER_FAKE_CHARGES, water_box

QM_GROUP = 1
KJ_NM = unit.kilojoule_per_mole / unit.nanometer


def cuda_platform():
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        system = mm.System()
        system.addParticle(1.0)
        mm.Context(system, mm.VerletIntegrator(0.001), platform, {"Precision": "double"})
    except Exception as exc:  # no plugin, no device or driver mismatch
        pytest.skip(f"CUDA platform unavailable: {exc}")
    return platform


def fake_potential(charges):
    charges = np.asarray(charges, dtype=float)
    return ORCAPotential("HF", basis="STO-3G", backend_factory=lambda: FakeBackend(charges))


def link_atom_case():
    topology, system, positions = dipeptide()
    qm, pairs = side_chain(topology)
    mixed = fake_potential(np.linspace(-0.05, 0.05, len(qm) + 1)).createMixedSystem(
        topology, system, qm, boundaryPairs=pairs, forceGroup=QM_GROUP,
    )
    return mixed, positions


def periodic_case():
    topology, system, positions = water_box(2.5)
    box = np.array([v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()])
    # Put the QM water across the +x face, so its hydrogens need imaging.
    positions = positions + np.array([box[0, 0] - 0.02 - positions[0, 0], 0.0, 0.0])
    mixed = fake_potential(QM_WATER_FAKE_CHARGES).createMixedSystem(
        topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer, forceGroup=QM_GROUP,
    )
    return mixed, positions


def virtual_site_case():
    topology = helpers.water_topology(3)
    water = helpers.water_positions(3)
    system, extended = helpers.tip4pew_system(topology, water * unit.nanometer)
    modeller = app.Modeller(topology, water * unit.nanometer)
    modeller.addExtraParticles(app.ForceField("tip4pew.xml"))
    positions = np.array(modeller.getPositions().value_in_unit(unit.nanometer))
    qm = [a.index for a in extended.atoms() if a.residue.index == 0 and a.element is not None]
    mixed = fake_potential([-0.4, 0.2, 0.2]).createMixedSystem(
        extended, system, qm, forceGroup=QM_GROUP,
    )
    return mixed, positions


def evaluate(system, positions, platform, properties=None):
    context = mm.Context(system, mm.VerletIntegrator(0.0005), platform, properties or {})
    context.setPositions(positions * unit.nanometer)
    context.computeVirtualSites()
    result = {}
    for label, groups in (("qm", {QM_GROUP}), ("total", -1)):
        state = context.getState(getEnergy=True, getForces=True, groups=groups)
        result[label] = (
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(KJ_NM),
        )
    return result


@pytest.mark.parametrize("case", [link_atom_case, periodic_case, virtual_site_case],
                         ids=["link_atoms", "periodic_imaging", "virtual_sites"])
def test_cuda_matches_reference(case):
    cuda = cuda_platform()
    system, positions = case()
    for force in system.getForces():
        if force.getForceGroup() != QM_GROUP:
            force.setForceGroup(0)
    reference = evaluate(system, positions, mm.Platform.getPlatformByName("Reference"))
    gpu = evaluate(system, positions, cuda, {"Precision": "double"})

    energy, forces = reference["qm"]
    scale = np.abs(forces).max()
    assert scale > 0
    assert gpu["qm"][0] == pytest.approx(energy, rel=1e-10, abs=1e-8)
    np.testing.assert_allclose(gpu["qm"][1], forces, rtol=0, atol=1e-9 * scale)

    # Whole system, including the platforms' own MM kernels. Measured on an
    # RTX 2080 Ti / OpenMM 8.5.2: energy up to 5.8e-6 relative (PME) and
    # forces up to 3.5e-6 of the largest force; the QM group above agrees to
    # better than 1e-12.
    energy, forces = reference["total"]
    assert gpu["total"][0] == pytest.approx(energy, rel=2e-5)
    np.testing.assert_allclose(gpu["total"][1], forces, rtol=0, atol=2e-5 * np.abs(forces).max())
