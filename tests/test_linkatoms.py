"""Tests for H link atoms at covalent QM/MM boundaries (Task 16, spec §10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
from openmm import unit

from openmmorca.backend.fake import FakeBackend
from openmmorca.qmmm.linkatoms import (
    BoundaryPair,
    LinkAtomManager,
    check_boundary_pairs,
    default_link_ratio,
)
from openmmorca.qmmm.system import check_whole_molecules
from openmmorca.potential import ORCAPotential

DIPEPTIDE_PDB = Path(__file__).parent / "data" / "ace_ala_nme.pdb"


def dipeptide():
    """ACE-ALA-NME with amber14 in vacuum (topology, system, positions nm)."""
    pdb = app.PDBFile(str(DIPEPTIDE_PDB))
    forcefield = app.ForceField("amber14-all.xml")
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None
    )
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return pdb.topology, system, np.array(positions)


def atom_index(topology, residue_name, atom_name):
    for atom in topology.atoms():
        if atom.residue.name == residue_name and atom.name == atom_name:
            return atom.index
    raise KeyError((residue_name, atom_name))


def side_chain(topology):
    """QM = ALA methyl (CB, HB1-3); boundary CB-CA."""
    qm = [atom_index(topology, "ALA", n) for n in ("CB", "HB1", "HB2", "HB3")]
    pairs = [(atom_index(topology, "ALA", "CB"), atom_index(topology, "ALA", "CA"))]
    return qm, pairs


def two_fragments(topology):
    """QM = ALA methyl + NME methyl; boundaries CB-CA (C-C) and NME C-N (C-N)."""
    qm = [atom_index(topology, "ALA", n) for n in ("CB", "HB1", "HB2", "HB3")]
    qm += [atom_index(topology, "NME", n) for n in ("C", "H1", "H2", "H3")]
    pairs = [
        (atom_index(topology, "ALA", "CB"), atom_index(topology, "ALA", "CA")),
        (atom_index(topology, "NME", "C"), atom_index(topology, "NME", "N")),
    ]
    return qm, pairs


def fake_potential(n_fake_charges: int):
    fake_charges = np.linspace(-0.05, 0.05, n_fake_charges)
    return ORCAPotential(
        "HF", basis="def2-SVP", backend_factory=lambda: FakeBackend(fake_charges)
    )


def total_energy(context, positions):
    context.setPositions(positions * unit.nanometer)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )


def kicked(positions, atoms, seed=7):
    """Deterministic small displacement of *atoms*, so fake bond terms are active."""
    rng = np.random.default_rng(seed)
    out = positions.copy()
    out[atoms] += rng.normal(scale=0.003, size=(len(atoms), 3))
    return out


def check_fd(qm, pairs, topology, system, positions, atoms_to_check):
    potential = fake_potential(len(qm) + len(pairs))
    mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs)
    context = mm.Context(
        mixed, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    total_energy(context, positions)  # fixes the FakeBackend r0 at the start geometry
    moved = kicked(positions, list(qm) + [m1 for _, m1 in pairs])
    context.setPositions(moved * unit.nanometer)
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    h = 1e-5
    for particle in atoms_to_check:
        for axis in range(3):
            plus, minus = moved.copy(), moved.copy()
            plus[particle, axis] += h
            minus[particle, axis] -= h
            fd = (total_energy(context, plus) - total_energy(context, minus)) / (2 * h)
            assert abs(fd + forces[particle, axis]) < 1e-3, (
                f"particle {particle} axis {axis}: fd {fd}, force {-forces[particle, axis]}"
            )
    # The fake QM part must actually see the link atoms.
    backend = potential.backends[0]
    assert backend.last_request.n_qm == len(qm) + len(pairs)
    assert backend.last_request.qm_elements[len(qm):] == ("H",) * len(pairs)
    potential.close()


# ---------------------------------------------------------------------------


def test_link_position_formula():
    manager = LinkAtomManager([BoundaryPair(q1=0, m1=1, g=0.7)])
    positions = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    np.testing.assert_allclose(manager.link_positions(positions), [[0.105, 0.0, 0.0]])
    assert manager.n_links == 1


def test_redistribute_chain_rule():
    manager = LinkAtomManager([BoundaryPair(q1=0, m1=1, g=0.7)])
    forces = np.zeros((2, 3))
    manager.redistribute(forces, np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(forces[0], [0.3, 0.6, 0.9])
    np.testing.assert_allclose(forces[1], [0.7, 1.4, 2.1])


def test_fd_with_link_atoms_fake_backend():
    topology, system, positions = dipeptide()
    qm, pairs = side_chain(topology)
    q1, m1 = pairs[0]
    check_fd(qm, pairs, topology, system, positions, [q1, m1])


def test_fd_two_links_disconnected_qm():
    topology, system, positions = dipeptide()
    qm, pairs = two_fragments(topology)
    atoms = [q1 for q1, _ in pairs] + [m1 for _, m1 in pairs]
    check_fd(qm, pairs, topology, system, positions, atoms)


def test_boundary_bonded_terms_kept():
    topology, system, _ = dipeptide()
    qm, pairs = side_chain(topology)
    potential = fake_potential(len(qm) + 1)
    mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs)
    q1, m1 = pairs[0]
    qm_set = set(qm)
    kept_bond = kept_angle = kept_torsion = False
    for force in mixed.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, _, k = force.getBondParameters(i)
                atoms = {p1, p2}
                if atoms == {q1, m1}:
                    assert k.value_in_unit(k.unit) > 0
                    kept_bond = True
                elif atoms <= qm_set:
                    assert k.value_in_unit(k.unit) == 0
        elif isinstance(force, mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, p2, p3, _, k = force.getAngleParameters(i)
                atoms = {p1, p2, p3}
                if p2 == q1 and m1 in atoms:  # HB-CB-CA
                    assert k.value_in_unit(k.unit) > 0
                    kept_angle = True
                elif atoms <= qm_set:
                    assert k.value_in_unit(k.unit) == 0
        elif isinstance(force, mm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                *particles, _, _, k = force.getTorsionParameters(i)
                if q1 in particles and m1 in particles and not set(particles) <= qm_set:
                    if k.value_in_unit(k.unit) != 0:
                        kept_torsion = True
    assert kept_bond and kept_angle and kept_torsion
    potential.close()


def test_boundary_validation():
    topology, _, _ = dipeptide()
    qm, pairs = side_chain(topology)
    ca = atom_index(topology, "ALA", "CA")
    cb = atom_index(topology, "ALA", "CB")
    n = atom_index(topology, "ALA", "N")
    hb1 = atom_index(topology, "ALA", "HB1")

    def pair(q1, m1):
        return BoundaryPair(q1=q1, m1=m1, g=0.7)

    with pytest.raises(ValueError, match="not in the QM region"):
        check_boundary_pairs(topology, qm, [pair(ca, cb)])  # q1 is MM
    with pytest.raises(ValueError, match="is in the QM region"):
        check_boundary_pairs(topology, qm, [pair(cb, hb1)])  # m1 is QM
    with pytest.raises(ValueError, match="not bonded"):
        check_boundary_pairs(topology, qm, [pair(cb, n)])
    with pytest.raises(ValueError, match="more than one boundary"):
        check_boundary_pairs(topology, qm + [ca, atom_index(topology, "ALA", "HA")],
                             [pair(ca, n), pair(ca, atom_index(topology, "ALA", "C"))])
    with pytest.raises(ValueError, match="cuts a covalent bond"):
        check_boundary_pairs(topology, qm, [])  # CB-CA cut but not declared
    with pytest.raises(ValueError, match="g must be in"):
        BoundaryPair(q1=cb, m1=ca, g=1.2)
    check_boundary_pairs(topology, qm, [pair(*pairs[0])])  # the valid case passes


def test_default_ratio_unknown_pair():
    assert default_link_ratio("C", "C") == pytest.approx(1.09 / 1.526)
    assert default_link_ratio("C", "N") == pytest.approx(1.09 / 1.449)
    with pytest.raises(ValueError, match="no default link ratio"):
        default_link_ratio("O", "S")


def test_explicit_link_ratios():
    topology, system, _ = dipeptide()
    qm, pairs = side_chain(topology)
    potential = fake_potential(len(qm) + 1)
    with pytest.raises(ValueError, match="linkRatios"):
        potential.createMixedSystem(
            topology, system, qm, boundaryPairs=pairs, linkRatios=[0.7, 0.7]
        )
    potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs, linkRatios=[0.7])
    potential.close()


def test_link_h_counted_in_electrons(tmp_path):
    from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend

    topology, system, positions = dipeptide()
    qm, pairs = side_chain(topology)
    potential = fake_potential(len(qm) + 1)
    mixed = potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs)
    context = mm.Context(
        mixed, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions * unit.nanometer)
    context.getState(getEnergy=True)
    elements = potential.backends[0].last_request.qm_elements
    assert elements == ("C", "H", "H", "H", "H")  # methyl + link H = methane
    potential.close()

    for multiplicity, ok in ((1, True), (2, False)):
        backend = ORCAOPIBackend(
            ORCAConfig(method="HF", multiplicity=multiplicity, scratch_root=str(tmp_path)),
            check_version=False,
        )
        if ok:
            backend._check_electron_count(elements)
        else:
            with pytest.raises(ValueError, match="incompatible"):
                backend._check_electron_count(elements)
        backend.close()


def test_oniom_unaffected():
    """Without allowed_bonds, check_whole_molecules keeps rejecting cut molecules."""
    topology, _, _ = dipeptide()
    qm, pairs = side_chain(topology)
    with pytest.raises(ValueError, match="cuts a covalent bond"):
        check_whole_molecules(topology, qm)
    check_whole_molecules(topology, qm, allowed_bonds=pairs)

