"""Tests for boundary embedding charges (Task 17, spec §10.4)."""

from __future__ import annotations

import warnings

import numpy as np
import openmm as mm
import pytest
from openmm import unit

from openmmorca.qmmm import build_mixed_system
from openmmorca.qmmm.charges import ChargeShift
from openmmorca.qmmm.linkatoms import BoundaryPair, make_boundary_pairs
from openmmorca.qmmm.system import get_nonbonded_force, particle_charges
from test_linkatoms import atom_index, dipeptide, side_chain


def side_chain_parts(**kwargs):
    topology, system, _ = dipeptide()
    qm, pairs = side_chain(topology)
    boundary = make_boundary_pairs(topology, pairs)
    parts = build_mixed_system(topology, system, qm, boundary_pairs=boundary, **kwargs)
    return topology, system, qm, boundary, parts


def test_total_embedding_charge_integer():
    """Charge shift conserves charge; the residue correction makes the total integer."""
    topology, system, qm, _, parts = side_chain_parts()
    charges = particle_charges(get_nonbonded_force(system))
    mm_total = sum(charges[p] for p in range(system.getNumParticles()) if p not in qm)
    assert mm_total == pytest.approx(0.0016, abs=1e-4)  # neutral dipeptide minus −0.0016 e
    assert parts.mm_charges_e.sum() == pytest.approx(0.0, abs=1e-12)
    assert parts.qm_formal_charge == 0


def test_m1_removed_and_m2_shifted():
    topology, system, qm, boundary, parts = side_chain_parts()
    charges = particle_charges(get_nonbonded_force(system))
    m1 = boundary[0].m1  # ALA CA
    m2 = [atom_index(topology, "ALA", n) for n in ("N", "HA", "C")]
    assert m1 not in parts.mm_atoms
    embedded = dict(zip(parts.mm_atoms, parts.mm_charges_e))
    residual = sum(charges[p] for p in qm)  # −0.0016 e, rounds to 0
    for atom in m2:  # shifted M1 charge + this residue's share of the residual
        assert embedded[atom] == pytest.approx(
            charges[atom] + charges[m1] / 3 + residual / 3, abs=1e-12
        )
    # Every other MM atom keeps its force-field charge.
    for atom, charge in embedded.items():
        if atom not in m2:
            assert charge == pytest.approx(charges[atom], abs=1e-12)


def test_m1_without_mm_neighbours():
    topology, system, _ = dipeptide()
    ha = atom_index(topology, "ALA", "HA")
    ca = atom_index(topology, "ALA", "CA")
    qm = [a.index for a in topology.atoms() if a.index != ha]  # HA is the only MM atom
    boundary = make_boundary_pairs(topology, [(ca, ha)], ratios=[0.7])
    with pytest.raises(ValueError, match="no MM neighbours"):
        build_mixed_system(topology, system, qm, boundary_pairs=boundary)


def test_zero_charge_m2_kept():
    """An M2 whose charge becomes 0 — or was 0 before the shift — stays embedded."""
    topology, _, _ = dipeptide()
    ca = atom_index(topology, "ALA", "CA")
    cb = atom_index(topology, "ALA", "CB")
    n, ha, c = (atom_index(topology, "ALA", name) for name in ("N", "HA", "C"))
    pair = BoundaryPair(q1=cb, m1=ca, g=0.7)
    # M1 carries +0.3 → each of N, HA, C receives +0.1. N ends at exactly 0,
    # HA starts at 0 (so it is not in the input embedding at all).
    mm_atoms = (n, ca, c)
    mm_charges = np.array([-0.1, 0.3, 0.5])
    atoms, charges = ChargeShift().embedding_charges(topology, mm_atoms, mm_charges, [pair])
    embedded = dict(zip(atoms, charges))
    assert ca not in embedded
    assert embedded[n] == pytest.approx(0.0, abs=1e-15)
    assert embedded[ha] == pytest.approx(0.1)
    assert embedded[c] == pytest.approx(0.6)
    assert list(atoms) == sorted(atoms)
    assert charges.sum() == pytest.approx(mm_charges.sum())


def test_openmm_mm_electrostatics_unchanged():
    """Charge shift changes only the ORCA embedding, never the OpenMM MM–MM terms."""
    topology, system, qm, _, parts = side_chain_parts()
    original = get_nonbonded_force(system)
    modified = get_nonbonded_force(parts.system)
    qm_set = set(qm)
    for p in range(system.getNumParticles()):
        if p in qm_set:
            continue
        assert modified.getParticleParameters(p)[0] == original.getParticleParameters(p)[0]
    original_exceptions = {
        tuple(sorted(original.getExceptionParameters(i)[:2])): original.getExceptionParameters(i)[2]
        for i in range(original.getNumExceptions())
    }
    for i in range(modified.getNumExceptions()):
        p1, p2, charge_prod, _, _ = modified.getExceptionParameters(i)
        if p1 in qm_set or p2 in qm_set:
            continue
        assert charge_prod == original_exceptions[tuple(sorted((p1, p2)))]

    # And the energy of an MM-only configuration agrees: with every QM charge
    # zeroed in both Systems, the NonbondedForce energies must be identical.
    reference = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    ref_nb = get_nonbonded_force(reference)
    for p in qm:
        charge, sigma, epsilon = ref_nb.getParticleParameters(p)
        ref_nb.setParticleParameters(p, 0.0, sigma, 0.0)
    for i in range(ref_nb.getNumExceptions()):
        p1, p2, _, sigma, epsilon = ref_nb.getExceptionParameters(i)
        if p1 in qm_set or p2 in qm_set:
            ref_nb.setExceptionParameters(i, p1, p2, 0.0, sigma, 0.0)
    mod_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(parts.system))
    mod_nb = get_nonbonded_force(mod_system)
    for p in qm:
        charge, sigma, _ = mod_nb.getParticleParameters(p)
        mod_nb.setParticleParameters(p, 0.0, sigma, 0.0)
    for i in range(mod_nb.getNumExceptions()):
        p1, p2, _, sigma, _ = mod_nb.getExceptionParameters(i)
        if p1 in qm_set or p2 in qm_set:
            mod_nb.setExceptionParameters(i, p1, p2, 0.0, sigma, 0.0)
    _, _, positions = dipeptide()
    energies = []
    for sys_ in (reference, mod_system):
        for force in sys_.getForces():
            force.setForceGroup(1 if isinstance(force, mm.NonbondedForce) else 0)
        context = mm.Context(
            sys_, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(positions * unit.nanometer)
        energies.append(
            context.getState(getEnergy=True, groups={1})
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    assert energies[0] == pytest.approx(energies[1], abs=1e-9)


def test_residue_residual_moved_to_its_own_m2():
    """QM = ALA CA, HA, CB, HB1-3, C, O (force-field charge +0.1438 e).

    Boundaries CA–N(ALA) and C(ALA)–N(NME). ALA's residual goes to the M2 atoms
    *in ALA* only — here ALA H (the other M2 of the N boundary is ACE C, and
    both M2 of the NME-N boundary are NME atoms).
    """
    topology, system, _ = dipeptide()
    names = ("CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O")
    qm = [atom_index(topology, "ALA", n) for n in names]
    n_ala, h_ala = atom_index(topology, "ALA", "N"), atom_index(topology, "ALA", "H")
    c_ace = atom_index(topology, "ACE", "C")
    boundary = make_boundary_pairs(
        topology,
        [
            (atom_index(topology, "ALA", "CA"), n_ala),
            (atom_index(topology, "ALA", "C"), atom_index(topology, "NME", "N")),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean residual: no warning
        parts = build_mixed_system(topology, system, qm, boundary_pairs=boundary)
    charges = particle_charges(get_nonbonded_force(system))
    residual = sum(charges[p] for p in qm)
    assert residual == pytest.approx(0.1438, abs=1e-4)
    embedded = dict(zip(parts.mm_atoms, parts.mm_charges_e))
    assert embedded[h_ala] == pytest.approx(
        charges[h_ala] + charges[n_ala] / 2 + residual, abs=1e-12
    )
    assert embedded[c_ace] == pytest.approx(charges[c_ace] + charges[n_ala] / 2, abs=1e-12)
    assert parts.mm_charges_e.sum() == pytest.approx(0.0, abs=1e-12)
    assert parts.qm_formal_charge == 0


def test_ambiguous_residual_warns():
    """A cut residue whose QM part is near a half-integer charge cannot be rounded safely."""
    from openmmorca.qmmm.charges import residue_charge_corrections

    topology, _, _ = dipeptide()
    qm, pairs = side_chain(topology)
    boundary = make_boundary_pairs(topology, pairs)
    charges = np.zeros(topology.getNumAtoms())
    charges[qm[0]] = 0.45  # QM part of ALA carries +0.45 e
    with pytest.warns(UserWarning, match="ambiguous"):
        residue_charge_corrections(topology, set(qm), charges, boundary)


def test_formal_charge_mismatch_warns():
    """createMixedSystem warns when the user's QM charge disagrees with the force field."""
    from openmmorca.backend.fake import FakeBackend
    from openmmorca.potential import ORCAPotential

    topology, system, _ = dipeptide()
    qm, pairs = side_chain(topology)
    potential = ORCAPotential(
        "HF", charge=-1, backend_factory=lambda: FakeBackend(np.zeros(len(qm) + 1))
    )
    with pytest.warns(UserWarning, match="charge -1.*force field.*0"):
        potential.createMixedSystem(topology, system, qm, boundaryPairs=pairs)
    potential.close()
