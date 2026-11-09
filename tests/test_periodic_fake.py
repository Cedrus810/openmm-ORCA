"""Periodic QM/MM with cutoff embedding through the OpenMM layer (Task 21, spec §11).

FakeBackend, Reference platform. QM–MM electrostatics beyond the embedding
cutoff are neglected by design; these tests check the plumbing (imaging,
selection, force scatter, charge shift, box handling), not that physics.
"""

from __future__ import annotations

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
from openmm import unit

from openmmorca.backend.fake import FakeBackend
from openmmorca.potential import ORCAPotential
from openmmorca.runtime.diagnostics import TIMING_FIELDS
from test_linkatoms import DIPEPTIDE_PDB, atom_index

OH = 0.09572
QM_WATER_FAKE_CHARGES = [-0.4, 0.2, 0.2]


def forcefield():
    return app.ForceField("amber14-all.xml", "amber14/tip3p.xml")


def water_box(box_nm: float = 2.5, method=app.PME, cutoff_nm: float = 1.0):
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield(), boxSize=mm.Vec3(box_nm, box_nm, box_nm) * unit.nanometer)
    system = forcefield().createSystem(
        modeller.topology,
        nonbondedMethod=method,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        rigidWater=False,
    )
    positions = modeller.getPositions().value_in_unit(unit.nanometer)
    return modeller.topology, system, np.array(positions)


def fake_potential(charges):
    charges = np.asarray(charges, dtype=float)
    return ORCAPotential("HF", basis="STO-3G", backend_factory=lambda: FakeBackend(charges))


def make_context(system, positions):
    context = mm.Context(
        system, mm.VerletIntegrator(0.0005), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions * unit.nanometer)
    return context


def box_of(system):
    return np.array(
        [v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()]
    )


def test_periodic_forces_consistent_without_crossings():
    topology, system, positions = water_box()
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(
        topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer
    )
    python_force = next(f for f in mixed.getForces() if isinstance(f, mm.PythonForce))
    assert python_force.usesPeriodicBoundaryConditions()
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)  # fixes the FakeBackend r0
    moved = positions.copy()
    moved[0] += np.array([0.004, -0.003, 0.002])  # distort the QM water

    # No group may sit within the finite-difference step of the cutoff, or the
    # selection would change between the two displaced evaluations.
    box = box_of(mixed)
    from openmmorca.qmmm.imaging import minimum_image

    rel = minimum_image(moved[3:] - moved[0], box)
    distances = np.linalg.norm(rel, axis=1)
    assert np.min(np.abs(distances - 1.0)) > 1e-3

    context.setPositions(moved * unit.nanometer)
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    request = potential.backends[0].last_request
    assert request.n_mm > 0  # the embedding is really in play
    h = 1e-5
    for axis in range(3):
        energies = []
        for sign in (1, -1):
            trial = moved.copy()
            trial[0, axis] += sign * h
            context.setPositions(trial * unit.nanometer)
            energies.append(
                context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
            )
        fd = (energies[0] - energies[1]) / (2 * h)
        assert abs(fd + forces[0, axis]) < 1e-2, (axis, fd, -forces[0, axis])
    potential.close()


def test_qm_split_across_boundary():
    topology, system, positions = water_box()
    box = box_of(system)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer)
    wrapped = positions.copy()
    wrapped[1] += box[0]  # QM H1 wrapped on its own to the other side of the box
    wrapped[2] -= box[1] + box[2]
    context = make_context(mixed, wrapped)
    context.getState(getEnergy=True)
    request = potential.backends[0].last_request
    qm = request.qm_positions_nm
    assert np.linalg.norm(qm[1] - qm[0]) == pytest.approx(
        np.linalg.norm(positions[1] - positions[0]), abs=1e-9
    )
    assert np.linalg.norm(qm[2] - qm[0]) == pytest.approx(
        np.linalg.norm(positions[2] - positions[0]), abs=1e-9
    )
    potential.close()


def test_embedding_counts_logged():
    topology, system, positions = water_box()
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer)
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)
    assert "n_embed_groups" in TIMING_FIELDS and "embed_changed" in TIMING_FIELDS
    diagnostics = potential.backends[0].last_request.diagnostics
    assert diagnostics["n_embed_groups"] > 0
    assert diagnostics["embed_changed"] == diagnostics["n_embed_groups"]  # first step: all enter
    context.getState(getEnergy=True)
    assert potential.backends[0].last_request.diagnostics["embed_changed"] == 0
    potential.close()


def test_cutoff_periodic_rejected():
    topology, system, _ = water_box(method=app.CutoffPeriodic)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    with pytest.raises(ValueError, match="CutoffPeriodic"):
        potential.createMixedSystem(topology, system, [0, 1, 2])
    potential.close()


def test_embedding_cutoff_checked_against_box():
    topology, system, _ = water_box()  # 2.5 nm box: half width 1.25 nm
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    with pytest.raises(ValueError, match="half"):
        potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=1.3 * unit.nanometer)
    potential.close()


def solvated_dipeptide():
    pdb = app.PDBFile(str(DIPEPTIDE_PDB))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield(), padding=1.2 * unit.nanometer)
    system = forcefield().createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=False,
    )
    positions = np.array(modeller.getPositions().value_in_unit(unit.nanometer))
    return modeller.topology, system, positions


def test_charge_shift_with_cutoff_embedding():
    topology, system, positions = solvated_dipeptide()
    qm = [atom_index(topology, "ALA", n) for n in ("CB", "HB1", "HB2", "HB3")]
    cb, ca = atom_index(topology, "ALA", "CB"), atom_index(topology, "ALA", "CA")
    m2 = [atom_index(topology, "ALA", n) for n in ("N", "HA", "C")]
    nonbonded = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charge = {
        i: nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in [ca] + m2
    }
    potential = fake_potential(np.full(len(qm) + 1, 0.05))
    mixed = potential.createMixedSystem(
        topology, system, qm, boundaryPairs=[(cb, ca)], embeddingCutoff=0.8 * unit.nanometer
    )
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)
    request = potential.backends[0].last_request
    assert request.qm_elements == ("C", "H", "H", "H", "H")  # methyl + link H
    # Backends see positions and charges only; find atoms by (minimum-image) position.
    from openmmorca.qmmm.imaging import minimum_image

    def embedded_rows(atom):
        rel = minimum_image(request.mm_positions_nm - positions[atom], box_of(mixed))
        return np.nonzero(np.linalg.norm(rel, axis=1) < 1e-9)[0]

    assert len(embedded_rows(ca)) == 0  # M1 is not embedded
    residual = sum(
        nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in qm
    )  # −0.0016 e, corrected on ALA's M2 (all three are ALA atoms)
    for atom in m2:  # M2 always selected, with the shifted + corrected charge
        (row,) = embedded_rows(atom)
        assert request.mm_charges_e[row] == pytest.approx(
            charge[atom] + charge[ca] / 3 + residual / 3, abs=1e-12
        )
    potential.close()


def test_barostat_extra_evaluations():
    # 3 nm box: the soft FakeBackend water stretches under 300 K Langevin, and
    # the per-step cutoff + QM-extent guard needs room (2.5 nm is too tight).
    topology, system, positions = water_box(3.0)
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 5))
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.0005)
    context = mm.Context(mixed, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    integrator.step(50)
    assert potential.backends[0].n_calls == 50 + 2 * 10
    potential.close()


def test_nonperiodic_unchanged():
    """Non-periodic systems take the old path: no imaging, every charge embedded."""
    import helpers

    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2])
    python_force = next(f for f in mixed.getForces() if isinstance(f, mm.PythonForce))
    assert not python_force.usesPeriodicBoundaryConditions()
    context = make_context(mixed, helpers.water_positions(2))
    context.getState(getEnergy=True)
    request = potential.backends[0].last_request
    assert request.n_mm == 3 and request.diagnostics == {}
    potential.close()


# Whole-box minimum-image embedding (embeddingCutoff=None, roadmap B4 scheme S2).


def total_energy(context, positions):
    context.setPositions(positions * unit.nanometer)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )


def test_full_box_embeds_all_mm_charges():
    topology, system, positions = water_box()
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=None)
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)
    request = potential.backends[0].last_request
    n_waters = topology.getNumResidues()
    assert request.n_mm == 3 * (n_waters - 1)
    assert request.mm_charges_e.sum() == pytest.approx(0.0, abs=1e-9)
    diagnostics = request.diagnostics
    assert diagnostics["n_embed_groups"] == n_waters - 1
    assert diagnostics["embed_changed"] == n_waters - 1  # first step: all enter
    # Every embedded atom sits within half a cell of the QM anchor, per axis.
    rel = request.mm_positions_nm - request.qm_positions_nm[0]
    assert np.all(np.abs(rel) < 0.5 * 2.5 + 0.2)  # + group extent around the face
    potential.close()


def test_full_box_finite_difference():
    topology, system, positions = water_box()
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=None)
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)  # fixes the FakeBackend r0
    backend = potential.backends[0]
    moved = positions.copy()
    moved[0] += np.array([0.004, -0.003, 0.002])  # distort the QM water
    context.setPositions(moved * unit.nanometer)
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    mm_o = 3 * 7  # an MM water oxygen
    h = 1e-5
    for atom in (0, mm_o):
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                trial = moved.copy()
                trial[atom, axis] += sign * h
                energies.append(total_energy(context, trial))
                # No group may switch image between the displaced evaluations.
                assert backend.last_request.diagnostics["embed_changed"] == 0
            fd = (energies[0] - energies[1]) / (2 * h)
            assert abs(fd + forces[atom, axis]) < 1e-2, (atom, axis, fd, -forces[atom, axis])
    potential.close()


def test_full_box_lattice_invariance():
    topology, system, positions = water_box()
    box = box_of(system)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=None)
    context = make_context(mixed, positions)
    e0 = total_energy(context, positions)  # also fixes the FakeBackend r0
    backend = potential.backends[0]
    shifted = positions + box[0] - box[1]  # whole system by a lattice vector
    wrapped = positions.copy()
    wrapped[1] += box[2]  # one QM H on its own
    wrapped[30:33] -= box[0]  # one MM water on its own
    for trial in (shifted, wrapped):
        assert total_energy(context, trial) == pytest.approx(e0, abs=1e-6)
        assert backend.last_request.diagnostics["embed_changed"] == 0
    potential.close()


def test_full_box_image_switch():
    """A water crossing the face of the anchor-centred cell switches image.

    The group stays embedded (total charge unchanged) and is counted in
    embed_changed; its placement jumps by one lattice vector.
    """
    topology, system, positions = water_box()
    box = box_of(system)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=None)
    context = make_context(mixed, positions)
    backend = potential.backends[0]
    from openmmorca.qmmm.imaging import minimum_image

    # Put MM water 5 rigidly just inside the +x face relative to the QM O.
    water = slice(15, 18)
    rel = minimum_image(positions[water] - positions[0], box)
    trial = positions.copy()
    trial[water] = positions[0] + rel - rel[0] + np.array([1.20, 0.0, 0.0])
    # Keep every atom inside the face (half width 1.25 nm) so the right image is unambiguous.
    trial[water, 0] -= max(0.0, np.max(trial[water, 0] - positions[0, 0]) - 1.22)
    total_energy(context, trial)
    before = backend.last_request
    n_mm = before.n_mm
    trial2 = trial.copy()
    trial2[water] += np.array([0.12, 0.0, 0.0])  # O now beyond the face: nearest via the left image
    total_energy(context, trial2)
    after = backend.last_request
    assert after.n_mm == n_mm
    assert after.diagnostics["embed_changed"] == 1
    np.testing.assert_allclose(after.mm_charges_e.sum(), before.mm_charges_e.sum(), atol=1e-12)
    (rows,) = np.nonzero(
        np.linalg.norm(
            minimum_image(after.mm_positions_nm - trial2[15], box), axis=1
        ) < 1e-9
    )
    assert after.mm_positions_nm[rows[0], 0] - after.qm_positions_nm[0, 0] < 0  # left image now
    potential.close()


# ------------------------------------------------------------------
# S3: whole-box embedding + classical Ewald correction (V3 plan Task 3)
# ------------------------------------------------------------------


def s3_water_box(charges=QM_WATER_FAKE_CHARGES):
    topology, system, positions = water_box()
    potential = fake_potential(charges)
    mixed = potential.createMixedSystem(
        topology, system, [0, 1, 2], embeddingCutoff=None, ewaldCorrection=True, forceGroup=1
    )
    return potential, mixed, positions


def test_s3_requires_whole_box():
    topology, system, positions = water_box()
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    with pytest.raises(ValueError, match="embeddingCutoff=None"):
        potential.createMixedSystem(
            topology, system, [0, 1, 2], embeddingCutoff=1.0 * unit.nanometer, ewaldCorrection=True
        )
    import helpers

    top = helpers.water_topology(2)
    with pytest.raises(ValueError, match="periodic"):
        potential.createMixedSystem(
            top, helpers.flexible_tip3p_system(top), [0, 1, 2], embeddingCutoff=None, ewaldCorrection=True
        )


def test_s3_request_contents():
    potential, mixed, positions = s3_water_box()
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)
    request = potential.backends[0].last_request
    n_real = 3 * (len(positions) // 3 - 1)
    assert request.n_mm == n_real + 50
    np.testing.assert_allclose(request.esp_points_nm, request.mm_positions_nm[n_real:])
    assert request.diagnostics["s3_q_absmax"] > 0
    assert request.diagnostics["s3_qm_extent_nm"] < 0.1  # one water
    assert request.diagnostics["t_ewald"] > 0
    potential.close()


def test_s3_off_by_default():
    topology, system, positions = water_box()
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], embeddingCutoff=None)
    make_context(mixed, positions).getState(getEnergy=True)
    request = potential.backends[0].last_request
    assert request.esp_points_nm is None
    assert request.n_mm == 3 * (len(positions) // 3 - 1)
    assert "t_ewald" not in request.diagnostics
    potential.close()


def test_s3_energy_matches_classical_ewald():
    """E_S3 - E_S2 for the fake QM charges = sum_a q_a dV(R_a), dV = Ewald - whole-box (independent sum).

    Residual budget: the pseudo charges reproduce the 2nd-order polynomial to ~0.1 kJ/mol/e RMS in
    the template ball (Task 0 §1), and the Taylor truncation over a 0.1 nm water is far smaller;
    with sum |q_a| = 1.4 e this gives the 0.1 kJ/mol tolerance. A net fake QM charge makes the
    correction sizeable (a neutral QM water gives only ~0.4 kJ/mol here).
    """
    from test_ewald_correction import reference_dV

    charges = [-1.0, 0.2, 0.2]
    potential, mixed, positions = s3_water_box(charges)
    context = make_context(mixed, positions)
    e_s3 = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    request = potential.backends[0].last_request
    n_real = request.n_mm - 50
    fake = FakeBackend(np.array(charges))
    from openmmorca.backend.base import QMRequest

    e_s2 = fake.evaluate(
        QMRequest(qm_elements=request.qm_elements, qm_positions_nm=request.qm_positions_nm,
                  mm_positions_nm=request.mm_positions_nm[:n_real],
                  mm_charges_e=request.mm_charges_e[:n_real])
    ).energy_kj_mol
    dV = reference_dV(request.qm_positions_nm, request.mm_positions_nm[:n_real],
                      request.mm_charges_e[:n_real], box_of(mixed))
    expected = float(np.dot(charges, dV))
    assert abs(expected) > 1.0  # the correction is not trivially small here
    assert (e_s3 - e_s2) == pytest.approx(expected, abs=0.1)
    potential.close()


def test_s3_finite_difference_with_link():
    """QM, link Q1/M1/M2, a near and the farthest MM atom: forces = -dE/dx (fake QM, S3 on)."""
    topology, system, positions = solvated_dipeptide()
    qm = [atom_index(topology, "ALA", n) for n in ("CB", "HB1", "HB2", "HB3")]
    cb, ca = atom_index(topology, "ALA", "CB"), atom_index(topology, "ALA", "CA")
    n_atom = atom_index(topology, "ALA", "N")
    potential = fake_potential([-0.3, 0.1, 0.1, 0.1, 0.05])
    mixed = potential.createMixedSystem(
        topology, system, qm, boundaryPairs=[(cb, ca)], embeddingCutoff=None, ewaldCorrection=True
    )
    context = make_context(mixed, positions)
    context.getState(getEnergy=True)  # fixes the FakeBackend r0
    backend = potential.backends[0]
    moved = positions.copy()
    moved[qm[1]] += np.array([0.003, -0.002, 0.002])
    context.setPositions(moved * unit.nanometer)
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    from openmmorca.qmmm.imaging import minimum_image

    rel = np.linalg.norm(minimum_image(moved - moved[cb], box_of(mixed)), axis=1)
    waters = [r for r in topology.residues() if r.name == "HOH"]
    oxygens = [next(a.index for a in r.atoms() if a.name == "O") for r in waters]
    near_o = min(oxygens, key=lambda i: rel[i])
    far_o = max(oxygens, key=lambda i: rel[i])
    h = 1e-5
    for atom in (cb, qm[1], ca, n_atom, near_o, far_o):
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                trial = moved.copy()
                trial[atom, axis] += sign * h
                energies.append(total_energy(context, trial))
                assert backend.last_request.diagnostics["embed_changed"] == 0
            fd = (energies[0] - energies[1]) / (2 * h)
            assert abs(fd + forces[atom, axis]) < 1e-2, (atom, axis, fd, -forces[atom, axis])
    potential.close()


def test_s3_momentum_and_lattice_invariance():
    potential, mixed, positions = s3_water_box()
    box = box_of(mixed)
    context = make_context(mixed, positions)
    e0 = total_energy(context, positions)
    f = context.getState(getForces=True, groups={1}).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    assert np.linalg.norm(f.sum(axis=0)) < 1e-6 * np.abs(f).max()
    backend = potential.backends[0]
    shifted = positions + box[0] - box[1]
    wrapped = positions.copy()
    wrapped[30:33] -= box[0]
    for trial in (shifted, wrapped):
        assert total_energy(context, trial) == pytest.approx(e0, abs=1e-6)
        assert backend.last_request.diagnostics["embed_changed"] == 0
    potential.close()


def _image_switch_jump(ewald_correction):
    """Energy change across the point where MM water 5 switches image (both sides 1e-6 nm apart)."""
    topology, system, positions = water_box()
    box = box_of(system)
    potential = fake_potential(QM_WATER_FAKE_CHARGES)
    mixed = potential.createMixedSystem(
        topology, system, [0, 1, 2], embeddingCutoff=None, ewaldCorrection=ewald_correction
    )
    context = make_context(mixed, positions)
    backend = potential.backends[0]
    from openmmorca.qmmm.imaging import minimum_image

    water = slice(15, 18)
    rel = minimum_image(positions[water] - positions[0], box)
    base = positions.copy()
    base[water] = positions[0] + rel - rel[0] + np.array([1.20, 0.0, 0.0])
    base[water, 0] -= max(0.0, np.max(base[water, 0] - positions[0, 0]) - 1.22)

    def switched(s):
        total_energy(context, base)
        trial = base.copy()
        trial[water] += np.array([s, 0.0, 0.0])
        e = total_energy(context, trial)
        return backend.last_request.diagnostics["embed_changed"] == 1, e

    lo, hi = 0.0, 0.12
    assert not switched(lo)[0] and switched(hi)[0]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if switched(mid)[0] else (mid, hi)
    jump = switched(hi)[1] - switched(lo)[1]
    potential.close()
    return jump, hi - lo


def test_s3_image_switch_is_smaller_than_s2():
    jump_s2, width = _image_switch_jump(False)
    jump_s3, _ = _image_switch_jump(True)
    assert width < 1e-9
    assert abs(jump_s3) < abs(jump_s2), (jump_s2, jump_s3)
