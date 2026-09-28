"""Tests for the two-layer ONIOM scheme (plan: docs/plans/2026-09-28-oniom.md)."""

from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from openmm import unit

import helpers
from openmmorca.backend.base import QMRequest
from openmmorca.backend.fake import FakeBackend
from openmmorca.oniom import ONIOMCallback, ONIOMPotential
from openmmorca.potential import ORCAPotential

N_WATERS = 3
N_PARTICLES = 3 * N_WATERS
MODEL = (0, 1, 2)
ELEMENTS = ("O", "H", "H") * N_WATERS
HIGH_K = 1000.0
LOW_K = 300.0


class FlexibleFake(FakeBackend):
    """FakeBackend that adapts its (zero) embedding charges to any request size.

    With zero fake charges the embedding term vanishes, so the same class can
    serve the 3-atom model requests and the 9-atom full-system requests.
    """

    def __init__(self, k_bond: float = 1000.0) -> None:
        super().__init__(qm_charges_e=[0.0])
        self.k_bond = float(k_bond)
        self.n_closes = 0

    def evaluate(self, request: QMRequest):
        self.qm_charges_e = np.zeros(request.n_qm)
        return super().evaluate(request)

    def close(self) -> None:
        self.n_closes += 1


def perturbed_positions() -> np.ndarray:
    """Equilibrium waters with a deterministic kick, so energies/forces are nonzero.

    The kick distorts the model water internally (atom 1), which activates the
    high-minus-low correction on the model atoms, and displaces a low-only atom
    (atom 4), which activates the full-system low-level force there.
    """
    positions = helpers.water_positions(N_WATERS) + np.array([0.004, -0.003, 0.002])
    positions[1, 1] += 0.01
    positions[4, 1] += 0.01
    return positions


def build_oniom(model=MODEL, n_waters=N_WATERS):
    """ONIOMPotential with FlexibleFake backends (high k=1000, low k=300)."""
    topology = helpers.water_topology(n_waters)
    high = ORCAPotential(
        "HF", basis="def2-SVP", backend_factory=lambda: FlexibleFake(HIGH_K)
    )
    low = ORCAPotential("XTB", backend_factory=lambda: FlexibleFake(LOW_K))
    oniom = ONIOMPotential(high=high, low=low)
    system = oniom.createONIOMSystem(topology, list(model), forceGroup=1)
    return topology, oniom, system


def make_context(system, positions):
    context = mm.Context(
        system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions * unit.nanometer)
    return context


def warmed_context(system, positions):
    """Context whose FakeBackend r0 was fixed at the equilibrium geometry.

    A FakeBackend records its equilibrium distances on the first call; without
    this warmup the first evaluation geometry *is* the equilibrium geometry and
    every energy/force is identically zero (the assertions would pass vacuously).
    """
    context = make_context(system, helpers.water_positions(N_WATERS))
    context.getState(getEnergy=True)
    context.setPositions(positions * unit.nanometer)
    return context


def layer_backends(oniom: ONIOMPotential):
    """(high, low_full, low_model) in creation order."""
    assert len(oniom.high.backends) == 1
    assert len(oniom.low.backends) == 2
    return oniom.high.backends[0], oniom.low.backends[0], oniom.low.backends[1]


def reference_values(oniom, model=MODEL):
    """Re-evaluate each backend on its recorded request and combine."""
    high, low_full, low_model = layer_backends(oniom)
    e_high = high.evaluate(high.last_request)
    e_full = low_full.evaluate(low_full.last_request)
    e_model_low = low_model.evaluate(low_model.last_request)
    energy = (
        e_high.energy_kj_mol + e_full.energy_kj_mol - e_model_low.energy_kj_mol
    )
    forces = e_full.qm_forces_kj_mol_nm.copy()
    forces[list(model)] += e_high.qm_forces_kj_mol_nm - e_model_low.qm_forces_kj_mol_nm
    return energy, forces


def test_oniom_energy_arithmetic():
    _, oniom, system = build_oniom()
    context = warmed_context(system, perturbed_positions())
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    expected, _ = reference_values(oniom)
    assert energy == pytest.approx(expected, abs=1e-10)
    oniom.close()


def test_oniom_forces_combination():
    _, oniom, system = build_oniom()
    context = warmed_context(system, perturbed_positions())
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    _, expected = reference_values(oniom)
    np.testing.assert_allclose(forces, expected, atol=1e-10)

    # Spot-check the physics: a low-only atom carries pure low-level full-system
    # force; a model atom carries the high-minus-low correction on top.
    _, low_full, _ = layer_backends(oniom)
    full_forces = low_full.evaluate(low_full.last_request).qm_forces_kj_mol_nm
    np.testing.assert_allclose(forces[5], full_forces[5], atol=1e-10)
    assert not np.allclose(forces[0], full_forces[0])
    oniom.close()


def test_oniom_forces_are_exact_gradient():
    _, oniom, system = build_oniom()
    context = warmed_context(system, perturbed_positions())
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    def total_energy(pos):
        context.setPositions(pos * unit.nanometer)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    positions = perturbed_positions()
    h = 1e-5
    for particle, axis in ((0, 0), (2, 1), (4, 2)):  # model O, model H, low-only H
        plus, minus = positions.copy(), positions.copy()
        plus[particle, axis] += h
        minus[particle, axis] -= h
        fd = (total_energy(plus) - total_energy(minus)) / (2 * h)
        assert abs(fd - (-forces[particle, axis])) < 1e-3, (
            f"particle {particle} axis {axis}: fd={fd} force={-forces[particle, axis]}"
        )
    oniom.close()


def test_oniom_momentum_conservation():
    _, oniom, system = build_oniom()
    context = warmed_context(system, perturbed_positions())
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    assert np.linalg.norm(forces.sum(axis=0)) < 1e-9
    oniom.close()


def test_create_oniom_system_structure():
    topology, oniom, system = build_oniom()
    assert system.getNumParticles() == N_PARTICLES
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(3)]
    assert masses[0] == pytest.approx(15.999, abs=0.01)
    assert masses[1] == pytest.approx(1.008, abs=0.01)
    force_types = [type(f).__name__ for f in system.getForces()]
    assert "PythonForce" in force_types
    assert "CMMotionRemover" in force_types
    python_force = next(f for f in system.getForces() if type(f).__name__ == "PythonForce")
    assert python_force.getForceGroup() == 1
    assert python_force.getName() == "ORCA ONIOM"
    oniom.close()


def test_whole_molecules_required():
    topology = helpers.water_topology(2)
    high = ORCAPotential("HF", basis="def2-SVP", backend_factory=FlexibleFake)
    low = ORCAPotential("XTB", backend_factory=FlexibleFake)
    oniom = ONIOMPotential(high=high, low=low)
    with pytest.raises(ValueError, match="cuts a covalent bond"):
        oniom.createONIOMSystem(topology, [0, 1])  # O-H of water 0, no O
    oniom.close()


def test_atom_selection_validation():
    topology = helpers.water_topology(2)
    high = ORCAPotential("HF", basis="def2-SVP", backend_factory=FlexibleFake)
    low = ORCAPotential("XTB", backend_factory=FlexibleFake)
    oniom = ONIOMPotential(high=high, low=low)
    with pytest.raises(ValueError, match="must not be empty"):
        oniom.createONIOMSystem(topology, [])
    with pytest.raises(ValueError, match="duplicates"):
        oniom.createONIOMSystem(topology, [0, 0, 1])
    with pytest.raises(ValueError, match=r"must be in \[0, 6\)"):
        oniom.createONIOMSystem(topology, [0, 6])
    oniom.close()


def test_layer_type_check():
    with pytest.raises(TypeError, match="high must be an ORCAPotential"):
        ONIOMPotential(high="not a potential", low=ORCAPotential("XTB"))
    with pytest.raises(TypeError, match="low must be an ORCAPotential"):
        ONIOMPotential(high=ORCAPotential("XTB"), low=None)


def test_model_covers_all_warns():
    topology = helpers.water_topology(1)
    high = ORCAPotential("HF", basis="def2-SVP", backend_factory=FlexibleFake)
    low = ORCAPotential("XTB", backend_factory=FlexibleFake)
    oniom = ONIOMPotential(high=high, low=low)
    with pytest.warns(UserWarning, match="reduces to the high level alone"):
        oniom.createONIOMSystem(topology, [0, 1, 2])
    oniom.close()


def test_three_fresh_backends_and_close():
    topology = helpers.water_topology(2)
    high = ORCAPotential("HF", basis="def2-SVP", backend_factory=FlexibleFake)
    low = ORCAPotential("XTB", backend_factory=FlexibleFake)
    oniom = ONIOMPotential(high=high, low=low)
    oniom.createONIOMSystem(topology, [0, 1, 2])
    oniom.createONIOMSystem(topology, [3, 4, 5])
    assert len(oniom.high.backends) == 2
    assert len(oniom.low.backends) == 4
    # The two low backends of one System are distinct instances: the full-system
    # and model-region restart chains must never share MO-guess files.
    first_system_low = oniom.low.backends[:2]
    assert first_system_low[0] is not first_system_low[1]
    oniom.close()
    assert all(backend.n_closes == 1 for backend in oniom.high.backends + oniom.low.backends)
    oniom.close()  # idempotent
    assert all(backend.n_closes == 1 for backend in oniom.high.backends + oniom.low.backends)


def test_evaluation_counts_per_step():
    _, oniom, system = build_oniom()
    context = make_context(system, perturbed_positions())
    for _ in range(2):
        context.getState(getForces=True)
    high, low_full, low_model = layer_backends(oniom)
    assert high.n_calls == 2
    assert low_full.n_calls == 2
    assert low_model.n_calls == 2
    # Requests are routed by role: full system at the low level, model at both.
    assert low_full.last_request.n_qm == N_PARTICLES
    assert low_model.last_request.n_qm == len(MODEL)
    assert high.last_request.n_qm == len(MODEL)
    assert high.last_request.step == 1
    oniom.close()


def test_callback_input_validation():
    high, low_full, low_model = FlexibleFake(HIGH_K), FlexibleFake(LOW_K), FlexibleFake(LOW_K)
    with pytest.raises(ValueError, match="model atoms but"):
        ONIOMCallback(high, low_full, low_model, MODEL, ("O", "H"), ELEMENTS, N_PARTICLES)
    with pytest.raises(ValueError, match="full-system elements"):
        ONIOMCallback(high, low_full, low_model, MODEL, ("O", "H", "H"), ELEMENTS[:-1], N_PARTICLES)


def test_summarize_timings(tmp_path):
    from openmmorca.runtime.diagnostics import TIMING_FIELDS, TimingLog
    from openmmorca.runtime.scratch import ScratchDir

    _, oniom, _ = build_oniom()
    for i, backend in enumerate(oniom.high.backends + oniom.low.backends):
        backend.scratch = ScratchDir(tmp_path / f"scratch{i}")
        backend.n_fresh_retries = 0
        backend.timings = TimingLog(backend.scratch.root / "timings.csv")
        row = {field: 0.0 for field in TIMING_FIELDS}
        row.update({"step": 0, "t_total": 3.0 + i, "t_orca": 2.0})
        backend.timings.append(row)

    summaries = oniom.summarize_timings()
    assert len(summaries) == 3
    means = sorted(summary["t_total"]["mean"] for summary in summaries)
    assert means == pytest.approx([3.0, 4.0, 5.0])
    oniom.close()


# ---------------------------------------------------------------------------
# Real-ORCA tests (marked "orca", skipped automatically without OPI_ORCA)
# ---------------------------------------------------------------------------


@pytest.mark.orca
def test_orca_oniom_identity_reduces_to_low_level():
    """With high == low, ONIOM is exact: E_ONIOM must reproduce the full-QM result."""
    topology = helpers.water_topology(2)
    positions = helpers.water_positions(2) + np.array([0.004, -0.003, 0.002])
    config = dict(method="HF", basis="STO-3G", extra_keywords=("TightSCF",))
    oniom = ONIOMPotential(high=ORCAPotential(**config), low=ORCAPotential(**config))
    oniom_system = oniom.createONIOMSystem(topology, [0, 1, 2])

    reference_potential = ORCAPotential(**config)
    reference_system = reference_potential.createSystem(topology)

    energies = []
    reference_forces = None
    for system in (oniom_system, reference_system):
        context = make_context(system, positions)
        state = context.getState(getEnergy=True, getForces=True)
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        if reference_forces is None:
            reference_forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        else:
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            # The high-minus-low correction cancels at this level of theory up
            # to SCF convergence noise.
            np.testing.assert_allclose(forces, reference_forces, atol=0.5)
    assert energies[0] == pytest.approx(energies[1], abs=0.05)
    oniom.close()
    reference_potential.close()


@pytest.mark.orca
def test_orca_oniom_heterogeneous_levels_two_steps():
    """HF(model) : xTB(full) over two steps — exercises all three restart chains."""
    topology = helpers.water_topology(2)
    positions = helpers.water_positions(2)
    oniom = ONIOMPotential(
        high=ORCAPotential(method="HF", basis="STO-3G", extra_keywords=("TightSCF",)),
        low=ORCAPotential(method="XTB"),
    )
    system = oniom.createONIOMSystem(topology, [0, 1, 2])
    context = make_context(system, positions)
    energies = []
    for _ in range(2):
        state = context.getState(getEnergy=True, getForces=True)
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        assert np.isfinite(energies[-1])
        assert np.all(np.isfinite(forces))
        assert np.linalg.norm(forces.sum(axis=0)) < 1e-6

    high, low_full, low_model = layer_backends(oniom)
    # Step-2 energy matches a manual recombination at the same geometry.
    elements = tuple(atom.element.symbol for atom in topology.atoms())
    request_full = QMRequest(elements, positions)
    request_model = QMRequest(elements[:3], positions[:3])
    manual = (
        high.evaluate(request_model).energy_kj_mol
        + low_full.evaluate(request_full).energy_kj_mol
        - low_model.evaluate(request_model).energy_kj_mol
    )
    assert energies[-1] == pytest.approx(manual, abs=0.05)
    # Three restart chains were exercised (one MORead guess per backend).
    assert high.n_fresh_retries + low_full.n_fresh_retries + low_model.n_fresh_retries == 0
    oniom.close()
