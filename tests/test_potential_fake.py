"""Tests for ORCAPotential with an injected fake backend (plan Task 9)."""

from __future__ import annotations

import numpy as np
import openmm as mm
import pytest
from openmm import unit

import helpers
from openmmorca.backend.fake import FakeBackend
from openmmorca.potential import ORCAPotential

TIP3P_CHARGES = (-0.834, 0.417, 0.417)


class RecordingFake(FakeBackend):
    def __init__(self):
        super().__init__(qm_charges_e=TIP3P_CHARGES)
        self.n_closes = 0

    def close(self):
        self.n_closes += 1


def fake_potential(**kwargs) -> ORCAPotential:
    return ORCAPotential(
        method="HF", basis="def2-SVP", backend_factory=RecordingFake, **kwargs
    )


def test_create_system_full_qm():
    potential = fake_potential()
    topology = helpers.water_topology(1)
    system = potential.createSystem(topology)
    assert system.getNumParticles() == 3
    masses = [system.getParticleMass(i) for i in range(3)]
    assert all(float(m.value_in_unit(unit.dalton)) > 0 for m in masses)
    force_types = [type(f).__name__ for f in system.getForces()]
    assert "PythonForce" in force_types
    assert "CMMotionRemover" in force_types

    positions = helpers.water_positions(1)
    context = mm.Context(
        system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions * unit.nanometer)
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    backend = potential.backends[0]
    direct = backend.evaluate(backend.last_request)
    assert energy == pytest.approx(direct.energy_kj_mol, abs=1e-10)
    potential.close()


def test_create_mixed_system():
    potential = fake_potential()
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    mixed = potential.createMixedSystem(topology, system, [0, 1, 2], forceGroup=1)
    force_groups = {f.getForceGroup() for f in mixed.getForces()}
    assert 1 in force_groups

    positions = helpers.water_positions(2)
    context = mm.Context(
        mixed, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions * unit.nanometer)
    energy = context.getState(getEnergy=True, groups=1 << 1).getPotentialEnergy()
    energy = energy.value_in_unit(unit.kilojoule_per_mole)
    backend = potential.backends[0]
    direct = backend.evaluate(backend.last_request)
    assert energy == pytest.approx(direct.energy_kj_mol, abs=1e-10)
    potential.close()


def test_each_system_gets_its_own_backend():
    potential = fake_potential()
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    mixed1 = potential.createMixedSystem(topology, system, [0, 1, 2])
    mixed2 = potential.createMixedSystem(topology, system, [0, 1, 2])
    assert len(potential.backends) == 2
    assert potential.backends[0] is not potential.backends[1]
    potential.close()


def test_interpolate_rejected():
    potential = fake_potential()
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    with pytest.raises(NotImplementedError):
        potential.createMixedSystem(topology, system, [0, 1, 2], interpolate=True)


def test_embedding_rejected():
    potential = fake_potential()
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    with pytest.raises(ValueError):
        potential.createMixedSystem(topology, system, [0, 1, 2], embedding="mechanical")


def test_config_errors_raised_at_construction():
    with pytest.raises(ValueError):
        ORCAPotential("HF", basis="def2-SVP", extra_keywords=("Opt",))


def test_close_closes_all_backends():
    potential = fake_potential()
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    potential.createMixedSystem(topology, system, [0, 1, 2])
    potential.createMixedSystem(topology, system, [0, 1, 2])
    potential.close()
    assert all(backend.n_closes == 1 for backend in potential.backends)
    potential.close()  # idempotent
    assert all(backend.n_closes == 1 for backend in potential.backends)


def test_summarize_timings(tmp_path):
    from openmmorca.runtime.diagnostics import TIMING_FIELDS, TimingLog
    from openmmorca.runtime.scratch import ScratchDir

    potential = fake_potential()
    topology = helpers.water_topology(2)
    system = helpers.flexible_tip3p_system(topology)
    potential.createMixedSystem(topology, system, [0, 1, 2])
    backend = potential.backends[0]
    backend.scratch = ScratchDir(tmp_path / "scratch")
    backend.n_fresh_retries = 0
    backend.timings = TimingLog(backend.scratch.root / "timings.csv")
    for i, t_total in enumerate([10.0, 2.0, 4.0]):
        row = {field: 0.0 for field in TIMING_FIELDS}
        row.update({"step": i, "t_total": t_total, "t_orca": t_total - 1})
        backend.timings.append(row)

    summaries = potential.summarize_timings()
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["scratch"] == str(backend.scratch.root)
    assert summary["n_fresh_retries"] == 0
    assert summary["t_total"]["mean"] == pytest.approx(3.0)
    assert summary["t_total"]["p50"] == pytest.approx(3.0)
    assert summary["t_total"]["p95"] == pytest.approx(3.9)
    potential.close()
