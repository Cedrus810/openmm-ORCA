"""DhlA workflow regressions; no ORCA, OpenFF or full enzyme preparation."""

from __future__ import annotations

import csv
import io
import json
from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
from openmm import unit

import helpers
from examples import enzyme_qmmm as enzyme
from examples.enzyme_workflow import CacheMismatchError, RunMismatchError, StageCache, acceptance_summary
from openmmorca.backend.fake import FakeBackend
from openmmorca.potential import ORCAPotential


def temperature_rows(n, temperature=300.0):
    return [{"step": step, "temperature_K": temperature} for step in range(1, n + 1)]


@pytest.mark.parametrize("steps", [1, 200, 1999])
def test_short_run_is_smoke_only(steps):
    summary = acceptance_summary(temperature_rows(steps), steps, 0.1)
    assert summary["kind"] == "smoke"
    assert summary["status"] == "PASS"
    assert summary["task22_acceptance"] == "NOT_EVALUATED"


def test_full_acceptance_uses_last_500_samples():
    rows = temperature_rows(2000, 450.0)
    for row in rows[-500:]:
        row["temperature_K"] = 310.0
    summary = acceptance_summary(rows, 2000, 0.20)
    assert summary["task22_acceptance"] == "PASS"
    assert summary["temperature_mean_K"] == 310.0
    rows[-1]["temperature_K"] = 311.0
    assert acceptance_summary(rows, 2000, 0.20)["task22_acceptance"] == "FAIL"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nan", "bond", "zero_temperature"])
def test_bad_run_cannot_pass_full_acceptance(problem):
    rows = temperature_rows(2000)
    deviation = 0.1
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[9]["step"] = 9
    elif problem == "nan":
        rows[0]["temperature_K"] = float("nan")
    elif problem == "zero_temperature":
        rows[0]["temperature_K"] = 0
    else:
        deviation = 0.201
    assert acceptance_summary(rows, 2000, deviation)["task22_acceptance"] == "FAIL"


def test_reactive_bond_exclusion_does_not_rewrite_original_task22_gate():
    summary = acceptance_summary(temperature_rows(2000), 2000, 0.1, 0.35)
    assert summary["status"] == "PASS"  # configured stable-bond check
    assert summary["task22_acceptance"] == "FAIL"  # original all-bond gate


@pytest.mark.parametrize("steps", [0, -1])
def test_invalid_steps_rejected_before_side_effects(steps, tmp_path):
    args = SimpleNamespace(steps=steps, outdir=str(tmp_path / "unused"))
    with pytest.raises(ValueError, match="steps must be positive"):
        enzyme.run(args)
    assert not (tmp_path / "unused").exists()


def test_reaction_bonds_must_be_known_and_leave_stable_bonds():
    top = helpers.water_topology(1)
    stable, reactive = enzyme.partition_qm_bonds(top, [0, 1, 2], [(1, 0)])
    assert stable == [(0, 2)]
    assert reactive == [(0, 1)]
    with pytest.raises(ValueError, match="covalent bond"):
        enzyme.partition_qm_bonds(top, [0, 1, 2], [(1, 2)])
    with pytest.raises(ValueError, match="duplicate"):
        enzyme.partition_qm_bonds(top, [0, 1, 2], [(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="at least one"):
        enzyme.partition_qm_bonds(top, [0, 1, 2], [(0, 1), (0, 2)])


def test_cache_roundtrip_and_recipe_change_preserve_artifact(tmp_path):
    artifact = tmp_path / "stage.txt"
    artifact.write_text("original result")
    identity = {"stage": "prepare", "recipe": {"pH": 7.0}}
    cache = StageCache(artifact, identity)
    cache.store({"n_particles": 3})
    assert cache.load()["observations"] == {"n_particles": 3}
    identity["recipe"]["pH"] = 8.0
    # The old cache object froze its recipe at stage construction.
    assert cache.load() is not None
    with pytest.raises(CacheMismatchError, match=r"identity.recipe.pH"):
        StageCache(artifact, identity).load()
    assert artifact.read_text() == "original result"


@pytest.mark.parametrize("problem", ["legacy", "manifest_only", "tamper", "bad_json", "missing_identity"])
def test_cache_rejects_legacy_incomplete_or_modified_data(tmp_path, problem):
    artifact = tmp_path / "stage.txt"
    artifact.write_text("data")
    cache = StageCache(artifact, {"stage": "prepare"})
    cache.store({})
    if problem == "legacy":
        cache.manifest.unlink()
    elif problem == "manifest_only":
        artifact.unlink()
    elif problem == "tamper":
        artifact.write_text("changed")
    elif problem == "bad_json":
        cache.manifest.write_text("{broken")
    else:
        cache.manifest.write_text('{"schema_version": 1, "observations": {}}')
    with pytest.raises(CacheMismatchError):
        cache.load()


@pytest.fixture
def prepared_water(tmp_path, monkeypatch):
    monkeypatch.setattr(enzyme, "dependency_versions", lambda: {"openmm": mm.__version__})
    source = tmp_path / "source.pdb"
    source.write_text("test input")
    monkeypatch.setattr(enzyme, "DATA", source)
    topology = helpers.water_topology(1)
    topology.setPeriodicBoxVectors(tuple(mm.Vec3(*row) for row in np.eye(3) * 3) * unit.nanometer)
    text = io.StringIO()
    app.PDBFile.writeFile(topology, helpers.water_positions(1) * unit.nanometer, text)
    artifact = tmp_path / "prepared.pdb"
    artifact.write_text(text.getvalue())
    pdb = app.PDBFile(str(artifact))
    enzyme.preparation_cache(tmp_path).store(enzyme.prepared_observations(pdb))
    return pdb


def test_preparation_cache_reuses_valid_data_and_rejects_changed_recipe(prepared_water, tmp_path, monkeypatch):
    # A validated cache does not need the preparation-only dependencies.
    cached = enzyme.prepare(tmp_path, forcefield=None)
    assert enzyme.topology_digest(cached.topology) == enzyme.topology_digest(prepared_water.topology)
    monkeypatch.setitem(enzyme.PREPARATION_RECIPE, "pH", 8.0)
    with pytest.raises(CacheMismatchError, match="identity.recipe.pH"):
        enzyme.prepare(tmp_path, forcefield=None)


def test_preparation_cache_rejects_changed_source(prepared_water, tmp_path):
    enzyme.DATA.write_text("different source")
    with pytest.raises(CacheMismatchError, match="source_sha256"):
        enzyme.prepare(tmp_path, forcefield=None)


def test_preparation_cache_checks_atom_mapping_beyond_file_checksum(prepared_water, tmp_path):
    cache = enzyme.preparation_cache(tmp_path)
    manifest = json.loads(cache.manifest.read_text())
    manifest["observations"]["topology_sha256"] = "wrong atom mapping"
    cache.manifest.write_text(json.dumps(manifest))
    with pytest.raises(CacheMismatchError, match="observations.topology_sha256"):
        enzyme.prepare(tmp_path, forcefield=None)


def cache_equilibrated_water(tmp_path, pdb, n_particles=None):
    ff = app.ForceField("tip3p.xml")
    system = enzyme.mm_system(pdb.topology, ff)
    if n_particles is not None:
        system = mm.System()
        for _ in range(n_particles):
            system.addParticle(1)
        system.setDefaultPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    n = system.getNumParticles()
    context.setPositions(helpers.water_positions(1)[:n] * unit.nanometer)
    context.setVelocities(np.zeros((n, 3)) * unit.nanometer / unit.picosecond)
    context.setTime(70 * unit.picosecond)
    state = context.getState(getPositions=True, getVelocities=True)
    artifact = tmp_path / "equilibrated.xml"
    artifact.write_text(mm.XmlSerializer.serialize(state))
    return ff, mm.XmlSerializer.deserialize(artifact.read_text())


def test_equilibration_cache_reuse_and_changed_seed(prepared_water, tmp_path, monkeypatch):
    ff, state = cache_equilibrated_water(tmp_path, prepared_water)
    platform = mm.Platform.getPlatformByName("Reference")
    cache = enzyme.equilibration_cache(tmp_path, prepared_water, enzyme.mm_system(prepared_water.topology, ff), platform)
    cache.store(enzyme.state_observations(state, prepared_water.topology))
    loaded = enzyme.equilibrate(tmp_path, prepared_water, ff, platform)
    np.testing.assert_array_equal(loaded.getPositions(asNumpy=True), state.getPositions(asNumpy=True))
    monkeypatch.setitem(enzyme.EQUILIBRATION_RECIPE, "integrator_seed", 999)
    with pytest.raises(CacheMismatchError, match="integrator_seed"):
        enzyme.equilibrate(tmp_path, prepared_water, ff, platform)


def test_equilibration_identity_tracks_system_and_prepared_order(prepared_water, tmp_path):
    ff = app.ForceField("tip3p.xml")
    system = enzyme.mm_system(prepared_water.topology, ff)
    platform = mm.Platform.getPlatformByName("Reference")
    original = enzyme.equilibration_cache(tmp_path, prepared_water, system, platform).identity
    system.setParticleMass(0, 17 * unit.dalton)
    changed = enzyme.equilibration_cache(tmp_path, prepared_water, system, platform).identity
    assert changed["system_sha256"] != original["system_sha256"]
    atoms = list(prepared_water.topology.atoms())
    atoms[1].name, atoms[2].name = atoms[2].name, atoms[1].name
    changed = enzyme.equilibration_cache(tmp_path, prepared_water, system, platform).identity
    assert changed["prepared"]["topology_sha256"] != original["prepared"]["topology_sha256"]


def test_equilibration_state_rejects_particle_count_mismatch(prepared_water, tmp_path):
    ff, state = cache_equilibrated_water(tmp_path, prepared_water, n_particles=2)
    platform = mm.Platform.getPlatformByName("Reference")
    cache = enzyme.equilibration_cache(tmp_path, prepared_water, enzyme.mm_system(prepared_water.topology, ff), platform)
    cache.store({"n_particles": 3})  # A checksum alone cannot validate a State's shape.
    with pytest.raises(ValueError, match="match 3 topology atoms"):
        enzyme.equilibrate(tmp_path, prepared_water, ff, platform)


def test_equilibration_writes_and_reuses_a_new_validated_cache(prepared_water, tmp_path, monkeypatch):
    # Exercise the real cold-cache path on a small MM system, with a short
    # recipe for this software test. This is not a 70 ps equilibration claim.
    monkeypatch.setitem(enzyme.EQUILIBRATION_RECIPE, "nvt_steps", 2)
    monkeypatch.setitem(enzyme.EQUILIBRATION_RECIPE, "npt_steps", 2)
    ff = app.ForceField("tip3p.xml")
    platform = mm.Platform.getPlatformByName("Reference")
    state = enzyme.equilibrate(tmp_path, prepared_water, ff, platform)
    manifest = json.loads((tmp_path / "equilibrated.xml.manifest.json").read_text())
    assert manifest["identity"]["recipe"]["nvt_steps"] == 2
    assert state.getTime().value_in_unit(unit.picosecond) == pytest.approx(0.008)
    loaded = enzyme.equilibrate(tmp_path, prepared_water, ff, platform)
    np.testing.assert_array_equal(state.getPositions(asNumpy=True), loaded.getPositions(asNumpy=True))


class TimedFake(FakeBackend):
    def __init__(self, fail_at=None):
        # O-H-like stiffness keeps the smoke bond check far from its 20 % gate
        # on any OpenMM version's random stream.
        super().__init__([-0.834, 0.417, 0.417], k_bond=4e5)
        # Fixed reference geometry: a rebuilt backend (resume) must give the
        # same forces as the original one.
        qm = helpers.water_positions(1)
        self._r0 = np.linalg.norm(qm[:, None, :] - qm[None, :, :], axis=-1)
        self.timings = SimpleNamespace(rows=[])
        self.n_fresh_retries = 0
        self.n_closes = 0
        self.fail_at = fail_at

    def evaluate(self, request):
        if request.step == self.fail_at:
            raise RuntimeError("injected QM failure")
        result = super().evaluate(request)
        self.timings.rows.append({
            "t_write": 0, "t_orca": 0.001, "t_read": 0, "t_total": 0.001,
            "scf_cycles": 12, "restart_used": request.step > 0, "fresh_retries": 0,
            "n_embed_groups": 0, "embed_changed": 0,
        })
        return result

    def close(self):
        self.n_closes += 1


def setup_tiny_run(tmp_path, monkeypatch, fail_at=None, steps=1, **options):
    import openmmorca

    topology = helpers.water_topology(1)
    system = helpers.flexible_tip3p_system(topology)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(helpers.water_positions(1) * unit.nanometer)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 2026)
    state = context.getState(getPositions=True, getVelocities=True)
    fake = TimedFake(fail_at)
    potential = ORCAPotential(method="HF", basis="STO-3G", backend_factory=lambda: fake)
    monkeypatch.setattr(openmmorca, "ORCAPotential", lambda **kwargs: potential)
    monkeypatch.setattr(enzyme, "forcefield_with_dce", lambda: None)
    monkeypatch.setattr(enzyme, "prepare", lambda *args: SimpleNamespace(topology=topology))
    monkeypatch.setattr(enzyme, "equilibrate", lambda *args: state)
    monkeypatch.setattr(enzyme, "mm_system", lambda *args: helpers.flexible_tip3p_system(topology))
    monkeypatch.setattr(enzyme, "qm_selection", lambda *args: ([0, 1, 2], [], [0], 1, 2))
    args = SimpleNamespace(
        outdir=str(tmp_path), platform="Reference", method="HF", basis="STO-3G", keyword=[],
        region="A", steps=steps, min_iterations=0, nprocs=1, cutoff=1.2, reaction_bond=[],
        checkpoint_interval=10, resume=None, archive_existing=False,
    )
    for key, value in options.items():
        setattr(args, key, value)
    return args, fake


def test_single_step_example_saves_smoke_result_and_closes_backend(tmp_path, monkeypatch, capsys):
    args, fake = setup_tiny_run(tmp_path, monkeypatch)
    assert enzyme.run(args)
    summary = json.loads((tmp_path / "acceptance.json").read_text())
    assert summary["kind"] == "smoke"
    assert summary["task22_acceptance"] == "NOT_EVALUATED"
    assert fake.n_closes == 1
    output = capsys.readouterr().out
    assert "NOT_EVALUATED" in output
    assert "nan" not in output.lower()


def test_failed_example_preserves_completed_csv_and_clears_stale_pass(tmp_path, monkeypatch):
    args, fake = setup_tiny_run(tmp_path, monkeypatch, fail_at=1, steps=2)
    (tmp_path / "acceptance.json").write_text('{"task22_acceptance": "PASS"}')
    with pytest.raises(mm.OpenMMException, match="injected QM failure"):
        enzyme.run(args)
    with open(tmp_path / "steps.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["1"]
    summary = json.loads((tmp_path / "acceptance.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["task22_acceptance"] == "NOT_EVALUATED"
    assert fake.n_closes == 1


# --- T04: checkpoints, run status and resume ---------------------------------

def read_steps(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def final_arrays(outdir):
    state = mm.XmlSerializer.deserialize((outdir / "final_state.xml").read_text())
    return (state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
            state.getStepCount())


def interrupted_run(tmp_path, monkeypatch):
    """30-step run that fails during step 26; the last checkpoint is step 20."""
    args, _ = setup_tiny_run(tmp_path, monkeypatch, fail_at=25, steps=30)
    with pytest.raises(mm.OpenMMException, match="injected QM failure"):
        enzyme.run(args)


def test_failure_keeps_log_status_and_last_checkpoint(tmp_path, monkeypatch):
    interrupted_run(tmp_path, monkeypatch)
    assert [row["step"] for row in read_steps(tmp_path / "steps.csv")] == [str(i) for i in range(1, 26)]
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["checkpoint"]["step"] == 20
    assert record["attempts"][0]["valid_through_step"] == 20
    assert record["attempts"][0]["error_type"] == "OpenMMException"
    # Only the committed checkpoint pair is kept, and both are readable.
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == ["step-0000020.chk", "step-0000020.xml"]
    state = mm.XmlSerializer.deserialize((tmp_path / "checkpoints/step-0000020.xml").read_text())
    assert state.getStepCount() == 20
    assert json.loads((tmp_path / "acceptance.json").read_text())["status"] == "FAILED"


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    args, _ = setup_tiny_run(reference, monkeypatch, steps=30)
    assert enzyme.run(args)

    resumed = tmp_path / "resumed"
    interrupted_run(resumed, monkeypatch)
    before = (resumed / "steps.csv").read_bytes()
    args, fake = setup_tiny_run(resumed, monkeypatch, steps=30, resume="checkpoint")
    assert enzyme.run(args)
    assert len(fake.timings.rows) == 10  # only steps 21..30 recomputed

    # Integrator, RNG and step count restored: bitwise-identical trajectory.
    for a, b in zip(final_arrays(reference), final_arrays(resumed)):
        np.testing.assert_array_equal(a, b)
    physics = ["step", "temperature_K", "d_OD_C1_nm", "d_C1_Cl1_nm", "max_bond_dev"]
    expected = [[row[k] for k in physics] for row in read_steps(reference / "steps.csv")]
    assert [[row[k] for k in physics] for row in read_steps(resumed / "steps.csv")] == expected

    # Checkpointed history kept byte for byte; recomputed rows preserved aside.
    record = json.loads((resumed / "run.json").read_text())
    assert (resumed / "steps.csv").read_bytes().startswith(before[:before.index(b"\n21,")])
    superseded = read_steps(resumed / "steps.superseded.attempt-001.csv")
    assert [row["step"] for row in superseded] == ["21", "22", "23", "24", "25"]
    assert record["status"] == "completed"
    assert [(a["mode"], a["start_step"], a["valid_through_step"], a["trajectory"]) for a in record["attempts"]] == [
        ("fresh", 0, 20, "trajectory.dcd"), ("checkpoint", 20, 30, "trajectory.attempt-002.dcd"),
    ]
    assert record["attempts"][0]["status"] == "failed"
    assert (resumed / "trajectory.dcd").is_file() and (resumed / "trajectory.attempt-002.dcd").is_file()
    summary = json.loads((resumed / "acceptance.json").read_text())
    assert summary["completed_steps"] == 30 and summary["attempts"] == 2


def test_state_resume_restarts_from_state_with_new_seed(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    args, _ = setup_tiny_run(reference, monkeypatch, steps=30)
    assert enzyme.run(args)
    resumed = tmp_path / "resumed"
    interrupted_run(resumed, monkeypatch)
    args, _ = setup_tiny_run(resumed, monkeypatch, steps=30, resume="state")
    assert enzyme.run(args)
    assert [row["step"] for row in read_steps(resumed / "steps.csv")] == [str(i) for i in range(1, 31)]
    record = json.loads((resumed / "run.json").read_text())
    assert record["attempts"][1]["mode"] == "state"
    assert final_arrays(resumed)[2] == 30
    # New Langevin noise: a State restart is not the uninterrupted trajectory.
    assert not np.array_equal(final_arrays(reference)[0], final_arrays(resumed)[0])


def test_completed_run_can_be_extended_but_not_rerun(tmp_path, monkeypatch):
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=20)
    assert enzyme.run(args)
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=20, resume="checkpoint")
    with pytest.raises(RunMismatchError, match="checkpointed at step 20"):
        enzyme.run(args)
    assert json.loads((tmp_path / "acceptance.json").read_text())["status"] == "PASS"
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=30, resume="checkpoint")
    assert enzyme.run(args)
    assert len(read_steps(tmp_path / "steps.csv")) == 30


def test_fresh_run_refuses_existing_products_unless_archived(tmp_path, monkeypatch):
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=10)
    assert enzyme.run(args)
    original = (tmp_path / "steps.csv").read_bytes()
    args, fake = setup_tiny_run(tmp_path, monkeypatch, steps=10)
    with pytest.raises(RunMismatchError, match="already holds QM/MM run products"):
        enzyme.run(args)
    assert fake.timings.rows == [] and (tmp_path / "steps.csv").read_bytes() == original
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=10, archive_existing=True)
    assert enzyme.run(args)
    (archive,) = (tmp_path / "archive").iterdir()
    assert (archive / "steps.csv").read_bytes() == original
    assert {"run.json", "acceptance.json", "trajectory.dcd", "checkpoints"} <= {p.name for p in archive.iterdir()}


def test_legacy_outdir_without_run_record_is_protected(tmp_path, monkeypatch):
    (tmp_path / "steps.csv").write_text("step,temperature_K\n1,300\n")
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=10, resume="checkpoint")
    with pytest.raises(RunMismatchError, match="needs"):
        enzyme.run(args)
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=10)
    with pytest.raises(RunMismatchError, match="already holds"):
        enzyme.run(args)


@pytest.mark.parametrize("problem", ["cutoff", "csv", "checkpoint", "platform"])
def test_resume_rejects_changed_run(tmp_path, monkeypatch, problem):
    interrupted_run(tmp_path, monkeypatch)
    options = {"resume": "checkpoint"}
    if problem == "cutoff":
        options["cutoff"] = 1.1
        match = "identity.qm.cutoff_nm"
    elif problem == "csv":
        text = (tmp_path / "steps.csv").read_text()
        (tmp_path / "steps.csv").write_text(text.replace("\n3,", "\n4,", 1))
        match = "modified before checkpoint step 20"
    elif problem == "checkpoint":
        (tmp_path / "checkpoints/step-0000020.chk").write_bytes(b"corrupt")
        match = "does not match run.json"
    else:
        record = json.loads((tmp_path / "run.json").read_text())
        record["checkpoint"]["platform"] = "CUDA"
        (tmp_path / "run.json").write_text(json.dumps(record))
        match = "use --resume state"
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    args, fake = setup_tiny_run(tmp_path, monkeypatch, steps=30, **options)
    with pytest.raises(RunMismatchError, match=match):
        enzyme.run(args)
    assert fake.timings.rows == []
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert after == before


def test_second_run_in_same_outdir_is_locked(tmp_path, monkeypatch):
    import fcntl

    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=10)
    with open(tmp_path / ".run.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another run holds"):
            enzyme.run(args)


@pytest.mark.parametrize("interval", [0, 15])
def test_checkpoint_interval_must_align_with_dcd_frames(tmp_path, interval):
    args = SimpleNamespace(steps=10, min_iterations=0, nprocs=1, cutoff=1.2,
                           checkpoint_interval=interval, outdir=str(tmp_path / "unused"))
    with pytest.raises(ValueError, match="multiple of 10"):
        enzyme.run(args)
    assert not (tmp_path / "unused").exists()


# --- T06: independent analysis of run products --------------------------------

def analyze(outdir):
    from examples.analyze_enzyme_run import Analysis

    return Analysis(outdir).run()


def test_analysis_recomputes_a_complete_run(tmp_path, monkeypatch):
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=30)
    assert enzyme.run(args)
    report = analyze(tmp_path)
    assert report["problems"] == []
    assert report["length"] == {"requested_steps": 30, "logged_steps": 30, "complete": True,
                                "duration_ps": 0.015, "attempts": 1}
    summary = json.loads((tmp_path / "acceptance.json").read_text())
    assert report["temperature"]["tail_mean_K"] == pytest.approx(summary["temperature_mean_K"], rel=1e-12)
    trajectory = report["bonds"]["trajectory"]
    assert trajectory["frames"] == 3
    assert 0 < trajectory["max_bond_dev"] <= report["bonds"]["max_bond_dev"] + 1e-4
    assert report["qm_calls"]["scf_cycles_max"] == 12
    assert report["provenance"]["attempts"][0]["environment"]["versions"]["openmm"] == mm.__version__


def test_analysis_follows_resumed_attempts(tmp_path, monkeypatch):
    interrupted_run(tmp_path, monkeypatch)
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=30, resume="checkpoint")
    assert enzyme.run(args)
    report = analyze(tmp_path)
    assert report["problems"] == []
    assert report["length"]["attempts"] == 2
    # Frames at steps 10, 20 (attempt 1) and 30 (attempt 2); none superseded.
    assert report["bonds"]["trajectory"]["frames"] == 3
    assert report["qm_calls"]["fresh_retries_by_attempt"] == {"1": 0, "2": 0}


def recommit_csv(outdir, transform):
    """Edit steps.csv and update run.json so only the edited content differs."""
    import hashlib

    path = outdir / "steps.csv"
    path.write_text(transform(path.read_text()))
    record = json.loads((outdir / "run.json").read_text())
    data = path.read_bytes()
    record["checkpoint"].update(steps_csv_bytes=len(data), steps_csv_sha256=hashlib.sha256(data).hexdigest())
    (outdir / "run.json").write_text(json.dumps(record))


@pytest.mark.parametrize("problem", ["acceptance", "csv", "understated_bonds"])
def test_analysis_reports_inconsistent_products(tmp_path, monkeypatch, problem):
    args, _ = setup_tiny_run(tmp_path, monkeypatch, steps=30)
    assert enzyme.run(args)
    if problem == "acceptance":
        summary = json.loads((tmp_path / "acceptance.json").read_text())
        summary["status"] = "FAIL"
        (tmp_path / "acceptance.json").write_text(json.dumps(summary))
        match = "acceptance.json status"
    elif problem == "csv":
        path = tmp_path / "steps.csv"
        path.write_text(path.read_text().replace("\n5,", "\n6,", 1))
        match = "differs from the final checkpoint"
    else:
        def zero_bond_columns(text):
            rows = list(csv.DictReader(io.StringIO(text)))
            for row in rows:
                row["max_bond_dev"] = row["max_stable_bond_dev"] = "0.0"
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\r\n")
            writer.writeheader()
            writer.writerows(rows)
            return out.getvalue()
        recommit_csv(tmp_path, zero_bond_columns)
        match = "DCD bond deviation exceeds the logged maximum"
    problems = analyze(tmp_path)["problems"]
    assert any(match in p for p in problems), problems
