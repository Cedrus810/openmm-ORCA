"""Tests for the SCF restart state machine and failure bundles (plan Task 13)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openmmorca.backend.base import QMRequest
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend, _StepOutput
from openmmorca.errors import ORCACalculationError, ORCAOutputError

WATER_ELEMENTS = ("O", "H", "H")


def make_request(step=0) -> QMRequest:
    return QMRequest(
        qm_elements=WATER_ELEMENTS,
        qm_positions_nm=np.array([[0.0, 0, 0], [0.096, 0, 0], [-0.024, 0.093, 0]]),
        step=step,
    )


def make_backend(scratch_root, **overrides) -> ORCAOPIBackend:
    config = ORCAConfig(method="HF", basis="def2-SVP", scratch_root=str(scratch_root), **overrides)
    return ORCAOPIBackend(config, check_version=False)


def fake_step_output(request, backend):
    """A valid _StepOutput; also fakes the gbw file needed for commit()."""
    backend.scratch.current.mkdir(parents=True, exist_ok=True)
    (backend.scratch.current / "qm.gbw").write_bytes(b"fake-gbw")
    return _StepOutput(
        energy_eh=-76.0,
        qm_gradient=np.zeros((request.n_qm, 3)),
        pc_gradient=None,
        t_write=0.0,
        t_orca=0.1,
        t_read=0.0,
        scf_cycles=5,
    )


def test_state_machine_first_step_fresh_only(scratch_root):
    backend = make_backend(scratch_root)
    calls = []

    def fake_run(request, use_guess):
        calls.append(use_guess)
        raise ORCACalculationError("boom")

    backend._run_step = fake_run
    with pytest.raises(ORCACalculationError):
        backend.evaluate(make_request())
    assert calls == [False]


def test_state_machine_retry_after_moread_failure(scratch_root):
    backend = make_backend(scratch_root)
    backend.restart_state.last_good.write_bytes(b"last-good")
    calls = []

    def fake_run(request, use_guess):
        calls.append(use_guess)
        if use_guess:
            raise ORCACalculationError("MORead failed")
        return fake_step_output(request, backend)

    backend._run_step = fake_run
    with pytest.warns(RuntimeWarning, match="fresh SCF converged"):
        backend.evaluate(make_request(step=1))
    assert calls == [True, False]
    assert backend.n_fresh_retries == 1


def test_state_machine_both_fail(scratch_root):
    backend = make_backend(scratch_root)
    backend.restart_state.last_good.write_bytes(b"last-good")
    calls = []

    def fake_run(request, use_guess):
        calls.append(use_guess)
        raise ORCACalculationError("boom")

    backend._run_step = fake_run
    with pytest.raises(ORCACalculationError):
        backend.evaluate(make_request(step=3))
    assert calls == [True, False]
    bundles = list(backend.scratch.failures.glob("failure_step_000003/*"))
    assert any(path.name == "metadata.json" for path in bundles)


def test_output_error_not_retried(scratch_root):
    backend = make_backend(scratch_root)
    calls = []

    def fake_run(request, use_guess):
        calls.append(use_guess)
        raise ORCAOutputError("missing file")

    backend._run_step = fake_run
    with pytest.raises(ORCAOutputError):
        backend.evaluate(make_request())
    assert calls == [False]


def test_reentrancy_rejected(scratch_root):
    backend = make_backend(scratch_root)

    def fake_run(request, use_guess):
        # The outer evaluate still holds the lock, so the inner call must fail.
        with pytest.raises(RuntimeError, match="not re-entrant"):
            backend.evaluate(request)
        return fake_step_output(request, backend)

    backend._run_step = fake_run
    backend.evaluate(make_request())


@pytest.mark.orca
def test_second_step_uses_moread(scratch_root):
    backend = make_backend(scratch_root)
    positions = np.array([[0.0, 0, 0], [0.096, 0, 0], [-0.024, 0.093, 0]])
    for step in range(3):
        request = QMRequest(
            qm_elements=WATER_ELEMENTS,
            qm_positions_nm=positions.copy(),
            step=step,
        )
        backend.evaluate(request)
        positions[1, 0] += 0.002 / 18.89726125  # 0.002 Å in nm
    rows = backend.timings.rows
    assert rows[0]["restart_used"] is False
    assert rows[1]["restart_used"] is True
    assert rows[2]["restart_used"] is True
    assert 0 < rows[1]["scf_cycles"] < rows[0]["scf_cycles"]


@pytest.mark.orca
def test_restart_energy_matches_fresh(scratch_root):
    request = make_request()
    backend_on = make_backend(scratch_root / "on", restart=True)
    backend_off = make_backend(scratch_root / "off", restart=False)
    result_on = backend_on.evaluate(request)
    result_off = backend_off.evaluate(request)
    assert abs(result_on.energy_kj_mol - result_off.energy_kj_mol) < 1e-6 * 2625.4996394799
    backend_on.close()
    backend_off.close()


@pytest.mark.orca
def test_forced_scf_failure_bundle(scratch_root):
    backend = make_backend(
        scratch_root, restart=False, extra_blocks=("%scf maxiter 2 end",)
    )
    with pytest.raises(ORCACalculationError) as excinfo:
        backend.evaluate(make_request(step=5))
    assert "Failure bundle" in str(excinfo.value)
    bundle_dir = backend.scratch.failures / "failure_step_000005"
    assert (bundle_dir / "qm.inp").is_file()
    assert "SCF NOT CONVERGED" in (bundle_dir / "qm.out").read_text()


@pytest.mark.orca
def test_failed_run_does_not_clobber_last_good(scratch_root):
    backend = make_backend(scratch_root)
    backend.evaluate(make_request(step=0))
    last_good_bytes = backend.restart_state.last_good.read_bytes()
    backend.config = dataclasses.replace(
        backend.config,
        extra_blocks=("%scf maxiter 2 end",),
        restart=False,  # MORead converges within 2 cycles; a fresh SCF cannot
    )
    with pytest.raises(ORCACalculationError):
        backend.evaluate(make_request(step=1))
    assert backend.restart_state.last_good.read_bytes() == last_good_bytes
