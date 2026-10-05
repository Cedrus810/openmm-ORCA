"""Tests for MPI environment handling and nprocs behaviour (plan Task 14)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from openmmorca.backend.base import QMRequest
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend

PC_NM = np.array([[0.30, 0.0, 0.0], [0.35, 0.08, 0.0]])
PC_CHARGES = np.array([-0.834, 0.417])


def _mpi_viable() -> tuple[bool, str]:
    """Check that parallel ORCA can actually run on this node.

    Two failure modes seen in the wild (spec §2.7, §8.6):

    * ``mpirun`` on PATH but ORCA's MPI launcher libs unresolved — OPI only
      adds the MPI lib directory when ``OPI_MPI`` is set;
    * inside a PBS job, PRRTE reads the scheduler's allocation and refuses
      ``mpirun -np 2`` with "Not enough slots available" when the job was
      granted fewer cores than requested processes.

    A real ``mpirun -np 2 hostname`` probe catches both plus anything else,
    and gives the user an actionable message instead of an ORCA crash.
    """
    if os.environ.get("OPI_MPI"):
        mpirun = str(Path(os.environ["OPI_MPI"]) / "bin" / "mpirun")
    else:
        found = shutil.which("mpirun")
        if not found:
            return False, "no mpirun on PATH and OPI_MPI not set"
        mpirun = found
    try:
        probe = subprocess.run(
            [mpirun, "-np", "2", "hostname"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"mpirun probe failed: {exc}"
    if probe.returncode != 0:
        tail = (probe.stderr or probe.stdout or "").strip().splitlines()
        head = "; ".join(tail[:3]) if tail else f"exit {probe.returncode}"
        return False, (
            f"mpirun -np 2 cannot start 2 processes ({head}). Inside a batch job "
            "this usually means the job was allocated fewer cores than "
            "nprocs: request at least nprocs slots, "
            "or set OPI_MPI and/or OMPI_MCA_rmaps_default_mapping_policy="
            ":oversubscribe deliberately"
        )
    return True, ""


@pytest.fixture
def require_mpi():
    viable, reason = _mpi_viable()
    if not viable:
        pytest.skip(f"parallel ORCA not viable: {reason}")


def make_request() -> QMRequest:
    return QMRequest(
        qm_elements=("O", "H", "H"),
        qm_positions_nm=np.array([[0.0, 0, 0], [0.096, 0, 0], [-0.024, 0.093, 0]]),
        mm_positions_nm=PC_NM.copy(),
        mm_charges_e=PC_CHARGES.copy(),
    )


def make_backend(scratch_root, *, check_version=False, **overrides) -> ORCAOPIBackend:
    config = ORCAConfig(
        method="HF", basis="def2-SVP", scratch_root=str(scratch_root), **overrides
    )
    return ORCAOPIBackend(config, check_version=check_version)


@pytest.mark.orca
def test_nprocs2_matches_serial(scratch_root, require_mpi):
    serial = make_backend(scratch_root / "serial", nprocs=1)
    parallel = make_backend(scratch_root / "parallel", nprocs=2)
    request = make_request()
    r1 = serial.evaluate(request)
    r2 = parallel.evaluate(request)
    assert abs(r1.energy_kj_mol - r2.energy_kj_mol) / 2625.4996394799 < 1e-8
    np.testing.assert_allclose(
        r1.qm_forces_kj_mol_nm, r2.qm_forces_kj_mol_nm, atol=1e-6 * 49614.752589
    )
    serial.close()
    parallel.close()


@pytest.mark.orca
def test_mpi_env_restored(scratch_root, require_mpi):
    backend = make_backend(scratch_root, nprocs=2)
    before = {k: v for k, v in os.environ.items() if k.startswith("OMPI_MCA_")}
    backend.evaluate(make_request())
    after = {k: v for k, v in os.environ.items() if k.startswith("OMPI_MCA_")}
    assert after == before
    backend.close()


@pytest.mark.orca
def test_user_mpi_env_not_overridden(scratch_root, require_mpi, monkeypatch):
    monkeypatch.setenv("OMPI_MCA_btl", "self,tcp")
    captured = {}
    backend = make_backend(scratch_root, nprocs=2)
    original_run = backend._get_runner().run_orca

    def spy(inpfile, *args, **kwargs):
        captured["btl"] = os.environ.get("OMPI_MCA_btl")
        return original_run(inpfile, *args, **kwargs)

    monkeypatch.setattr(backend._runner, "run_orca", spy)
    backend.evaluate(make_request())
    assert captured["btl"] == "self,tcp"
    backend.close()


@pytest.mark.orca
def test_nprocs2_is_not_pathologically_slow(scratch_root, require_mpi):
    # A serial ORCA run first: OPI's Runner used to leave os.environ as a plain
    # dict, after which the MCA tuning of later parallel runs never reached ORCA.
    serial = make_backend(scratch_root / "serial", nprocs=1)
    serial.evaluate(make_request())
    serial.close()
    backend = make_backend(scratch_root / "parallel", nprocs=2)
    backend.evaluate(make_request())  # warm-up (first run pays one-time costs)
    t0 = time.perf_counter()
    backend.evaluate(make_request())
    wall = time.perf_counter() - t0
    assert wall < 5.0, f"nprocs=2 took {wall:.1f} s — MCA tuning likely ineffective"
    backend.close()


# ---------------------------------------------------------------------------
# Process environment seen by child processes (OPI Runner side effects).
#
# OPI's Runner wraps every call in ``_orca_environment``, whose ``finally``
# does ``os.environ = os.environ.copy()``. Checking ``os.environ`` itself is
# therefore not enough: these tests look at what a child process inherits,
# each in a fresh interpreter so earlier tests cannot mask the problem.
# ---------------------------------------------------------------------------

_FRESH_PRELUDE = """
import json, os, subprocess, sys
sys.path.insert(0, {tests_dir!r})
from test_orca_mpi import make_backend, make_request
from pathlib import Path

def child_env():
    out = subprocess.run(["env", "-0"], capture_output=True, check=True).stdout
    return dict(item.split("=", 1) for item in out.decode().split("\\0") if item)
"""


def _run_fresh(body: str) -> dict:
    """Run *body* in a new interpreter; it must print one JSON object last."""
    code = _FRESH_PRELUDE.format(tests_dir=str(Path(__file__).parent)) + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.orca
def test_os_environ_changes_reach_children_after_backend_use(scratch_root):
    result = _run_fresh(f"""
        backend = make_backend(Path({str(scratch_root)!r}), nprocs=1, check_version=True)
        backend.evaluate(make_request())
        os.environ["OPENMMORCA_PROBE"] = "visible"
        print(json.dumps({{"probe": child_env().get("OPENMMORCA_PROBE")}}))
    """)
    assert result["probe"] == "visible"


@pytest.mark.orca
def test_orca_run_leaves_child_environment_unchanged(scratch_root, require_mpi):
    result = _run_fresh(f"""
        before = child_env()
        backend = make_backend(Path({str(scratch_root)!r}), nprocs=2)
        backend.evaluate(make_request())
        after = child_env()
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        print(json.dumps({{"changed": changed}}))
    """)
    assert result["changed"] == []


@pytest.mark.orca
def test_check_version_leaves_child_environment_unchanged(scratch_root):
    result = _run_fresh(f"""
        before = child_env()
        backend = make_backend(Path({str(scratch_root)!r}), nprocs=1, check_version=True)
        after = child_env()
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        print(json.dumps({{"changed": changed}}))
    """)
    assert result["changed"] == []
