"""Tests for the ORCA/OPI backend (implementation plan Task 4)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from openmmorca.backend.orca_files import read_pcgrad
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend
from openmmorca.backend.base import QMRequest
from openmmorca.errors import ORCACalculationError
from openmmorca.units import gradient_to_forces

DATA_DIR = Path(__file__).parent / "data"

WATER_ELEMENTS = ("O", "H", "H")
WATER_NM = np.array([[0.0, 0.0, 0.0], [0.096, 0.0, 0.0], [-0.024, 0.093, 0.0]])
PC_CHARGES = np.array([-0.834, 0.417])
PC_NM = np.array([[0.30, 0.0, 0.0], [0.35, 0.08, 0.0]])

HF_DEF2SVP = dict(method="HF", basis="def2-SVP")


def make_backend(scratch_root, **config_overrides) -> ORCAOPIBackend:
    config = ORCAConfig(scratch_root=str(scratch_root), **config_overrides)
    return ORCAOPIBackend(config)


def water_request(n_mm: int = 0, step: int = 0) -> QMRequest:
    return QMRequest(
        qm_elements=WATER_ELEMENTS,
        qm_positions_nm=WATER_NM.copy(),
        mm_positions_nm=PC_NM.copy() if n_mm else None,
        mm_charges_e=PC_CHARGES.copy() if n_mm else None,
        step=step,
    )


# ------------------------------------------------------------------
# Config validation (no ORCA needed)
# ------------------------------------------------------------------


def test_config_rejects_task_keywords():
    for keyword in ("Opt", "engrad"):
        with pytest.raises(ValueError):
            ORCAConfig(method="HF", basis="def2-SVP", extra_keywords=(keyword,))


def test_config_rejects_managed_blocks():
    for block in ('%pal nprocs 4 end', '  %PointCharges "x"'):
        with pytest.raises(ValueError):
            ORCAConfig(method="HF", basis="def2-SVP", extra_blocks=(block,))


# ------------------------------------------------------------------
# Backend behaviour (needs ORCA)
# ------------------------------------------------------------------


@pytest.mark.orca
def test_full_qm_matches_reference(scratch_root):
    backend = make_backend(scratch_root, **HF_DEF2SVP)
    result = backend.evaluate(water_request())
    energy_eh = result.energy_kj_mol / 2625.4996394799
    assert energy_eh == pytest.approx(-75.960838761435, abs=1e-8)
    assert result.mm_forces_kj_mol_nm is None
    inp_text = (backend.scratch.current / "qm.inp").read_text().lower()
    assert "%pointcharges" not in inp_text
    backend.close()


@pytest.mark.orca
def test_point_charges_match_fixture(scratch_root):
    backend = make_backend(scratch_root, **HF_DEF2SVP)
    result = backend.evaluate(water_request(n_mm=2))
    energy_eh = result.energy_kj_mol / 2625.4996394799
    assert energy_eh == pytest.approx(-75.977160671564, abs=1e-8)
    fixture_pcgrad = read_pcgrad(DATA_DIR / "h2o_2pc.pcgrad", n_charges=2)
    expected_mm_forces = gradient_to_forces(fixture_pcgrad)
    # 1e-8 Eh/bohr expressed in kJ/mol/nm
    np.testing.assert_allclose(
        result.mm_forces_kj_mol_nm, expected_mm_forces, atol=1e-8 * 49614.752589
    )
    backend.close()


@pytest.mark.orca
def test_input_file_contents(scratch_root):
    backend = make_backend(scratch_root, **HF_DEF2SVP)
    backend.evaluate(water_request(n_mm=2))
    inp_text = (backend.scratch.current / "qm.inp").read_text()
    assert "engrad" in inp_text.lower()
    assert '%pointcharges "pc.pc"' in inp_text
    assert "jsonpropfile" in inp_text
    assert "jsongbwfile" not in inp_text
    # No inline-Q point charges in the coordinate block.
    in_coords_block = False
    for line in inp_text.splitlines():
        if line.strip().startswith("*"):
            in_coords_block = not in_coords_block
            continue
        if in_coords_block and line.strip().upper().startswith("Q"):
            pytest.fail(f"inline Q line in coordinate block: {line!r}")
    backend.close()


@pytest.mark.orca
def test_scf_failure_raises(scratch_root):
    backend = make_backend(
        scratch_root, **HF_DEF2SVP, extra_blocks=("%scf maxiter 2 end",)
    )
    with pytest.raises(ORCACalculationError):
        backend.evaluate(water_request())
    backend.close()


@pytest.mark.orca
def test_stale_engrad_is_never_read(scratch_root):
    backend = make_backend(scratch_root, **HF_DEF2SVP)
    backend.evaluate(water_request())
    backend.config = dataclasses.replace(
        backend.config,
        extra_blocks=("%scf maxiter 2 end",),
        restart=False,
    )
    shifted = WATER_NM.copy()
    shifted[1, 0] += 0.0001
    request = QMRequest(
        qm_elements=WATER_ELEMENTS,
        qm_positions_nm=shifted,
        step=1,
    )
    with pytest.raises(ORCACalculationError):
        backend.evaluate(request)
    backend.close()


@pytest.mark.orca
def test_impossible_multiplicity_rejected(scratch_root):
    backend = make_backend(scratch_root, **HF_DEF2SVP, multiplicity=2)
    with pytest.raises(ValueError) as excinfo:
        backend.evaluate(water_request())
    message = str(excinfo.value)
    assert "10 electrons" in message
    assert "multiplicity 2" in message
    backend.close()


@pytest.mark.orca
def test_timings_reported(scratch_root):
    backend = make_backend(scratch_root, **HF_DEF2SVP)
    result = backend.evaluate(water_request())
    for key in ("write", "orca", "read", "total"):
        assert key in result.timings_s
        assert result.timings_s[key] >= 0.0
    backend.close()


@pytest.mark.orca
def test_close_removes_scratch(scratch_root):
    backend = make_backend(scratch_root, **HF_DEF2SVP)
    backend.evaluate(water_request())
    backend.close()
    assert not backend.scratch.current.exists()
    backend.close()  # idempotent


@pytest.mark.orca
def test_xtb_runs(scratch_root):
    backend = make_backend(scratch_root, method="XTB", basis=None)
    result = backend.evaluate(water_request(n_mm=2))
    assert np.isfinite(result.energy_kj_mol)
    assert np.all(np.isfinite(result.qm_forces_kj_mol_nm))
    assert np.all(np.isfinite(result.mm_forces_kj_mol_nm))
    backend.close()
