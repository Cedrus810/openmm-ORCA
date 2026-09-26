"""QM backend protocol and request/result data types (spec §7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class QMRequest:
    """One QM evaluation request, in backend-boundary units (nm, e)."""

    qm_elements: tuple[str, ...]
    qm_positions_nm: np.ndarray  # (n_qm, 3)
    mm_positions_nm: np.ndarray | None = None  # (n_mm, 3)
    mm_charges_e: np.ndarray | None = None  # (n_mm,)
    step: int = 0

    def __post_init__(self) -> None:
        qm_positions = np.asarray(self.qm_positions_nm, dtype=float)
        object.__setattr__(self, "qm_positions_nm", qm_positions)
        if qm_positions.shape != (len(self.qm_elements), 3):
            raise ValueError(
                f"qm_positions_nm must have shape ({len(self.qm_elements)}, 3), "
                f"got {qm_positions.shape}"
            )
        if (self.mm_positions_nm is None) != (self.mm_charges_e is None):
            raise ValueError("mm_positions_nm and mm_charges_e must be given together")
        if self.mm_positions_nm is not None:
            mm_positions = np.asarray(self.mm_positions_nm, dtype=float)
            mm_charges = np.asarray(self.mm_charges_e, dtype=float)
            if mm_positions.ndim != 2 or mm_positions.shape[1] != 3:
                raise ValueError(
                    f"mm_positions_nm must have shape (n_mm, 3), got {mm_positions.shape}"
                )
            if mm_positions.shape[0] != mm_charges.shape[0]:
                raise ValueError(
                    f"got {mm_positions.shape[0]} MM positions but {mm_charges.shape[0]} charges"
                )
            object.__setattr__(self, "mm_positions_nm", mm_positions)
            object.__setattr__(self, "mm_charges_e", mm_charges)

    @property
    def n_qm(self) -> int:
        return len(self.qm_elements)

    @property
    def n_mm(self) -> int:
        return 0 if self.mm_positions_nm is None else self.mm_positions_nm.shape[0]


@dataclass(frozen=True)
class QMResult:
    """One QM evaluation result, in backend-boundary units (kJ/mol)."""

    energy_kj_mol: float
    qm_forces_kj_mol_nm: np.ndarray  # (n_qm, 3)
    mm_forces_kj_mol_nm: np.ndarray | None = None  # (n_mm, 3); required when n_mm > 0
    timings_s: dict[str, float] = field(default_factory=dict)


class QMBackend(Protocol):
    """Anything that can turn a QMRequest into a QMResult (fake, ORCA, PySCF, ...)."""

    def evaluate(self, request: QMRequest) -> QMResult: ...

    def close(self) -> None: ...


def check_result(request: QMRequest, result: QMResult) -> None:
    """Validate that *result* is consistent with *request*; raise ValueError otherwise."""
    if not np.isfinite(result.energy_kj_mol):
        raise ValueError("result energy is not finite")
    if result.qm_forces_kj_mol_nm.shape != (request.n_qm, 3):
        raise ValueError(
            f"qm_forces_kj_mol_nm must have shape ({request.n_qm}, 3), "
            f"got {result.qm_forces_kj_mol_nm.shape}"
        )
    if not np.all(np.isfinite(result.qm_forces_kj_mol_nm)):
        raise ValueError("qm_forces_kj_mol_nm contains non-finite values")
    if request.n_mm > 0:
        if result.mm_forces_kj_mol_nm is None:
            raise ValueError(
                "MM forces are required when the request contains point charges "
                "(returning none would break momentum conservation)"
            )
        if result.mm_forces_kj_mol_nm.shape != (request.n_mm, 3):
            raise ValueError(
                f"mm_forces_kj_mol_nm must have shape ({request.n_mm}, 3), "
                f"got {result.mm_forces_kj_mol_nm.shape}"
            )
        if not np.all(np.isfinite(result.mm_forces_kj_mol_nm)):
            raise ValueError("mm_forces_kj_mol_nm contains non-finite values")
