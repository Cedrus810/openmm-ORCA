"""Analytic fake backend for testing the OpenMM layer without ORCA (spec §13.2).

Potential::

    E = sum_{a<b in QM}  1/2 k (|R_a - R_b| - r0_ab)^2
      + sum_{a in QM, j in emb}  k_e q_a^fake q_j / |R_a - R_j|

with r0_ab taken from the *first* evaluate call and
k_e = COULOMB_KJ_MOL_NM. It has an exact analytic gradient, so tests can
verify "QM forces land on the right atoms", "MM reaction forces land on the
right atoms", unit correctness and absence of double counting.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from openmmorca.backend.base import QMRequest, QMResult, check_result
from openmmorca.units import COULOMB_KJ_MOL_NM


class FakeBackend:
    """Analytic QMBackend implementation with exact forces."""

    def __init__(self, qm_charges_e: Sequence[float], k_bond: float = 1000.0) -> None:
        self.qm_charges_e = np.asarray(qm_charges_e, dtype=float)
        self.k_bond = float(k_bond)
        self.n_calls: int = 0
        self.last_request: QMRequest | None = None
        self._r0: np.ndarray | None = None  # (n_qm, n_qm) equilibrium distances

    def evaluate(self, request: QMRequest) -> QMResult:
        if self.qm_charges_e.shape[0] != request.n_qm:
            raise ValueError(
                f"FakeBackend was constructed with {self.qm_charges_e.shape[0]} fake "
                f"charges but the request contains {request.n_qm} QM atoms"
            )
        qm = request.qm_positions_nm
        # Pairwise displacement vectors and distances within the QM region.
        disp = qm[:, np.newaxis, :] - qm[np.newaxis, :, :]  # disp[i, j] = R_i - R_j
        dist = np.linalg.norm(disp, axis=-1)
        if self._r0 is None:
            self._r0 = dist.copy()
        k = self.k_bond

        energy = 0.0
        qm_forces = np.zeros_like(qm)
        n = request.n_qm
        if n > 1:
            iu, ju = np.triu_indices(n, k=1)
            r = dist[iu, ju]
            r0 = self._r0[iu, ju]
            # 1/2 k (r - r0)^2 per pair
            energy += 0.5 * k * float(np.sum((r - r0) ** 2))
            # dE/dR_a = k (r - r0) * (R_a - R_b) / r
            pair_force_mag = k * (r - r0) / r  # (n_pairs,)
            pair_vectors = disp[iu, ju]  # R_i - R_j for pair (i, j)
            forces_flat = np.zeros((n, 3))
            np.add.at(forces_flat, iu, (-pair_force_mag)[:, None] * pair_vectors)
            np.add.at(forces_flat, ju, pair_force_mag[:, None] * pair_vectors)
            qm_forces += forces_flat

        mm_forces = None
        if request.n_mm > 0:
            mm = request.mm_positions_nm
            q_charges = self.qm_charges_e[:, None]  # (n_qm, 1)
            m_charges = request.mm_charges_e[None, :]  # (1, n_mm)
            mm_disp = qm[:, None, :] - mm[None, :, :]  # (n_qm, n_mm, 3): R_a - R_j
            mm_dist = np.linalg.norm(mm_disp, axis=-1)  # (n_qm, n_mm)
            coulomb_energy = COULOMB_KJ_MOL_NM * q_charges * m_charges / mm_dist
            energy += float(np.sum(coulomb_energy))
            # E_aj = k_e q_a q_j / r -> dE/dR_a = -k_e q_a q_j (R_a - R_j) / r^3
            # force on QM a = -dE/dR_a = +k_e q_a q_j (R_a - R_j) / r^3
            coeff = COULOMB_KJ_MOL_NM * q_charges * m_charges / mm_dist**3
            qm_forces += np.sum(coeff[:, :, None] * mm_disp, axis=1)
            mm_forces = -np.sum(coeff[:, :, None] * mm_disp, axis=0)

        result = QMResult(
            energy_kj_mol=energy,
            qm_forces_kj_mol_nm=qm_forces,
            mm_forces_kj_mol_nm=mm_forces,
            timings_s={"total": 0.0},
        )
        check_result(request, result)
        self.n_calls += 1
        self.last_request = request
        return result

    def close(self) -> None:
        pass
