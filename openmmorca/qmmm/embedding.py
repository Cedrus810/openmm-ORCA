"""Cutoff embedding for periodic QM/MM (spec §11.1, Task 20).

The QM region sees only MM charge groups (residues by default) that come
within ``cutoff_nm`` of any QM atom under the minimum-image convention; a
selected group is embedded whole, at the image nearest the QM region.

**This is an approximation**: QM–MM electrostatics beyond the cutoff are
neglected, and groups entering or leaving the sphere make the energy
discontinuous. It is not PME-consistent.

Charges passed in are the final embedding charges: at link-atom boundaries
they are already charge-shifted, and M1 is in no group (prep C8).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import openmm.app

from openmmorca.qmmm.imaging import minimum_image


def perpendicular_widths(box_vectors_nm: np.ndarray) -> np.ndarray:
    """Distances between opposite faces of the periodic cell (nm)."""
    a, b, c = np.asarray(box_vectors_nm, dtype=float)
    volume = abs(float(np.dot(a, np.cross(b, c))))
    return np.array(
        [
            volume / np.linalg.norm(np.cross(b, c)),
            volume / np.linalg.norm(np.cross(c, a)),
            volume / np.linalg.norm(np.cross(a, b)),
        ]
    )


def check_cutoff_against_box(
    cutoff_nm: float, box_vectors_nm: np.ndarray, qm_extent_nm: float = 0.0
) -> None:
    """Require cutoff + QM extent < half the narrowest cell width.

    Selection uses "any group atom within the cutoff of any QM atom"; the
    nearest image of a group is unique only below that bound (prep C9).
    """
    half_width = 0.5 * float(perpendicular_widths(box_vectors_nm).min())
    if cutoff_nm + qm_extent_nm >= half_width:
        detail = f" + QM extent {qm_extent_nm:.3f} nm" if qm_extent_nm else ""
        raise ValueError(
            f"embedding cutoff {cutoff_nm:.3f} nm{detail} must stay below half "
            f"the narrowest box width ({half_width:.3f} nm)"
        )


def groups_from_topology(topology: openmm.app.Topology, mm_atoms) -> list[tuple[int, ...]]:
    """Group the embedded MM atoms by residue (in topology order)."""
    embedded = set(int(i) for i in mm_atoms)
    groups = []
    for residue in topology.residues():
        group = tuple(a.index for a in residue.atoms() if a.index in embedded)
        if group:
            groups.append(group)
    return groups


class CutoffEmbedding:
    """Per-step selection of whole MM charge groups within a cutoff of the QM region."""

    def __init__(
        self,
        groups: Sequence[tuple[int, ...]],
        charges_e: np.ndarray,
        cutoff_nm: float = 1.2,
    ) -> None:
        if cutoff_nm <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff_nm}")
        self.groups = [tuple(int(i) for i in group) for group in groups]
        seen: set[int] = set()
        for group in self.groups:
            if not group:
                raise ValueError("embedding groups must not be empty")
            if seen & set(group):
                raise ValueError(
                    f"atoms {sorted(seen & set(group))} belong to more than one group"
                )
            seen.update(group)
        self.charges_e = np.asarray(charges_e, dtype=float)
        self.cutoff_nm = float(cutoff_nm)
        self._atoms = np.array([i for group in self.groups for i in group], dtype=int)
        self._group_of = np.repeat(
            np.arange(len(self.groups)), [len(group) for group in self.groups]
        )
        # Index of each group's first atom within self._atoms.
        self._group_start = np.concatenate(
            [[0], np.cumsum([len(group) for group in self.groups])[:-1]]
        ).astype(int)
        self._previous: set[int] = set()
        self.last_n_groups: int = 0
        self.last_changed: int = 0

    def select(
        self,
        qm_positions_nm: np.ndarray,
        positions_nm: np.ndarray,
        box_vectors_nm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(selected OpenMM indices, minimum-image positions, charges).

        *qm_positions_nm* must already be whole (``make_qm_whole``); returned
        positions are in the same frame.
        """
        qm = np.asarray(qm_positions_nm, dtype=float)
        box = np.asarray(box_vectors_nm, dtype=float)
        if len(qm) > 1:
            extent = float(
                np.max(np.linalg.norm(qm[:, None, :] - qm[None, :, :], axis=-1))
            )
        else:
            extent = 0.0
        check_cutoff_against_box(self.cutoff_nm, box, extent)

        if len(self._atoms) == 0:
            return self._finish(set(), np.zeros(0, dtype=int), np.zeros((0, 3)))

        anchor = qm[0]
        positions = np.asarray(positions_nm, dtype=float)[self._atoms]
        # Atom-wise nearest images relative to the QM anchor; with the bound
        # checked above these give the true minimum-image distance to every
        # QM atom that is within the cutoff.
        near = anchor + minimum_image(positions - anchor, box)
        distance = np.min(np.linalg.norm(near[:, None, :] - qm[None, :, :], axis=-1), axis=1)
        inside = distance < self.cutoff_nm
        selected_groups = sorted(set(self._group_of[inside].tolist()))

        # Whole groups: internal geometry from each group's first atom, placed
        # at the image of the group's atom closest to the QM region.
        internal = minimum_image(
            positions - positions[self._group_start[self._group_of]], box
        )
        out_atoms, out_positions = [], []
        for g in selected_groups:
            members = np.nonzero(self._group_of == g)[0]
            best = members[np.argmin(distance[members])]
            placed = near[best] + internal[members] - internal[best]
            out_atoms.append(self._atoms[members])
            out_positions.append(placed)
        if not selected_groups:
            return self._finish(set(), np.zeros(0, dtype=int), np.zeros((0, 3)))
        return self._finish(
            set(selected_groups), np.concatenate(out_atoms), np.vstack(out_positions)
        )

    def _finish(self, selected: set[int], atoms: np.ndarray, positions: np.ndarray):
        self.last_n_groups = len(selected)
        self.last_changed = len(selected ^ self._previous)
        self._previous = selected
        return atoms, positions, self.charges_e[atoms]
