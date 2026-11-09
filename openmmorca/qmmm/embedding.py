"""Periodic electrostatic embedding: cutoff or whole-box (spec §11.1, Task 20, roadmap B4).

Two selection modes share one placement rule: each embedded MM charge group
(residue by default) is kept whole and placed at the image of its atom
nearest the QM region (atom images taken relative to the first QM atom).

* ``cutoff_nm`` a number: only groups with an atom within ``cutoff_nm`` of a
  QM atom (minimum image) are embedded (spec §11.1, the v0.4 default).
* ``cutoff_nm=None``: every group in the box is embedded at that minimum
  image (scheme S2 of ``docs/plans/2026-10-17-b4-embedding-options.md``).
  The embedded charge then equals the total MM charge at every step; a group
  only changes image near the face of the cell centred on the QM anchor.

**Both are approximations**: periodic images beyond the chosen one are
neglected, and groups entering/leaving the sphere (cutoff) or switching
image (whole box) make the energy discontinuous. Neither is PME-consistent.

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
    """Per-step selection of whole MM charge groups around the QM region.

    ``cutoff_nm=None`` embeds every group (whole-box minimum image).
    """

    def __init__(
        self,
        groups: Sequence[tuple[int, ...]],
        charges_e: np.ndarray,
        cutoff_nm: float | None = 1.2,
    ) -> None:
        if cutoff_nm is not None and not cutoff_nm > 0:
            raise ValueError(f"cutoff must be positive or None, got {cutoff_nm}")
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
        self.cutoff_nm = None if cutoff_nm is None else float(cutoff_nm)
        self._atoms = np.array([i for group in self.groups for i in group], dtype=int)
        self._group_of = np.repeat(
            np.arange(len(self.groups)), [len(group) for group in self.groups]
        )
        # Index of each group's first atom within self._atoms.
        self._group_start = np.concatenate(
            [[0], np.cumsum([len(group) for group in self.groups])[:-1]]
        ).astype(int)
        # Previous step: which groups were embedded, and where each group's
        # first atom sat relative to the QM anchor (to detect image switches).
        self._previous = np.zeros(len(self.groups), dtype=bool)
        self._previous_offset = np.zeros((len(self.groups), 3))
        self.last_n_groups: int = 0
        self.last_reimaged: int = 0
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
        if self.cutoff_nm is not None:
            if len(qm) > 1:
                extent = float(
                    np.max(np.linalg.norm(qm[:, None, :] - qm[None, :, :], axis=-1))
                )
            else:
                extent = 0.0
            check_cutoff_against_box(self.cutoff_nm, box, extent)

        if len(self._atoms) == 0:
            return self._finish(
                np.zeros(0, dtype=bool), np.zeros((0, 3)), box,
                np.zeros(0, dtype=int), np.zeros((0, 3)),
            )

        anchor = qm[0]
        positions = np.asarray(positions_nm, dtype=float)[self._atoms]
        # Atom-wise nearest images relative to the QM anchor; with the cutoff
        # bound checked above these give the true minimum-image distance to
        # every QM atom that is within the cutoff.
        near = anchor + minimum_image(positions - anchor, box)
        distance = np.min(np.linalg.norm(near[:, None, :] - qm[None, :, :], axis=-1), axis=1)
        if self.cutoff_nm is None:
            selected = np.ones(len(self.groups), dtype=bool)
        else:
            selected = np.zeros(len(self.groups), dtype=bool)
            selected[self._group_of[distance < self.cutoff_nm]] = True

        # Whole groups: internal geometry from each group's first atom, placed
        # at the image of the group's atom closest to the QM region (first
        # such atom on ties).
        internal = minimum_image(
            positions - positions[self._group_start[self._group_of]], box
        )
        best = np.lexsort((distance, self._group_of))[self._group_start]
        best_of_atom = best[self._group_of]
        placed = near[best_of_atom] + internal - internal[best_of_atom]
        mask = selected[self._group_of]
        return self._finish(
            selected, placed[self._group_start] - anchor, box,
            self._atoms[mask], placed[mask],
        )

    def _finish(self, selected, offsets, box, atoms: np.ndarray, positions: np.ndarray):
        kept = selected & self._previous
        reimaged = 0
        if kept.any():
            # A group kept from the previous step changed image when its offset
            # from the QM anchor jumped by a lattice vector.
            shift = (offsets[kept] - self._previous_offset[kept]) @ np.linalg.inv(box)
            reimaged = int(np.count_nonzero(np.any(np.rint(shift) != 0, axis=1)))
        self.last_n_groups = int(np.count_nonzero(selected))
        self.last_reimaged = reimaged
        self.last_changed = int(np.count_nonzero(selected ^ self._previous)) + reimaged
        self._previous = selected
        self._previous_offset = offsets
        return atoms, positions, self.charges_e[atoms]
