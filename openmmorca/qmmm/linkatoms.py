"""H link atoms at covalent QM/MM boundaries (spec §10, Task 16).

A boundary is a user-declared bond Q1–M1 (Q1 in the QM region, M1 in the MM
region). The QM calculation sees a hydrogen cap L on that bond::

    R_L = R_Q1 + g (R_M1 - R_Q1)

with a fixed ratio g. L is not an OpenMM particle; its force is split onto the
two boundary atoms by the chain rule (exact for fixed g)::

    F_Q1 += (1 - g) F_L
    F_M1 += g F_L

Only single-bond boundaries are supported, and each Q1 carries at most one
link atom (prep D2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import openmm.app

from openmmorca.qmmm.system import check_whole_molecules

# Equilibrium bond lengths (Å) behind the default ratios g = d(Q1–H) / d(Q1–M1).
_DEFAULT_LINK_RATIOS: dict[tuple[str, str], float] = {
    ("C", "C"): 1.09 / 1.526,
    ("C", "N"): 1.09 / 1.449,
}


@dataclass(frozen=True)
class BoundaryPair:
    """One QM/MM boundary bond: link atom at R_Q1 + g (R_M1 - R_Q1)."""

    q1: int  # QM boundary atom (OpenMM index)
    m1: int  # MM boundary atom (OpenMM index)
    g: float

    def __post_init__(self) -> None:
        if not 0.0 < self.g < 1.0:
            raise ValueError(f"link ratio g must be in (0, 1), got {self.g}")


def default_link_ratio(element_q1: str, element_m1: str) -> float:
    """Default g for a Q1–M1 element pair; unknown pairs need an explicit ratio."""
    try:
        return _DEFAULT_LINK_RATIOS[(element_q1, element_m1)]
    except KeyError:
        raise ValueError(
            f"no default link ratio for a {element_q1}–{element_m1} boundary; "
            f"supported: {sorted('–'.join(k) for k in _DEFAULT_LINK_RATIOS)}. "
            "Pass the ratio explicitly (linkRatios)."
        ) from None


def make_boundary_pairs(
    topology: openmm.app.Topology,
    pairs: Sequence[tuple[int, int]],
    ratios: Sequence[float] | None = None,
) -> tuple[BoundaryPair, ...]:
    """BoundaryPairs from (q1, m1) index pairs, with default ratios unless given."""
    pairs = [(int(q1), int(m1)) for q1, m1 in pairs]
    if ratios is not None and len(ratios) != len(pairs):
        raise ValueError(
            f"got {len(pairs)} boundary pairs but {len(ratios)} linkRatios"
        )
    atoms = list(topology.atoms())
    n_atoms = len(atoms)
    result = []
    for k, (q1, m1) in enumerate(pairs):
        for name, index in (("q1", q1), ("m1", m1)):
            if not 0 <= index < n_atoms:
                raise ValueError(
                    f"boundary atom {name}={index} is out of range [0, {n_atoms})"
                )
            if atoms[index].element is None:
                raise ValueError(
                    f"boundary atom {name}={index} has no element (virtual site?)"
                )
        if ratios is None:
            g = default_link_ratio(atoms[q1].element.symbol, atoms[m1].element.symbol)
        else:
            g = float(ratios[k])
        result.append(BoundaryPair(q1=q1, m1=m1, g=g))
    return tuple(result)


def check_boundary_pairs(
    topology: openmm.app.Topology, qm_atoms, pairs: Sequence[BoundaryPair]
) -> None:
    """Validate boundary pairs against the topology and the QM selection.

    q1 must be QM and m1 MM, the two must be bonded, each q1 appears at most
    once, and no bond other than the declared ones crosses the boundary.
    """
    qm = set(int(i) for i in qm_atoms)
    bonds = {
        frozenset((atom1.index, atom2.index)) for atom1, atom2 in topology.bonds()
    }
    seen_q1: set[int] = set()
    for pair in pairs:
        if pair.q1 not in qm:
            raise ValueError(
                f"boundary atom q1={pair.q1} is not in the QM region"
            )
        if pair.m1 in qm:
            raise ValueError(
                f"boundary atom m1={pair.m1} is in the QM region; m1 must be an MM atom"
            )
        if frozenset((pair.q1, pair.m1)) not in bonds:
            raise ValueError(
                f"boundary atoms q1={pair.q1} and m1={pair.m1} are not bonded in the topology"
            )
        if pair.q1 in seen_q1:
            raise ValueError(
                f"QM atom {pair.q1} carries more than one boundary; each q1 may "
                "have at most one link atom"
            )
        seen_q1.add(pair.q1)
    check_whole_molecules(
        topology, qm, allowed_bonds=[(pair.q1, pair.m1) for pair in pairs]
    )


class LinkAtomManager:
    """Link-atom positions and chain-rule force redistribution."""

    def __init__(self, pairs: Sequence[BoundaryPair]) -> None:
        self.pairs = tuple(pairs)
        self._q1 = np.array([p.q1 for p in self.pairs], dtype=int)
        self._m1 = np.array([p.m1 for p in self.pairs], dtype=int)
        self._g = np.array([p.g for p in self.pairs], dtype=float)

    @property
    def n_links(self) -> int:
        return len(self.pairs)

    def link_positions(self, positions_nm: np.ndarray) -> np.ndarray:
        """(n_links, 3) link-atom positions from the full particle positions."""
        r_q1 = positions_nm[self._q1]
        r_m1 = positions_nm[self._m1]
        return r_q1 + self._g[:, None] * (r_m1 - r_q1)

    def redistribute(self, forces: np.ndarray, link_forces: np.ndarray) -> None:
        """Add the link-atom forces onto Q1 and M1 (in place)."""
        link_forces = np.asarray(link_forces, dtype=float)
        if link_forces.shape != (self.n_links, 3):
            raise ValueError(
                f"link_forces must have shape ({self.n_links}, 3), got {link_forces.shape}"
            )
        np.add.at(forces, self._q1, (1.0 - self._g)[:, None] * link_forces)
        np.add.at(forces, self._m1, self._g[:, None] * link_forces)
