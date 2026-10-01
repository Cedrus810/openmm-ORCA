"""Embedding charges at covalent QM/MM boundaries (spec §10.4, Task 17).

The M1 point charge sits ~0.44 Å from its link atom and would over-polarize
the QM density, so it is removed from the embedding and spread evenly over
the MM atoms bonded to M1 (the M2 set). Only the charges handed to the QM
code change; OpenMM's MM–MM electrostatics are untouched.

This is a *simplified* charge shift: the literature scheme (Sherwood et al.,
ChemShell) additionally places point-charge pairs near each M2 to restore the
M1–M2 bond dipoles; that correction is not applied here. RCD-type schemes may
be added later as further strategy classes.

A QM region cut out of force-field residues usually carries a non-integer
force-field charge x_r per cut residue r, leaving the embedding with a
non-integer net charge. ``residue_charge_corrections`` adds
δ_r = x_r − round(x_r) to the M2 atoms of that residue, so each cut residue's
MM remainder carries the integer charge Z_r − round(x_r). The correction is
local and independent of the user's QM charge.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import openmm.app

from openmmorca.qmmm.linkatoms import BoundaryPair


class ChargeShift:
    """Remove each M1 from the embedding; add q_M1 / n_M2 to every M2."""

    def embedding_charges(
        self,
        topology: openmm.app.Topology,
        mm_atoms: Sequence[int],
        mm_charges_e: np.ndarray,
        boundary_pairs: Sequence[BoundaryPair],
    ) -> tuple[tuple[int, ...], np.ndarray]:
        """New (mm_atoms, charges), sorted by particle index.

        M2 atoms that end up with zero charge — or had zero charge and were
        therefore absent from *mm_atoms* — are kept in the embedding.
        """
        charges = {int(a): float(q) for a, q in zip(mm_atoms, mm_charges_e)}
        m1_atoms = {pair.m1 for pair in boundary_pairs}
        q1_atoms = {pair.q1 for pair in boundary_pairs}
        neighbours: dict[int, set[int]] = {}
        for atom1, atom2 in topology.bonds():
            neighbours.setdefault(atom1.index, set()).add(atom2.index)
            neighbours.setdefault(atom2.index, set()).add(atom1.index)

        for pair in boundary_pairs:
            # M2 = MM neighbours of M1, excluding other boundaries' M1 (their
            # own charge is being removed). With validated boundary pairs every
            # QM neighbour of an M1 is the q1 of a declared pair (any other
            # M1–QM bond would be an undeclared cut), so excluding q1 atoms
            # excludes the whole QM side.
            m2 = sorted(
                n
                for n in neighbours.get(pair.m1, ())
                if n not in m1_atoms and n not in q1_atoms
            )
            if not m2:
                raise ValueError(
                    f"boundary atom m1={pair.m1} has no MM neighbours to receive "
                    "its charge (charge shift needs at least one M2 atom)"
                )
            shifted = charges.pop(pair.m1, 0.0) / len(m2)
            for atom in m2:
                charges[atom] = charges.get(atom, 0.0) + shifted
        for m1 in m1_atoms:
            charges.pop(m1, None)

        atoms = tuple(sorted(charges))
        return atoms, np.array([charges[a] for a in atoms])


# A cut residue whose QM part is this close to a half-integer is ambiguous.
_AMBIGUOUS_RESIDUAL = 0.25


def residue_charge_corrections(
    topology: openmm.app.Topology,
    qm_side: set[int],
    charges_e: np.ndarray,
    boundary_pairs: Sequence[BoundaryPair],
) -> tuple[dict[int, float], int]:
    """Per-atom embedding corrections for cut residues, and the QM formal charge.

    *charges_e* are the force-field charges by particle index, *qm_side* the
    QM particles (including QM-side virtual sites). For every residue with
    both QM and MM atoms, δ_r = x_r − round(x_r) (x_r: force-field charge of
    its QM part) is spread evenly over the residue's M2 atoms — or, if the
    residue holds no M2, over its other MM atoms except M1. The formal charge
    is Σ round(x_r) over all residues touching the QM region: the charge the
    force field implies for the QM region (plus link H).
    """
    m1_atoms = {pair.m1 for pair in boundary_pairs}
    m2_atoms: set[int] = set()
    neighbours: dict[int, set[int]] = {}
    for atom1, atom2 in topology.bonds():
        neighbours.setdefault(atom1.index, set()).add(atom2.index)
        neighbours.setdefault(atom2.index, set()).add(atom1.index)
    q1_atoms = {pair.q1 for pair in boundary_pairs}
    for pair in boundary_pairs:
        m2_atoms.update(
            n for n in neighbours.get(pair.m1, ()) if n not in m1_atoms and n not in q1_atoms
        )

    corrections: dict[int, float] = {}
    formal_charge = 0
    for residue in topology.residues():
        atoms = [a.index for a in residue.atoms()]
        qm_part = [i for i in atoms if i in qm_side]
        if not qm_part:
            continue
        x = float(sum(charges_e[i] for i in qm_part))
        n = int(round(x))
        formal_charge += n
        mm_part = [i for i in atoms if i not in qm_side]
        if not mm_part:
            continue  # whole residue in the QM region
        delta = x - n
        if abs(delta) > _AMBIGUOUS_RESIDUAL:
            warnings.warn(
                f"the QM part of residue {residue.name}{residue.id} carries "
                f"{x:+.4f} e, close to a half-integer: rounding to {n:+d} e is "
                "ambiguous; check the QM selection",
                UserWarning,
                stacklevel=3,
            )
        targets = [i for i in mm_part if i in m2_atoms]
        if not targets:
            targets = [i for i in mm_part if i not in m1_atoms]
        if not targets:
            raise ValueError(
                f"residue {residue.name}{residue.id} has no MM atom besides M1 to "
                "carry its residual charge"
            )
        for i in targets:
            corrections[i] = corrections.get(i, 0.0) + delta / len(targets)
    return corrections, formal_charge
