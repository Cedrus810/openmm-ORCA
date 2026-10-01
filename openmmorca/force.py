"""PythonForce callback wiring (spec §2.8, §5.1 force layer).

The callback turns an OpenMM State into a QMRequest, asks the backend for
energy and forces, and scatters the forces back onto the full particle array
by OpenMM atom index. It performs no physics beyond unit conversion.

Known limitations (verified on OpenMM 8.5.2, see spec §2.8):

* A System holding a PythonForce cannot be XML-serialized (the callback is a
  live Python object).
* Exceptions raised inside the callback surface from ``Context.getState`` as
  ``openmm.OpenMMException`` carrying the original message — OpenMM wraps the
  Python exception.
* OpenMM does **not** redistribute forces returned by a PythonForce for
  virtual sites (verified empirically): standard force kernels spread the
  force of a massless site onto its parents, but the array returned by
  PythonForce is taken literally, leaving a nonzero force on a massless
  particle. The callback therefore redistributes virtual-site forces itself
  (average sites only; other site types are rejected).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import openmm as mm
from openmm import unit

from openmmorca.backend.base import QMBackend, QMRequest, check_result
from openmmorca.qmmm.embedding import CutoffEmbedding
from openmmorca.qmmm.imaging import make_qm_whole
from openmmorca.qmmm.linkatoms import LinkAtomManager

logger = logging.getLogger(__name__)


def _vsite_redistribution_map(system: mm.System) -> list[tuple[int, list[tuple[int, float]]]]:
    """(vsite index, [(parent index, weight), ...]) for every virtual site.

    Only weighted-average sites (Two-/ThreeParticleAverageSite, OutOfPlaneSite)
    are supported; their force distribution is a plain weighted sum.
    """
    mapping: list[tuple[int, list[tuple[int, float]]]] = []
    for p in range(system.getNumParticles()):
        if not system.isVirtualSite(p):
            continue
        site = system.getVirtualSite(p)
        if not isinstance(
            site,
            (mm.TwoParticleAverageSite, mm.ThreeParticleAverageSite, mm.OutOfPlaneSite),
        ):
            raise NotImplementedError(
                f"virtual site of type {type(site).__name__} on particle {p} is "
                "not supported (only weighted-average sites are)"
            )
        parents = [
            (site.getParticle(k), site.getWeight(k))
            for k in range(site.getNumParticles())
        ]
        mapping.append((p, parents))
    return mapping


class QMMMCallback:
    """State → QMRequest → backend → (energy kJ/mol, forces (N, 3) kJ/mol/nm)."""

    def __init__(
        self,
        backend: QMBackend,
        qm_atoms: Sequence[int],
        qm_elements: Sequence[str],
        n_particles: int,
        mm_atoms: Sequence[int] = (),
        mm_charges_e: Sequence[float] = (),
        system: mm.System | None = None,
        link_manager: LinkAtomManager | None = None,
        embedding: CutoffEmbedding | None = None,
        qm_graph: dict[int, list[int]] | None = None,
    ) -> None:
        self.backend = backend
        self.qm_atoms = tuple(int(i) for i in qm_atoms)
        self.qm_elements = tuple(str(e) for e in qm_elements)
        self.n_particles = int(n_particles)
        self.mm_atoms = tuple(int(i) for i in mm_atoms)
        self.mm_charges_e = np.asarray(mm_charges_e, dtype=float)
        if len(self.mm_charges_e) != len(self.mm_atoms):
            raise ValueError(
                f"got {len(self.mm_atoms)} MM atoms but {len(self.mm_charges_e)} MM charges"
            )
        self._vsite_redistribution = (
            _vsite_redistribution_map(system) if system is not None else []
        )
        self.link_manager = link_manager
        if (embedding is None) != (qm_graph is None):
            raise ValueError("embedding and qm_graph must be given together")
        # Periodic mode: re-image the QM region (and boundary M1) each step and
        # embed only the MM groups within the cutoff (spec §11).
        self.embedding = embedding
        self.qm_graph = qm_graph
        n_links = link_manager.n_links if link_manager is not None else 0
        # Link atoms are hydrogen caps appended after the QM atoms.
        self._request_elements = self.qm_elements + ("H",) * n_links
        self.step: int = 0

    def __call__(self, state: mm.State) -> tuple[float, np.ndarray]:
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if positions.shape[0] != self.n_particles:
            raise ValueError(
                f"state has {positions.shape[0]} particles but this callback was "
                f"configured for {self.n_particles}"
            )

        box = None
        if self.embedding is not None:
            box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            positions = make_qm_whole(positions, self.qm_atoms, self.qm_graph, box)

        qm_positions = positions[list(self.qm_atoms)]
        if self.link_manager is not None:
            qm_positions = np.vstack(
                [qm_positions, self.link_manager.link_positions(positions)]
            )
        mm_atoms: Sequence[int] = self.mm_atoms
        mm_positions = None
        mm_charges = None
        diagnostics: dict = {}
        if self.embedding is not None:
            mm_atoms, mm_positions, mm_charges = self.embedding.select(
                positions[list(self.qm_atoms)], positions, box
            )
            diagnostics = {
                "n_embed_groups": self.embedding.last_n_groups,
                "embed_changed": self.embedding.last_changed,
            }
            if self.embedding.last_changed:
                logger.debug(
                    "step %d: %d embedding groups entered/left the cutoff (%d embedded)",
                    self.step,
                    self.embedding.last_changed,
                    self.embedding.last_n_groups,
                )
            if len(mm_atoms) == 0:
                mm_positions = mm_charges = None
        elif self.mm_atoms:
            mm_positions = positions[list(self.mm_atoms)]
            mm_charges = self.mm_charges_e
        request = QMRequest(
            qm_elements=self._request_elements,
            qm_positions_nm=qm_positions,
            mm_positions_nm=mm_positions,
            mm_charges_e=mm_charges,
            step=self.step,
            diagnostics=diagnostics,
        )
        result = self.backend.evaluate(request)
        check_result(request, result)

        forces = np.zeros((self.n_particles, 3))
        n_qm = len(self.qm_atoms)
        forces[list(self.qm_atoms)] = result.qm_forces_kj_mol_nm[:n_qm]
        if self.link_manager is not None:
            self.link_manager.redistribute(forces, result.qm_forces_kj_mol_nm[n_qm:])
        if request.n_mm > 0:
            forces[list(mm_atoms)] += result.mm_forces_kj_mol_nm
        # OpenMM takes the PythonForce force array literally (no virtual-site
        # redistribution), so move site forces onto the parent atoms here.
        for vsite, parents in self._vsite_redistribution:
            site_force = forces[vsite].copy()
            forces[vsite] = 0.0
            for parent, weight in parents:
                forces[parent] += weight * site_force
        self.step += 1
        return float(result.energy_kj_mol), forces


def make_python_force(
    callback: QMMMCallback,
    force_group: int = 0,
    name: str = "ORCA QM/MM",
    periodic: bool = False,
) -> mm.PythonForce:
    """Wrap *callback* in an OpenMM PythonForce with the given force group."""
    force = mm.PythonForce(callback)
    force.setForceGroup(force_group)
    force.setName(name)
    force.setUsesPeriodicBoundaryConditions(periodic)
    return force
