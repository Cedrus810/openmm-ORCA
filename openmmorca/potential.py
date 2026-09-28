"""User-facing ORCAPotential, mirroring the openmm-ml / openmm-pyscf API (spec §8.1).

``ORCAPotential`` contains no ORCA logic itself: it validates the
configuration, owns backends (one per created System, spec §9.5) and assembles
System + PythonForce from the qmmm/ layer.
"""

from __future__ import annotations

from typing import Callable, Sequence

import openmm as mm
import openmm.app
from openmm import unit

from openmmorca.backend.base import QMBackend
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend
from openmmorca.force import QMMMCallback, make_python_force
from openmmorca.qmmm import build_mixed_system


def topology_masses(topology: openmm.app.Topology) -> list[float]:
    """Particle masses (dalton, plain floats) for every topology atom.

    Rejects atoms without an element (virtual sites): full-QM and ONIOM
    particles must be real atoms.
    """
    masses = []
    for atom in topology.atoms():
        if atom.element is None:
            raise ValueError(
                f"atom {atom.name} (index {atom.index}) has no element; "
                "full-QM systems require real atoms"
            )
        mass = atom.element.mass
        masses.append(
            float(mass.value_in_unit(unit.dalton))
            if hasattr(mass, "value_in_unit")
            else float(mass)
        )
    return masses


class ORCAPotential:
    """OpenMM potential backed by ORCA via OPI.

    A fresh backend (with its own scratch directory) is created for every
    ``createSystem``/``createMixedSystem`` call — one backend serves exactly
    one OpenMM Context (spec §9.5).
    """

    def __init__(
        self,
        method: str,
        basis: str | None = None,
        charge: int = 0,
        multiplicity: int = 1,
        nprocs: int = 1,
        maxcore_mb: int = 2000,
        extra_keywords: Sequence[str] = (),
        extra_blocks: Sequence[str] = (),
        scratch_root: str | None = None,
        restart: bool = True,
        keep_failed: bool = True,
        max_fresh_retries: int = 1,
        timeout_s: float | None = None,
        orca_path: str | None = None,
        backend_factory: Callable[[], QMBackend] | None = None,
    ) -> None:
        self.config = ORCAConfig(
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            nprocs=nprocs,
            maxcore_mb=maxcore_mb,
            extra_keywords=tuple(extra_keywords),
            extra_blocks=tuple(extra_blocks),
            scratch_root=scratch_root,
            restart=restart,
            keep_failed=keep_failed,
            max_fresh_retries=max_fresh_retries,
            timeout_s=timeout_s,
            orca_path=orca_path,
        )
        self._backend_factory = backend_factory
        self.backends: list[QMBackend] = []

    # ------------------------------------------------------------------

    def _create_backend(self) -> QMBackend:
        if self._backend_factory is not None:
            backend = self._backend_factory()
        else:
            backend = ORCAOPIBackend(self.config)
        self.backends.append(backend)
        return backend

    def createSystem(
        self, topology: openmm.app.Topology, removeCMMotion: bool = True
    ) -> mm.System:
        """Build a full-QM System: every topology atom becomes a QM particle."""
        backend = self._create_backend()
        masses = topology_masses(topology)
        elements = [atom.element.symbol for atom in topology.atoms()]
        n_particles = len(masses)
        callback = QMMMCallback(backend, range(n_particles), elements, n_particles)
        system = mm.System()
        for mass in masses:
            system.addParticle(mass)
        system.addForce(make_python_force(callback))
        if removeCMMotion:
            system.addForce(mm.CMMotionRemover())
        return system

    def createMixedSystem(
        self,
        topology: openmm.app.Topology,
        system: mm.System,
        atoms: Sequence[int],
        removeConstraints: bool = True,
        forceGroup: int = 0,
        interpolate: bool = False,
        embedding: str = "electronic",
    ) -> mm.System:
        """Build a QM/MM System: *atoms* (OpenMM indices) are treated by ORCA."""
        if interpolate:
            raise NotImplementedError(
                "interpolated QM/MM potentials are not supported"
            )
        if embedding != "electronic":
            raise ValueError(
                f"unsupported embedding {embedding!r}; supported: {self.getSupportedEmbeddings()}"
            )
        backend = self._create_backend()
        parts = build_mixed_system(topology, system, atoms, remove_constraints=removeConstraints)
        callback = QMMMCallback(
            backend,
            parts.qm_atoms,
            parts.qm_elements,
            system.getNumParticles(),
            parts.mm_atoms,
            parts.mm_charges_e,
            system=parts.system,
        )
        force = make_python_force(callback, force_group=forceGroup)
        parts.system.addForce(force)
        return parts.system

    def getSupportedEmbeddings(self) -> list[str]:
        return ["electronic"]

    def summarize_timings(self) -> list[dict]:
        """Per-backend timing summaries (mean/p50/p95 per phase, cold start skipped)."""
        summaries = []
        for backend in self.backends:
            timings = getattr(backend, "timings", None)
            scratch = getattr(backend, "scratch", None)
            if timings is None or scratch is None:
                continue
            summaries.append(
                {
                    "scratch": str(scratch.root),
                    "n_fresh_retries": backend.n_fresh_retries,
                    **timings.summarize(),
                }
            )
        return summaries

    def close(self) -> None:
        """Close every backend created by this potential (idempotent)."""
        for backend in self.backends:
            if getattr(backend, "_openmmorca_closed", False):
                continue
            backend.close()
            backend._openmmorca_closed = True
