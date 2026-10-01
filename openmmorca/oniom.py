"""Two-layer ONIOM: subtractive multi-level QM:QM potentials (plan: docs/plans/2026-09-28-oniom.md).

Standard ONIOM (Morokuma) for a high-level model region inside a low-level
full system::

    E_ONIOM = E_high(model) + E_low(full) - E_low(model)

with the corresponding force combination. The layers couple *mechanically*:
the subtractive cross-term cancellation assumes no point-charge embedding
between the two QM levels (embedding the model region would change the
scheme, not merely accelerate it). As in QM/MM, the model region must consist
of whole molecules — covalent boundaries need link atoms, which do not exist
yet.

Every topology atom is a real QM particle at one of the two levels, so the
OpenMM System is a pure particle container (masses + the PythonForce, no
force-field terms). Each MD step performs three QM evaluations (one high,
two low) through three independent backends, so every restart chain (MO-guess
``.gbw`` files) stays correctly sized for its own atom count.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import openmm as mm
import openmm.app
from openmm import unit

from openmmorca.backend.base import QMBackend, QMRequest, check_result
from openmmorca.force import make_python_force
from openmmorca.potential import ORCAPotential, topology_masses
from openmmorca.qmmm.system import check_whole_molecules, validate_atom_indices


class ONIOMCallback:
    """State → three QMRequests → subtractive (energy, forces) combination.

    Forces per particle: low-level full-system forces everywhere, plus the
    high-minus-low correction on the model region — the array OpenMM receives
    is the exact analytic gradient of E_ONIOM whenever each backend returns
    exact gradients.
    """

    def __init__(
        self,
        high_backend: QMBackend,
        low_full_backend: QMBackend,
        low_model_backend: QMBackend,
        model_atoms,
        model_elements,
        all_elements,
        n_particles: int,
    ) -> None:
        self.high_backend = high_backend
        self.low_full_backend = low_full_backend
        self.low_model_backend = low_model_backend
        self.model_atoms = tuple(int(i) for i in model_atoms)
        self.model_elements = tuple(str(e) for e in model_elements)
        self.all_elements = tuple(str(e) for e in all_elements)
        self.n_particles = int(n_particles)
        if len(self.model_elements) != len(self.model_atoms):
            raise ValueError(
                f"got {len(self.model_atoms)} model atoms but "
                f"{len(self.model_elements)} model elements"
            )
        if len(self.all_elements) != self.n_particles:
            raise ValueError(
                f"got {self.n_particles} particles but "
                f"{len(self.all_elements)} full-system elements"
            )
        self.step: int = 0

    def __call__(self, state: mm.State) -> tuple[float, np.ndarray]:
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if positions.shape[0] != self.n_particles:
            raise ValueError(
                f"state has {positions.shape[0]} particles but this callback was "
                f"configured for {self.n_particles}"
            )
        model_idx = list(self.model_atoms)

        request_full = QMRequest(self.all_elements, positions, step=self.step)
        request_model = QMRequest(
            self.model_elements, positions[model_idx], step=self.step
        )
        result_full = self.low_full_backend.evaluate(request_full)
        result_model_low = self.low_model_backend.evaluate(request_model)
        result_model_high = self.high_backend.evaluate(request_model)
        check_result(request_full, result_full)
        check_result(request_model, result_model_low)
        check_result(request_model, result_model_high)

        energy = (
            result_model_high.energy_kj_mol
            + result_full.energy_kj_mol
            - result_model_low.energy_kj_mol
        )
        if not np.isfinite(energy):
            raise ValueError("ONIOM energy is not finite")

        forces = result_full.qm_forces_kj_mol_nm.copy()
        forces[model_idx] += (
            result_model_high.qm_forces_kj_mol_nm
            - result_model_low.qm_forces_kj_mol_nm
        )
        self.step += 1
        return float(energy), forces


class ONIOMPotential:
    """Two-layer ONIOM assembled from two :class:`ORCAPotential` instances.

    ``high``'s charge/multiplicity are the model-region values, ``low``'s the
    full-system values. The low-level model-region evaluation runs the low
    configuration with the *model-region* charge/multiplicity (taken from
    ``high``), so the low-layer-only atoms may be charged or open-shell.

    Every ``createONIOMSystem`` call creates three fresh backends — one high,
    two low (full system, model region) — each with its own scratch directory
    and restart chain; one backend set serves exactly one Context.
    """

    def __init__(self, high: ORCAPotential, low: ORCAPotential) -> None:
        for name, potential in (("high", high), ("low", low)):
            if not isinstance(potential, ORCAPotential):
                raise TypeError(
                    f"{name} must be an ORCAPotential, got "
                    f"{type(potential).__name__}"
                )
        self.high = high
        self.low = low

    def createONIOMSystem(
        self,
        topology: openmm.app.Topology,
        atoms,
        removeCMMotion: bool = True,
        forceGroup: int = 0,
    ) -> mm.System:
        """Build the ONIOM System: *atoms* (OpenMM indices) form the high-level model region."""
        atoms = validate_atom_indices(atoms, topology.getNumAtoms(), name="atoms")
        check_whole_molecules(topology, atoms)
        if topology.getNumBonds() == 0 and len(atoms) < topology.getNumAtoms():
            warnings.warn(
                "the topology has no bonds, so the whole-molecule check cannot "
                "detect a model region that cuts a molecule; add bonds to the "
                "topology (e.g. createStandardBonds or PDB CONECT records)",
                UserWarning,
                stacklevel=2,
            )
        masses = topology_masses(topology)
        elements = [atom.element.symbol for atom in topology.atoms()]
        if len(atoms) == len(elements):
            warnings.warn(
                "the model region covers every atom; the ONIOM energy reduces "
                "to the high level alone",
                UserWarning,
                stacklevel=2,
            )

        high_backend = self.high._create_backend()
        low_full_backend = self.low._create_backend()
        # E_low(model) describes the model region, whose charge/multiplicity
        # are high's — not the full-system values of the low configuration.
        low_model_backend = self.low._create_backend(
            dataclasses.replace(
                self.low.config,
                charge=self.high.config.charge,
                multiplicity=self.high.config.multiplicity,
            )
        )
        callback = ONIOMCallback(
            high_backend,
            low_full_backend,
            low_model_backend,
            atoms,
            [elements[i] for i in atoms],
            elements,
            len(elements),
        )
        system = mm.System()
        for mass in masses:
            system.addParticle(mass)
        system.addForce(make_python_force(callback, force_group=forceGroup, name="ORCA ONIOM"))
        if removeCMMotion:
            system.addForce(mm.CMMotionRemover())
        return system

    def getSupportedEmbeddings(self) -> list[str]:
        """ONIOM layers couple mechanically; there is nothing to embed."""
        return []

    def summarize_timings(self) -> list[dict]:
        """Timing summaries of every backend of both layers."""
        if self.high is self.low:
            return self.high.summarize_timings()
        return self.high.summarize_timings() + self.low.summarize_timings()

    def close(self) -> None:
        """Close both layer potentials (and therefore every backend; idempotent)."""
        self.high.close()
        self.low.close()
