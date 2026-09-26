"""OpenMM-side System modification for QM/MM (spec §6).

The functions here know nothing about ORCA. They take an OpenMM System plus a
QM atom selection and produce the modified System in which:

* all-QM bonded terms are zeroed (energy-neutral removal, spec §6.3);
* QM particles carry no charge, QM–QM pairs carry no LJ, and QM–MM LJ is
  kept (spec §6.2) — electrostatic embedding comes from the QM backend;
* QM-internal constraints are removed on request (spec §6.3).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import openmm as mm
import openmm.app
from openmm import unit

# Forces allowed in an input System (spec §6.2). Anything else is rejected.
SUPPORTED_FORCE_TYPES: tuple[type, ...] = (
    mm.HarmonicBondForce,
    mm.HarmonicAngleForce,
    mm.PeriodicTorsionForce,
    mm.RBTorsionForce,
    mm.CMAPTorsionForce,
    mm.NonbondedForce,
    mm.CMMotionRemover,
    mm.MonteCarloBarostat,
)

_ZERO_CHARGE_TOL = 1e-12


def copy_system(system: mm.System) -> mm.System:
    """Deep-copy a System via an XML round trip."""
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def check_supported_forces(system: mm.System) -> None:
    """Reject Systems containing force types this package cannot handle."""
    unsupported = [
        type(force).__name__
        for force in system.getForces()
        if not isinstance(force, SUPPORTED_FORCE_TYPES)
    ]
    if unsupported:
        raise ValueError(
            "System contains force types that openmmorca cannot handle: "
            f"{sorted(set(unsupported))}. Supported: "
            f"{[cls.__name__ for cls in SUPPORTED_FORCE_TYPES]}"
        )


def get_nonbonded_force(system: mm.System) -> mm.NonbondedForce:
    """Return the one NonbondedForce; raise ValueError otherwise."""
    nonbonded = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    if len(nonbonded) == 0:
        raise ValueError("System must contain a NonbondedForce for QM/MM embedding")
    if len(nonbonded) > 1:
        raise ValueError(
            "System must contain exactly one NonbondedForce, "
            f"found {len(nonbonded)}"
        )
    return nonbonded[0]


def zero_qm_bonded_terms(system: mm.System, qm_atoms) -> int:
    """Zero bonded terms whose atoms are *all* in the QM region (spec §6.3).

    Zeroing force constants is energy- and force-equivalent to removal and
    keeps item indices stable. Returns the number of zeroed terms.
    """
    qm = set(qm_atoms)
    n_zeroed = 0
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                if p1 in qm and p2 in qm:
                    force.setBondParameters(i, p1, p2, length, 0.0)
                    n_zeroed += 1
        elif isinstance(force, mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(i)
                if p1 in qm and p2 in qm and p3 in qm:
                    force.setAngleParameters(i, p1, p2, p3, angle, 0.0)
                    n_zeroed += 1
        elif isinstance(force, mm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                if {p1, p2, p3, p4} <= qm:
                    force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0.0)
                    n_zeroed += 1
        elif isinstance(force, mm.RBTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, *c = force.getTorsionParameters(i)
                if {p1, p2, p3, p4} <= qm:
                    force.setTorsionParameters(i, p1, p2, p3, p4, *(0.0,) * 6)
                    n_zeroed += 1
        elif isinstance(force, mm.CMAPTorsionForce):
            # Terms are redirected to a fresh all-zero map of matching size.
            zero_map_by_size: dict[int, int] = {}
            for i in range(force.getNumTorsions()):
                map_index, *particles = force.getTorsionParameters(i)
                if set(particles) <= qm:
                    size = force.getMapParameters(map_index)[0]
                    if size not in zero_map_by_size:
                        zero_map_by_size[size] = force.addMap(
                            size, [0.0] * (size * size)
                        )
                    force.setTorsionParameters(i, zero_map_by_size[size], *particles)
                    n_zeroed += 1
    return n_zeroed


def apply_qm_embedding(nonbonded: mm.NonbondedForce, qm_atoms) -> dict[int, float]:
    """Apply the spec §6.2 NonbondedForce modification.

    QM charges are zeroed (QM–MM electrostatics therefore disappear), exceptions
    touching a QM atom get zero chargeProd, QM–QM exceptions additionally get
    zero epsilon, and QM–QM pairs without an exception get one (so no QM–QM LJ
    survives). Returns {particle index: original charge} for the zeroed atoms.
    """
    qm = set(qm_atoms)
    removed: dict[int, float] = {}
    for i in sorted(qm):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        removed[i] = charge.value_in_unit(unit.elementary_charge)
        nonbonded.setParticleParameters(i, 0.0, sigma, epsilon)
    for i in range(nonbonded.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nonbonded.getExceptionParameters(i)
        if p1 in qm or p2 in qm:
            new_epsilon = 0.0 if (p1 in qm and p2 in qm) else epsilon
            nonbonded.setExceptionParameters(i, p1, p2, 0.0, sigma, new_epsilon)
    existing = {
        tuple(sorted(nonbonded.getExceptionParameters(i)[:2]))
        for i in range(nonbonded.getNumExceptions())
    }
    qm_sorted = sorted(qm)
    for a_idx, a in enumerate(qm_sorted):
        _, sigma_a, eps_a = nonbonded.getParticleParameters(a)
        for b in qm_sorted[a_idx + 1 :]:
            if (a, b) in existing:
                continue
            _, sigma_b, eps_b = nonbonded.getParticleParameters(b)
            nonbonded.addException(
                a, b, 0.0, 0.5 * (sigma_a + sigma_b), 0.0, True
            )
    return removed


def remove_qm_constraints(system: mm.System, qm_atoms) -> int:
    """Remove constraints with both ends in the QM region; return the count."""
    qm = set(qm_atoms)
    n_removed = 0
    for i in reversed(range(system.getNumConstraints())):
        p1, p2, _distance = system.getConstraintParameters(i)
        if p1 in qm and p2 in qm:
            system.removeConstraint(i)
            n_removed += 1
    return n_removed


def particle_charges(nonbonded: mm.NonbondedForce) -> np.ndarray:
    """Charges (e) of all particles, in particle order, as plain floats."""
    return np.array(
        [
            nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
            for i in range(nonbonded.getNumParticles())
        ]
    )


def check_whole_molecules(topology: openmm.app.Topology, qm_atoms) -> None:
    """Reject QM selections that cut a molecule across a covalent bond."""
    qm = set(qm_atoms)
    for atom1, atom2 in topology.bonds():
        i1, i2 = atom1.index, atom2.index
        if (i1 in qm) != (i2 in qm):
            raise ValueError(
                f"the QM region cuts a covalent bond between atoms {i1} and "
                f"{i2}; the QM region must contain whole molecules (covalent "
                "QM/MM boundaries with link atoms are not supported yet)"
            )


def _qm_side_particles(system: mm.System, qm_atoms, charges: np.ndarray) -> set[int]:
    """QM region including charged virtual sites whose parents are all QM-side.

    A TIP4P-type M site belonging to a QM water molecule must behave as part
    of the QM region: its charge is zeroed and it is never embedded (the QM
    calculation describes that molecule). Virtual sites of MM molecules keep
    their charge and are embedded like ordinary MM particles.
    """
    qm_side = set(qm_atoms)
    changed = True
    while changed:
        changed = False
        for p in range(system.getNumParticles()):
            if p in qm_side or not system.isVirtualSite(p):
                continue
            virtual = system.getVirtualSite(p)
            parents = [
                virtual.getParticle(k) for k in range(virtual.getNumParticles())
            ]
            if parents and all(parent in qm_side for parent in parents):
                if abs(charges[p]) > _ZERO_CHARGE_TOL:
                    qm_side.add(p)
                    changed = True
    return qm_side


@dataclass(frozen=True)
class MixedSystemParts:
    """Result of build_mixed_system: the modified System plus index maps."""

    system: mm.System
    qm_atoms: tuple[int, ...]  # user order preserved
    qm_elements: tuple[str, ...]
    mm_atoms: tuple[int, ...]  # charged MM particles only (|q| > 1e-12)
    mm_charges_e: np.ndarray  # aligned with mm_atoms, original charges
    removed_qm_charges: dict[int, float]


def build_mixed_system(
    topology: openmm.app.Topology,
    system: mm.System,
    qm_atoms,
    remove_constraints: bool = True,
) -> MixedSystemParts:
    """Copy and modify *system* for QM/MM (spec §6, Task 7 ordering)."""
    qm_atoms = list(qm_atoms)
    n_particles = system.getNumParticles()
    if len(qm_atoms) == 0:
        raise ValueError("qm_atoms must not be empty")
    if len(set(qm_atoms)) != len(qm_atoms):
        raise ValueError(f"qm_atoms contains duplicates: {qm_atoms}")
    if any(i < 0 or i >= n_particles for i in qm_atoms):
        raise ValueError(
            f"qm_atoms indices must be in [0, {n_particles}), got {qm_atoms}"
        )

    if system.usesPeriodicBoundaryConditions():
        raise NotImplementedError(
            "periodic QM/MM is planned for M5; this build supports non-periodic systems only"
        )

    check_whole_molecules(topology, qm_atoms)

    new_system = copy_system(system)
    check_supported_forces(new_system)
    nonbonded = get_nonbonded_force(new_system)
    if (
        nonbonded.getNumParticleParameterOffsets() > 0
        or nonbonded.getNumExceptionParameterOffsets() > 0
    ):
        raise ValueError(
            "NonbondedForce parameter offsets (e.g. from free-energy setups) "
            "are not supported"
        )

    # Original charges are needed later for mm_charges_e; snapshot first.
    original_charges = particle_charges(nonbonded)
    qm_side = _qm_side_particles(new_system, qm_atoms, original_charges)

    zero_qm_bonded_terms(new_system, qm_side)
    removed_qm_charges = apply_qm_embedding(nonbonded, qm_side)

    qm_internal_constraints = sum(
        1
        for i in range(new_system.getNumConstraints())
        if set(new_system.getConstraintParameters(i)[:2]) <= set(qm_side)
    )
    if remove_constraints:
        remove_qm_constraints(new_system, qm_side)
    elif qm_internal_constraints:
        warnings.warn(
            f"{qm_internal_constraints} constraints inside the QM region are kept; "
            "constrained QM internals change the QM potential-energy surface. "
            "NVE validation must run with remove_constraints=True.",
            UserWarning,
        )

    topology_atoms = list(topology.atoms())
    qm_elements = []
    for i in qm_atoms:
        element = topology_atoms[i].element
        if element is None:
            raise ValueError(
                f"atom {i} has no element in the topology (virtual site?); "
                "QM atoms must be real atoms"
            )
        qm_elements.append(element.symbol)

    mm_atoms = tuple(
        p
        for p in range(n_particles)
        if p not in qm_side and abs(original_charges[p]) > _ZERO_CHARGE_TOL
    )
    mm_charges = np.array([original_charges[p] for p in mm_atoms])

    return MixedSystemParts(
        system=new_system,
        qm_atoms=tuple(qm_atoms),
        qm_elements=tuple(qm_elements),
        mm_atoms=mm_atoms,
        mm_charges_e=mm_charges,
        removed_qm_charges=removed_qm_charges,
    )
