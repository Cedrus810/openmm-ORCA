"""ORCA/OPI QM backend (spec §8, implementation plan Task 4).

Every ``evaluate`` call wipes ``current/``, writes a fresh ``qm.inp`` plus the
``%pointcharges`` file, runs ORCA through the OPI Runner, validates the output
(spec §8.5) and reads energy/forces from ``.engrad`` / ``.pcgrad``.

Facts this implementation relies on (verified, spec §2):

* ORCA's return code is untrustworthy — success is decided from the
  ``.out`` file (``ORCA TERMINATED NORMALLY`` + SCF ``SUCCESS``).
* Embedding charges go through an external ``%pointcharges`` file, never
  inline ``Q`` lines (inline Q double-counts MM–MM electrostatics and
  produces no pcgrad).
* ``.engrad``/``.pcgrad`` are the primary data source; the OPI property JSON
  is used only to cross-check the energy and to derive diagnostics.
* A ``%moinp`` file must exist when ``input.moinp`` is assigned and must not
  share the basename (ORCA overwrites ``qm.gbw`` before reading it) — restart
  (Task 13) therefore stages the guess as ``guess.gbw``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import re
import subprocess
import threading
import time
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from opi.core import Calculator
from opi.execution.core import Runner
from opi.input.structures import Atom, Structure
from opi.output.core import Output

from openmmorca.backend.base import QMRequest, QMResult, check_result
from openmmorca.backend.orca_files import read_engrad, read_pcgrad, write_pointcharges
from openmmorca.errors import ORCACalculationError, ORCAOutputError, ORCATimeoutError
from openmmorca.runtime.diagnostics import TimingLog, write_failure_bundle
from openmmorca.runtime.restart import RestartState
from openmmorca.runtime.scratch import ScratchDir
from openmmorca.units import gradient_to_forces, hartree_to_kj_mol, nm_to_angstrom

# Task keywords that users may not request (the backend always computes EnGrad).
_FORBIDDEN_KEYWORDS = frozenset(
    {"sp", "opt", "freq", "numfreq", "md", "engrad", "numgrad", "moread"}
)
# % blocks managed by the backend itself.
_FORBIDDEN_BLOCK_PREFIXES = (
    "%pointcharges",
    "%pal",
    "%maxcore",
    "%moinp",
    "%output",
)

_ENERGY_CROSSCHECK_TOL_EH = 1e-8
_SCF_CYCLES_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")

# MPI settings required to keep ORCA's per-module startup sane (spec §2.2):
# with OpenMPI's defaults each ORCA module spends ~10 s in MPI setup; these
# variables (shared-memory transport only) cut that to ~1.4 s.
_MPI_ENV_NPROCS_GREATER_1 = {
    "OMPI_MCA_pml": "ob1",
    "OMPI_MCA_btl": "self,sm",
    "OMPI_MCA_mtl": "^ofi",
    "OMPI_MCA_osc": "^ucx",
}


@contextlib.contextmanager
def _orca_mpi_env(nprocs: int):
    """Temporarily set OMP/MPI variables while ORCA runs.

    Always sets OMP_NUM_THREADS=1; adds the OMPI_MCA tuning for nprocs > 1.
    Variables the user already set in the environment are left untouched
    (cluster admins can override everything from the outside).
    """
    wanted = {"OMP_NUM_THREADS": "1"}
    if nprocs > 1:
        wanted.update(_MPI_ENV_NPROCS_GREATER_1)
    saved: dict[str, str | None] = {}
    for key, value in wanted.items():
        current = os.environ.get(key)
        if current:  # user-provided value wins
            continue
        saved[key] = current
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _warn_on_unreasonable_config(config: ORCAConfig) -> None:
    physical_cores = (os.cpu_count() or 2) // 2  # this machine has SMT
    if config.nprocs > physical_cores:
        warnings.warn(
            f"nprocs={config.nprocs} exceeds the estimated number of physical "
            f"cores ({physical_cores}); running beyond physical cores typically "
            "slows ORCA down",
            UserWarning,
        )
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total_mb = int(line.split()[1]) / 1024
                    break
            else:
                return
    except OSError:
        return
    if config.maxcore_mb * config.nprocs > 0.75 * mem_total_mb:
        warnings.warn(
            f"maxcore_mb × nprocs ({config.maxcore_mb} × {config.nprocs} = "
            f"{config.maxcore_mb * config.nprocs} MB) exceeds 75% of physical "
            f"memory ({mem_total_mb:.0f} MB)",
            UserWarning,
        )


@dataclass(frozen=True)
class ORCAConfig:
    """User-facing backend configuration (spec §8.1)."""

    method: str
    basis: str | None = None
    charge: int = 0
    multiplicity: int = 1
    nprocs: int = 1
    maxcore_mb: int = 2000
    extra_keywords: tuple[str, ...] = ()
    extra_blocks: tuple[str, ...] = ()
    scratch_root: str | None = None
    restart: bool = True  # takes effect in Task 13
    keep_failed: bool = True  # takes effect in Task 13
    max_fresh_retries: int = 1  # takes effect in Task 13
    timeout_s: float | None = None
    orca_path: str | None = None  # None: OPI resolves via OPI_ORCA or PATH

    def __post_init__(self) -> None:
        if self.nprocs < 1:
            raise ValueError(f"nprocs must be >= 1, got {self.nprocs}")
        if self.maxcore_mb < 1:
            raise ValueError(f"maxcore_mb must be >= 1, got {self.maxcore_mb}")
        if self.multiplicity < 1:
            raise ValueError(f"multiplicity must be >= 1, got {self.multiplicity}")
        for keyword in self.extra_keywords:
            if str(keyword).strip().lower() in _FORBIDDEN_KEYWORDS:
                raise ValueError(
                    f"extra_keywords must not contain task-type keyword {keyword!r}: "
                    "the backend always requests a single-point EnGrad calculation"
                )
        for block in self.extra_blocks:
            if str(block).lstrip().lower().startswith(_FORBIDDEN_BLOCK_PREFIXES):
                raise ValueError(
                    f"extra_blocks must not contain the managed block {block!r}"
                )


@dataclass(frozen=True)
class _StepOutput:
    """Raw outcome of a single ORCA run (before unit conversion)."""

    energy_eh: float
    qm_gradient: np.ndarray  # (n_qm, 3) Eh/bohr
    pc_gradient: np.ndarray | None  # (n_mm, 3) Eh/bohr
    t_write: float
    t_orca: float
    t_read: float
    scf_cycles: int


class ORCAOPIBackend:
    """Turns QMRequests into QMResults by running ORCA (spec §7, §8)."""

    BASENAME = "qm"

    def __init__(self, config: ORCAConfig, *, check_version: bool = True) -> None:
        self.config = config
        self.scratch = ScratchDir.create(config.scratch_root)
        self._orca_path = Path(config.orca_path) if config.orca_path else None
        self._runner: Runner | None = None
        self._electron_check_done = False
        self.restart_state = RestartState(self.scratch.restart, enabled=config.restart)
        self.timings = TimingLog(self.scratch.root / "timings.csv")
        self.n_fresh_retries: int = 0  # cumulative "MORead failed, fresh SCF saved it"
        self._lock = threading.Lock()
        _warn_on_unreasonable_config(config)
        if check_version:
            self._get_runner().check_version()
        # Remove the scratch tree when this object is garbage collected or at
        # interpreter exit, even if close() was never called.
        self._finalizer = weakref.finalize(self, self.scratch.cleanup)

    # ------------------------------------------------------------------
    # Runner management (lazy: construction must work without ORCA)
    # ------------------------------------------------------------------

    def _get_runner(self) -> Runner:
        if self._runner is None:
            runner = Runner(working_dir=self.scratch.current)
            if self._orca_path is not None:
                runner.set_orca_path(self._orca_path)
            self._runner = runner
        return self._runner

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    def _build_calculator(self, request: QMRequest, use_guess: bool = False) -> Calculator:
        config = self.config
        positions_ang = nm_to_angstrom(request.qm_positions_nm)
        calc = Calculator(
            self.BASENAME,
            working_dir=self.scratch.current,
            version_check=False,
        )
        calc.json_via_input = False
        calc.structure = Structure(
            atoms=[
                Atom(element, coordinates=(float(x), float(y), float(z)))
                for element, (x, y, z) in zip(request.qm_elements, positions_ang)
            ],
            charge=config.charge,
            multiplicity=config.multiplicity,
        )
        keywords = [config.method]
        if config.basis:
            keywords.append(config.basis)
        keywords.extend(config.extra_keywords)
        keywords.append("EnGrad")
        if use_guess:
            # The guess must be staged *before* input.moinp is assigned: OPI
            # checks file existence at assignment time (spec §2.7).
            staged = self.restart_state.stage_guess(self.scratch.current)
            if staged is not None:
                keywords.append("MORead")
                calc.input.moinp = staged
        calc.input.add_simple_keywords(*keywords)
        calc.input.memory = config.maxcore_mb
        if config.nprocs > 1:
            calc.input.ncores = config.nprocs
        # Property JSON only: the gbw JSON costs ~1 MB per step for H2O and is
        # never used (spec §2.7).
        calc.input.add_arbitrary_string("%output jsonpropfile true end")
        if request.n_mm > 0:
            write_pointcharges(
                self.scratch.current / "pc.pc",
                request.mm_charges_e,
                nm_to_angstrom(request.mm_positions_nm),
            )
            calc.input.add_arbitrary_string('%pointcharges "pc.pc"')
        for block in config.extra_blocks:
            calc.input.add_arbitrary_string(block)
        return calc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_electron_count(self, qm_elements: tuple[str, ...]) -> None:
        """Reject charge/multiplicity combinations ORCA cannot realize (spec §8.1)."""
        if self._electron_check_done:
            return
        structure = Structure(
            atoms=[Atom(element, coordinates=(0.0, 0.0, 0.0)) for element in qm_elements],
            charge=self.config.charge,
            multiplicity=self.config.multiplicity,
        )
        n_electrons = structure.nelectrons
        if not structure.multiplicity_is_possible:
            raise ValueError(
                f"charge {self.config.charge} and multiplicity "
                f"{self.config.multiplicity} are incompatible: the QM region has "
                f"{n_electrons} electrons; multiplicity must be "
                f"{'odd' if n_electrons % 2 == 0 else 'even'} for that electron count"
            )
        self._electron_check_done = True

    def _output_tail(self, name: str, max_lines: int = 25) -> str:
        """Tail of an output file, for self-diagnosing error messages.

        Failure bundles live in scratch directories that may be node-local
        and transient (PBS job directories); embedding the tail in the
        exception makes remote failures diagnosable from a paste alone.
        """
        path = self.scratch.current / name
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            return f"{name}: <not readable>"
        if not lines:
            return f"{name}: <empty>"
        return "\n".join(f"  | {line}" for line in lines[-max_lines:])

    def _validate_output(self, out: Output, method_is_xtb: bool) -> None:
        if not out.terminated_normally():
            raise ORCACalculationError(
                f"ORCA did not terminate normally (basename {self.BASENAME} in "
                f"{self.scratch.current}); ORCA's process return code is not "
                "trustworthy. Last output lines:\n"
                f"{self._output_tail(self.BASENAME + '.out')}\n"
                f"{self._output_tail(self.BASENAME + '.err')}"
            )
        # XTB is a semi-empirical method without an SCF block, so the "SUCCESS"
        # marker SCF methods write is not expected there (verified empirically
        # when this backend was built; see spec §2.6).
        if not method_is_xtb and not out.scf_converged():
            raise ORCACalculationError(
                f"SCF did not converge (basename {self.BASENAME} in "
                f"{self.scratch.current}); see qm.out"
            )

    # ------------------------------------------------------------------
    # QMBackend protocol
    # ------------------------------------------------------------------

    def _run_step(self, request: QMRequest, use_guess: bool) -> "_StepOutput":
        """One full ORCA attempt: clean dir → input → run → validate → read.

        Raises on any failure; retries are the caller's (evaluate's) job.
        Designed as a separate method so unit tests can monkeypatch it.
        """
        config = self.config
        t_write0 = time.perf_counter()
        self.scratch.reset_current()
        calc = self._build_calculator(request, use_guess=use_guess)
        calc.write_input()
        t_write = time.perf_counter() - t_write0

        t_orca0 = time.perf_counter()
        timeout = -1 if config.timeout_s is None else int(math.ceil(config.timeout_s))
        try:
            with _orca_mpi_env(config.nprocs):
                self._get_runner().run_orca(calc.inpfile, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ORCATimeoutError(
                f"ORCA exceeded the time limit of {config.timeout_s} s "
                f"(step {request.step}); partial output in {self.scratch.current}"
            ) from exc
        t_orca = time.perf_counter() - t_orca0

        t_read0 = time.perf_counter()
        out = Output(self.BASENAME, working_dir=self.scratch.current, version_check=False)
        method_is_xtb = "xtb" in config.method.lower()
        self._validate_output(out, method_is_xtb)

        # reset_current() guarantees these files are from this step.
        energy_eh, qm_gradient = read_engrad(
            self.scratch.current / f"{self.BASENAME}.engrad", request.n_qm
        )
        pc_gradient = None
        if request.n_mm > 0:
            pc_gradient = read_pcgrad(
                self.scratch.current / f"{self.BASENAME}.pcgrad", request.n_mm
            )

        # Cross-check the two independent energy sources (spec §8.5 item 7).
        property_json = self.scratch.current / f"{self.BASENAME}.property.json"
        if not property_json.is_file():
            raise ORCAOutputError(f"{property_json}: file not found")
        out.parse(do_create_property_json=False, do_create_gbw_json=False, read_gbw_json=False)
        final_energy = out.get_final_energy()
        if final_energy is None:
            raise ORCAOutputError(
                f"{property_json}: no final energy found for cross-check"
            )
        if abs(final_energy - energy_eh) >= _ENERGY_CROSSCHECK_TOL_EH:
            raise ORCAOutputError(
                f"energy mismatch between .engrad ({energy_eh} Eh) and property "
                f"JSON ({final_energy} Eh)"
            )
        t_read = time.perf_counter() - t_read0

        out_text = (self.scratch.current / f"{self.BASENAME}.out").read_text()
        match = _SCF_CYCLES_RE.search(out_text)
        scf_cycles = int(match.group(1)) if match else -1

        return _StepOutput(
            energy_eh=energy_eh,
            qm_gradient=qm_gradient,
            pc_gradient=pc_gradient,
            t_write=t_write,
            t_orca=t_orca,
            t_read=t_read,
            scf_cycles=scf_cycles,
        )

    def _write_failure_bundle(self, request: QMRequest, exc: Exception) -> None:
        if not self.config.keep_failed:
            return
        bundle = write_failure_bundle(
            failures_dir=self.scratch.failures,
            step=request.step,
            current_dir=self.scratch.current,
            qm_elements=request.qm_elements,
            qm_positions_ang=nm_to_angstrom(request.qm_positions_nm),
            metadata={
                "step": request.step,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "config": dataclasses.asdict(self.config),
                "qm_elements": list(request.qm_elements),
                "n_point_charges": request.n_mm,
                "n_fresh_retries_total": self.n_fresh_retries,
            },
        )
        exc.args = (f"{exc.args[0] if exc.args else exc}. Failure bundle: {bundle}",)

    def evaluate(self, request: QMRequest) -> QMResult:
        self._check_electron_count(request.qm_elements)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "ORCAOPIBackend.evaluate is not re-entrant; one backend serves "
                "one OpenMM Context"
            )
        try:
            return self._evaluate_locked(request)
        finally:
            self._lock.release()

    def _evaluate_locked(self, request: QMRequest) -> QMResult:
        """SCF restart state machine (spec §9.2)."""
        t_total0 = time.perf_counter()
        t_write = t_orca = t_read = 0.0
        restart_used = False
        fresh_retries = 0
        step_output: _StepOutput | None = None
        final_error: Exception | None = None

        try:
            has_guess = self.config.restart and self.restart_state.has_guess
            if has_guess:
                try:
                    step_output = self._run_step(request, use_guess=True)
                    restart_used = True
                except ORCACalculationError as exc:
                    final_error = exc
                # ORCAOutputError / ORCATimeoutError propagate immediately.

            if step_output is None:
                if has_guess and final_error is not None:
                    for attempt in range(self.config.max_fresh_retries):
                        try:
                            step_output = self._run_step(request, use_guess=False)
                            final_error = None
                            fresh_retries = attempt + 1
                            self.n_fresh_retries += 1
                            warnings.warn(
                                f"SCF restart (MORead) failed at step {request.step}; "
                                "a fresh SCF converged — consider tightening the "
                                "SCF settings",
                                RuntimeWarning,
                            )
                            break
                        except ORCACalculationError as exc:
                            final_error = exc
                else:
                    # First step or restart disabled: exactly one fresh attempt.
                    try:
                        step_output = self._run_step(request, use_guess=False)
                    except Exception as exc:  # noqa: BLE001 - recorded below
                        final_error = exc
        except Exception as exc:
            # ORCAOutputError / ORCATimeoutError (not retried).
            final_error = exc
            step_output = None

        if step_output is None or final_error is not None:
            if final_error is None:  # pragma: no cover - defensive
                final_error = ORCACalculationError("ORCA step failed")
            self._write_failure_bundle(request, final_error)
            raise final_error

        assert step_output is not None
        # Promote this step's wavefunction to the restart guess (atomic).
        self.restart_state.commit(self.scratch.current / f"{self.BASENAME}.gbw")

        t_total = time.perf_counter() - t_total0
        t_write += step_output.t_write
        t_orca += step_output.t_orca
        t_read += step_output.t_read
        self.timings.append(
            {
                "step": request.step,
                "t_write": t_write,
                "t_orca": t_orca,
                "t_read": t_read,
                "t_total": t_total,
                "scf_cycles": step_output.scf_cycles,
                "restart_used": restart_used,
                "fresh_retries": self.n_fresh_retries,
            }
        )

        result = QMResult(
            energy_kj_mol=hartree_to_kj_mol(step_output.energy_eh),
            qm_forces_kj_mol_nm=gradient_to_forces(step_output.qm_gradient),
            mm_forces_kj_mol_nm=(
                gradient_to_forces(step_output.pc_gradient)
                if step_output.pc_gradient is not None
                else None
            ),
            timings_s={
                "write": t_write,
                "orca": t_orca,
                "read": t_read,
                "total": t_total,
            },
        )
        check_result(request, result)
        return result

    def close(self) -> None:
        """Remove the scratch tree (keep failures/ and timings.csv); idempotent."""
        if self._finalizer.alive:
            self._finalizer()
