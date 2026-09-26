"""Exception hierarchy for openmmorca (spec §12)."""

from __future__ import annotations


class OpenMMORCAError(RuntimeError):
    """Base class for all openmmorca errors."""


class ORCACalculationError(OpenMMORCAError):
    """ORCA did not terminate normally, or the SCF did not converge."""


class ORCAOutputError(OpenMMORCAError):
    """An ORCA output file is missing, malformed, inconsistent, or contains NaN/Inf."""


class ORCATimeoutError(OpenMMORCAError):
    """An ORCA run exceeded its time limit."""
