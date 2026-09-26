"""OpenMM-side QM/MM system preparation (no ORCA knowledge, spec §5.2)."""

from openmmorca.qmmm.system import MixedSystemParts, build_mixed_system

__all__ = ["MixedSystemParts", "build_mixed_system"]
