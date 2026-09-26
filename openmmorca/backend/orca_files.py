"""Readers/writers for the ORCA files this project depends on.

Only three formats (spec §2.3–§2.5, §8.4):

* point charges:  first line N, then N lines ``q x y z`` (e, Å) — this is the
  ``%pointcharges`` file, the only supported embedding route;
* ``.engrad``:    N, total energy (Eh), 3N gradient rows (Eh/bohr), then N
  ``Z x y z`` rows (bohr);
* ``.pcgrad``:    N, then N gradient rows (Eh/bohr), one per point charge.

ORCA/OPI provides no readers for the last two; they live here so that no
other module ever touches ORCA file formats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openmmorca.errors import ORCAOutputError


def write_pointcharges(path: Path, charges_e, positions_ang) -> None:
    """Write a ``%pointcharges`` file (charges in e, positions in Å)."""
    charges = np.asarray(charges_e, dtype=float).ravel()
    positions = np.asarray(positions_ang, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_ang must have shape (N, 3), got {positions.shape}")
    if charges.shape[0] != positions.shape[0]:
        raise ValueError(
            f"got {charges.shape[0]} charges but {positions.shape[0]} positions"
        )
    with open(path, "w") as handle:
        handle.write(f"{charges.shape[0]}\n")
        for charge, (x, y, z) in zip(charges, positions):
            handle.write(f"{charge:.10f} {x:.10f} {y:.10f} {z:.10f}\n")


def _parse_float(token: str, path: Path, where: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ORCAOutputError(f"{path}: cannot parse {where} as a number: {token!r}") from exc
    if not np.isfinite(value):
        raise ORCAOutputError(f"{path}: non-finite value in {where}: {token!r}")
    return value


def read_engrad(path: Path, n_atoms: int) -> tuple[float, np.ndarray]:
    """Read an ``.engrad`` file; return (energy_Eh, gradient (n_atoms, 3) Eh/bohr)."""
    path = Path(path)
    if not path.is_file():
        raise ORCAOutputError(f"{path}: file not found")
    data = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    try:
        declared_atoms = int(_parse_float(data[0], path, "atom-count line"))
        energy = _parse_float(data[1], path, "energy line")
    except IndexError as exc:
        raise ORCAOutputError(f"{path}: file too short to be an .engrad file") from exc
    if declared_atoms != n_atoms:
        raise ORCAOutputError(
            f"{path}: .engrad reports {declared_atoms} atoms, expected {n_atoms}"
        )
    expected_rows = 3 * n_atoms + n_atoms
    if len(data) < 2 + expected_rows:
        raise ORCAOutputError(
            f"{path}: expected {expected_rows} value rows for {n_atoms} atoms, "
            f"found {len(data) - 2}"
        )
    # The gradient block is 3N rows of one value each (x1 y1 z1 x2 ... order).
    rows = [data[2 + i].split() for i in range(3 * n_atoms)]
    if any(len(row) != 1 for row in rows):
        raise ORCAOutputError(f"{path}: gradient block must have one value per row")
    gradient = np.array(
        [_parse_float(row[0], path, f"gradient row {i + 1}") for i, row in enumerate(rows)],
        dtype=float,
    ).reshape(n_atoms, 3)
    return energy, gradient


def read_pcgrad(path: Path, n_charges: int) -> np.ndarray:
    """Read a ``.pcgrad`` file; return the (n_charges, 3) gradient in Eh/bohr."""
    path = Path(path)
    if not path.is_file():
        raise ORCAOutputError(f"{path}: file not found")
    data = [line for line in path.read_text().splitlines() if line.strip()]
    try:
        declared_charges = int(_parse_float(data[0], path, "charge-count line"))
    except IndexError as exc:
        raise ORCAOutputError(f"{path}: file too short to be a .pcgrad file") from exc
    if declared_charges != n_charges:
        raise ORCAOutputError(
            f"{path}: .pcgrad reports {declared_charges} charges, expected {n_charges}"
        )
    if len(data) < 1 + n_charges:
        raise ORCAOutputError(
            f"{path}: expected {n_charges} gradient rows, found {len(data) - 1}"
        )
    gradient = np.array(
        [
            [_parse_float(token, path, f"gradient row {i + 1}") for token in data[1 + i].split()]
            for i in range(n_charges)
        ]
    )
    if gradient.shape != (n_charges, 3):
        raise ORCAOutputError(
            f"{path}: pcgrad rows do not have 3 components ({gradient.shape})"
        )
    return gradient
