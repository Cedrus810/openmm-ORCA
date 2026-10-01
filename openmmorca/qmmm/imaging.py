"""Re-image the QM region across periodic boundaries (spec §11.2, Task 19).

OpenMM hands PythonForce callbacks unwrapped positions, so a QM region that
starts whole stays whole; this module guards against input structures that
were wrapped atom by atom (e.g. a frame taken from a wrapped trajectory).
Starting from each connected component's first atom, a breadth-first walk
over the QM bond graph moves every atom to the image nearest its already
placed neighbour. Declared boundary atoms M1 are leaves of that graph, so the
link-atom geometry always uses an M1 in the same image as its Q1.

Box vectors follow the OpenMM reduced form (a along x, b in the xy plane),
which makes the sequential c, b, a minimum-image reduction exact for
displacements shorter than half the box width.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import openmm.app


def qm_bond_graph(
    topology: openmm.app.Topology, qm_atoms, boundary_pairs=()
) -> dict[int, list[int]]:
    """Adjacency of the QM-internal bonds plus the declared boundary bonds.

    Each boundary atom M1 appears only as a leaf attached to its Q1.
    """
    qm = [int(i) for i in qm_atoms]
    qm_set = set(qm)
    graph: dict[int, list[int]] = {i: [] for i in qm}
    for atom1, atom2 in topology.bonds():
        i1, i2 = atom1.index, atom2.index
        if i1 in qm_set and i2 in qm_set:
            graph[i1].append(i2)
            graph[i2].append(i1)
    for pair in boundary_pairs:
        graph[pair.q1].append(pair.m1)
        graph.setdefault(pair.m1, []).append(pair.q1)
    for neighbours in graph.values():
        neighbours.sort()
    return graph


def minimum_image(displacement: np.ndarray, box_vectors_nm: np.ndarray) -> np.ndarray:
    """Shortest periodic image of *displacement* (one vector or rows of vectors)."""
    d = np.array(displacement, dtype=float)
    a, b, c = (np.asarray(v, dtype=float) for v in box_vectors_nm)
    d -= np.multiply.outer(np.round(d[..., 2] / c[2]), c)
    d -= np.multiply.outer(np.round(d[..., 1] / b[1]), b)
    d -= np.multiply.outer(np.round(d[..., 0] / a[0]), a)
    return d


def make_qm_whole(
    positions_nm: np.ndarray,
    qm_atoms: Sequence[int],
    graph: dict[int, list[int]],
    box_vectors_nm: np.ndarray,
) -> np.ndarray:
    """Copy of *positions_nm* with every graph atom moved into one consistent image.

    Components are anchored at their first atom in *qm_atoms* order; the first
    component's anchor never moves, later anchors go to the image nearest it.
    Particles outside the graph are returned unchanged.
    """
    positions = np.array(positions_nm, dtype=float)
    box = np.asarray(box_vectors_nm, dtype=float)
    placed: set[int] = set()
    first_anchor: int | None = None
    for anchor in (int(i) for i in qm_atoms):
        if anchor in placed:
            continue
        if first_anchor is None:
            first_anchor = anchor
        else:
            positions[anchor] = positions[first_anchor] + minimum_image(
                positions[anchor] - positions[first_anchor], box
            )
        placed.add(anchor)
        queue = deque([anchor])
        while queue:
            current = queue.popleft()
            for neighbour in graph.get(current, ()):
                if neighbour in placed:
                    continue
                positions[neighbour] = positions[current] + minimum_image(
                    positions[neighbour] - positions[current], box
                )
                placed.add(neighbour)
                queue.append(neighbour)
    return positions
