"""Tests for re-imaging the QM region across periodic boundaries (Task 19, spec §11.2)."""

from __future__ import annotations

import numpy as np
import pytest

import helpers
from openmmorca.qmmm.imaging import make_qm_whole, qm_bond_graph
from openmmorca.qmmm.linkatoms import LinkAtomManager, make_boundary_pairs
from test_linkatoms import dipeptide, side_chain

OH = helpers.OH_NM


def orthorhombic(length: float) -> np.ndarray:
    return np.diag([length, length, length])


def truncated_octahedron(d: float) -> np.ndarray:
    """OpenMM reduced-form box vectors of a truncated octahedron."""
    return np.array(
        [
            [d, 0.0, 0.0],
            [d / 3.0, 2.0 * np.sqrt(2.0) * d / 3.0, 0.0],
            [-d / 3.0, np.sqrt(2.0) * d / 3.0, np.sqrt(6.0) * d / 3.0],
        ]
    )


def bond_lengths(positions, bonds):
    return [float(np.linalg.norm(positions[i] - positions[j])) for i, j in bonds]


def test_whole_molecule_unchanged():
    topology = helpers.water_topology(2)
    positions = helpers.water_positions(2) + 1.0
    graph = qm_bond_graph(topology, [0, 1, 2])
    assert graph == {0: [1, 2], 1: [0], 2: [0]}
    whole = make_qm_whole(positions, [0, 1, 2], graph, orthorhombic(3.0))
    np.testing.assert_allclose(whole, positions, atol=1e-15)


def test_split_across_x_boundary():
    topology = helpers.water_topology(1)
    box = orthorhombic(3.0)
    positions = helpers.water_positions(1) + np.array([2.95, 1.0, 1.0])
    positions[1] -= box[0]  # H1 wrapped to the other side of the box
    assert bond_lengths(positions, [(0, 1)])[0] > 2.0
    graph = qm_bond_graph(topology, [0, 1, 2])
    whole = make_qm_whole(positions, [0, 1, 2], graph, box)
    assert bond_lengths(whole, [(0, 1), (0, 2)]) == pytest.approx([OH, OH], abs=1e-12)
    np.testing.assert_allclose(whole[0], positions[0])  # the anchor does not move


def test_triclinic_box():
    topology = helpers.water_topology(1)
    box = truncated_octahedron(3.0)
    positions = helpers.water_positions(1) + np.array([0.5, 0.5, 0.5])
    positions[1] += box[2] - box[1]  # a lattice translation mixing two vectors
    positions[2] -= box[0] + box[2]
    graph = qm_bond_graph(topology, [0, 1, 2])
    whole = make_qm_whole(positions, [0, 1, 2], graph, box)
    assert bond_lengths(whole, [(0, 1), (0, 2)]) == pytest.approx([OH, OH], abs=1e-12)


def test_disconnected_qm_region():
    """Each connected component anchors at its first atom; components are
    placed at the image nearest the first component's anchor."""
    topology = helpers.water_topology(2)
    box = orthorhombic(3.0)
    positions = helpers.water_positions(2) + 0.2  # waters 0.3 nm apart along x
    reference = positions.copy()
    positions[3:6] += box[0] + box[1]  # water 1 shifted by a lattice vector
    positions[4] -= box[2]  # and one of its H wrapped independently
    graph = qm_bond_graph(topology, range(6))
    whole = make_qm_whole(positions, list(range(6)), graph, box)
    np.testing.assert_allclose(whole, reference, atol=1e-12)


def test_boundary_m1_imaged_with_q1():
    topology, _, positions = dipeptide()
    qm, pairs = side_chain(topology)
    boundary = make_boundary_pairs(topology, pairs)
    q1, m1 = pairs[0]
    box = orthorhombic(2.0)
    links = LinkAtomManager(boundary)
    reference_link = links.link_positions(positions)

    shifted = positions.copy()
    shifted[m1] += box[1]  # M1 (CA) one box length away from its Q1
    graph = qm_bond_graph(topology, qm, boundary)
    assert m1 in graph[q1] and graph[m1] == [q1]  # M1 is a leaf on Q1
    whole = make_qm_whole(shifted, qm, graph, box)
    assert np.linalg.norm(whole[m1] - whole[q1]) == pytest.approx(
        np.linalg.norm(positions[m1] - positions[q1]), abs=1e-12
    )
    np.testing.assert_allclose(links.link_positions(whole), reference_link, atol=1e-12)
    # Particles outside the QM region and the boundary are returned untouched.
    others = [i for i in range(len(positions)) if i not in qm and i != m1]
    np.testing.assert_array_equal(whole[others], shifted[others])
