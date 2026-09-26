"""Benchmark ORCA wall time vs nprocs for one geometry (plan Task 14).

    export OPI_ORCA=/home/ruigengji/ORCA611
    # 并行需要 MPI（登录节点没有系统 mpirun 时）：
    export OPI_MPI=/home/apps/openmpi/5.0.7_gcc13.3.0
    python examples/bench_nprocs.py examples/data/water12.xyz "HF def2-SVP" --nprocs 1,2,4,8,16,32,40

For each nprocs: one warm-up call, then 3 timed calls; reports the median.
The method string is passed verbatim as ORCA simple keywords (plus EnGrad).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from openmmorca.backend.base import QMRequest
from openmmorca.backend.orca_opi import ORCAConfig, ORCAOPIBackend
from openmmorca.units import nm_to_angstrom


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    """Minimal xyz reader: returns (elements, positions_ang)."""
    lines = Path(path).read_text().splitlines()
    n_atoms = int(lines[0].split()[0])
    elements, positions = [], []
    for line in lines[2 : 2 + n_atoms]:
        tokens = line.split()
        elements.append(tokens[0])
        positions.append([float(t) for t in tokens[1:4]])
    return tuple(elements), np.asarray(positions)


def make_request(path: Path) -> QMRequest:
    elements, positions_ang = read_xyz(path)
    return QMRequest(
        qm_elements=elements,
        qm_positions_nm=np.asarray(positions_ang) / 10.0,
    )


def time_nprocs(path: Path, method: str, nprocs: int, extra_keywords: tuple[str, ...]) -> float:
    request = make_request(path)
    config = ORCAConfig(
        method=method,
        extra_keywords=extra_keywords,
        nprocs=nprocs,
        restart=False,
    )
    backend = ORCAOPIBackend(config)
    try:
        backend.evaluate(request)  # warm-up: discard first (cold cache) call
        timings = []
        for _ in range(3):
            result = backend.evaluate(request)
            timings.append(result.timings_s["total"])
        return float(np.median(timings))
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path, help="geometry file (xyz, Å)")
    parser.add_argument("method", help='ORCA method keywords, e.g. "HF def2-SVP"')
    parser.add_argument("--nprocs", default="1,2,4,8,16,32,40")
    parser.add_argument(
        "--extra", nargs="*", default=["TightSCF"], help="extra simple keywords"
    )
    args = parser.parse_args()

    tokens = args.method.split()
    method = tokens[0]
    basis = tokens[1] if len(tokens) > 1 else None
    extra_keywords = tuple(args.extra)

    n_atoms = int(open(args.xyz).readline())
    print(f"geometry : {args.xyz} ({n_atoms} atoms)")
    print(f"method   : {args.method} {' '.join(args.extra)}")
    print(f"{'nprocs':>7} | {'median t (s)':>12}")
    print("-" * 24)
    for nprocs in [int(v) for v in args.nprocs.split(",")]:
        median = time_nprocs(args.xyz, method, nprocs, extra_keywords)
        print(f"{nprocs:>7} | {median:>12.2f}")


if __name__ == "__main__":
    main()
