"""Minimal: does PythonForce + loadCheckpoint (+ step) crash on a platform? No openmmorca, no ORCA.

Usage: repro.py PLATFORM MODE   MODE = same (load into the context that wrote it) | fresh (new Context)
"""
import sys

import numpy as np
import openmm as mm
from openmm import unit

platform_name, mode = sys.argv[1], sys.argv[2]
PERIODIC = len(sys.argv) > 3 and sys.argv[3] == "periodic"
calls = []


def compute(state):
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    if PERIODIC:
        state.getPeriodicBoxVectors()
    calls.append(1)
    return 0.5 * float((pos ** 2).sum()), -pos  # harmonic well


def build():
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    force = mm.PythonForce(compute)
    force.setUsesPeriodicBoundaryConditions(PERIODIC)
    system.addForce(force)
    integ = mm.LangevinMiddleIntegrator(300, 1, 0.001)
    integ.setRandomNumberSeed(1)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform_name))
    ctx.setPositions(np.eye(4, 3) * 0.1)
    return system, integ, ctx


system, integ, ctx = build()
integ.step(5)
chk = ctx.createCheckpoint()
print("wrote checkpoint, calls", len(calls), flush=True)
if mode == "fresh":
    system2, integ2, ctx2 = build()
    ctx2.loadCheckpoint(chk)
    print("loaded into new context; stepping", flush=True)
    integ2.step(5)
else:
    ctx.loadCheckpoint(chk)
    print("loaded into same context; stepping", flush=True)
    integ.step(5)
print("OK, calls", len(calls), flush=True)
