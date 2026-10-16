"""Tiny (4 particles, nonperiodic) PythonForce; checkpoint across processes. Usage: repro3.py PLATFORM write|load CHK"""
import sys
import numpy as np
import openmm as mm
from openmm import unit
platform_name, phase, chk = sys.argv[1:4]
def compute(state):
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return 0.5 * float((pos ** 2).sum()), -pos
system = mm.System()
for _ in range(4):
    system.addParticle(1.0)
system.addForce(mm.PythonForce(compute))
integ = mm.LangevinMiddleIntegrator(300, 1, 0.001)
ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform_name))
if phase == "write":
    ctx.setPositions(np.eye(4, 3) * 0.1); integ.step(5); open(chk, "wb").write(ctx.createCheckpoint()); print("written")
else:
    ctx.loadCheckpoint(open(chk, "rb").read()); integ.step(5); print("OK", flush=True)
