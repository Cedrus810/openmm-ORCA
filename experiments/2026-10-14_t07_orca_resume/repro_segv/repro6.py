"""New process; ctx loads checkpoint WITHOUT setPositions. Before that, optionally use another Context.
Usage: repro6.py none|other_setpos|other_setpos_force CHK"""
import sys
import numpy as np
import openmm as mm
from openmm import unit
mode, chk = sys.argv[1:3]
def compute(state):
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return 0.5 * float((pos ** 2).sum()), -pos
def make():
    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0)
    system.addForce(mm.PythonForce(compute))
    integ = mm.LangevinMiddleIntegrator(300, 1, 0.001)
    return system, integ, mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
if mode != "none":
    s0, i0, c0 = make()
    c0.setPositions(np.eye(4, 3) * 0.1)
    if mode == "other_setpos_force":
        c0.getState(getForces=True)
system, integ, ctx = make()
ctx.loadCheckpoint(open(chk, "rb").read())
ctx.getState(getForces=True)
print("OK", mode, flush=True)
