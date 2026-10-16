"""Tiny PythonForce, cross-process: which restore paths survive? Usage: repro4.py write|state|chk_reinit CHK STATEXML"""
import sys
import numpy as np
import openmm as mm
from openmm import unit
phase, chk, xml = sys.argv[1:4]
def compute(state):
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return 0.5 * float((pos ** 2).sum()), -pos
system = mm.System()
for _ in range(4):
    system.addParticle(1.0)
system.addForce(mm.PythonForce(compute))
integ = mm.LangevinMiddleIntegrator(300, 1, 0.001)
ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
if phase == "write":
    ctx.setPositions(np.eye(4, 3) * 0.1); integ.step(5)
    open(chk, "wb").write(ctx.createCheckpoint())
    open(xml, "w").write(mm.XmlSerializer.serialize(ctx.getState(getPositions=True, getVelocities=True)))
    print("written")
elif phase == "state":
    ctx.setState(mm.XmlSerializer.deserialize(open(xml).read())); integ.step(5); print("OK state", flush=True)
else:
    ctx.loadCheckpoint(open(chk, "rb").read()); ctx.reinitialize(preserveState=True); integ.step(5); print("OK chk+reinit", flush=True)
