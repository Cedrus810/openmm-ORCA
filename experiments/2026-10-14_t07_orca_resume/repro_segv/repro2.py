"""Water box (PME) + periodic PythonForce on a few atoms; checkpoint written in one process, loaded in another.

Usage: repro2.py PLATFORM write|load CHK_PATH [NWATER_BOX_NM]
"""
import sys

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

platform_name, phase, chk = sys.argv[1], sys.argv[2], sys.argv[3]
size = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
QM = [0, 1, 2]


def compute(state):
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    state.getPeriodicBoxVectors()
    f = np.zeros_like(pos)
    f[QM] = -0.1 * (pos[QM] - pos[QM].mean(0))
    return 0.0, f


ff = app.ForceField("amber14/tip3p.xml")
modeller = app.Modeller(app.Topology(), [])
modeller.addSolvent(ff, boxSize=mm.Vec3(size, size, size) * unit.nanometer)
system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, constraints=app.HBonds)
force = mm.PythonForce(compute)
force.setUsesPeriodicBoundaryConditions(True)
system.addForce(force)
integ = mm.LangevinMiddleIntegrator(300, 1, 0.0005)
integ.setRandomNumberSeed(7)
ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform_name))
print(platform_name, phase, system.getNumParticles(), "particles", flush=True)
if phase == "write":
    ctx.setPositions(modeller.positions)
    mm.LocalEnergyMinimizer.minimize(ctx, 10, 200)
    ctx.setVelocitiesToTemperature(300, 3)
    integ.step(20)
    open(chk, "wb").write(ctx.createCheckpoint())
    print("checkpoint written", flush=True)
else:
    ctx.loadCheckpoint(open(chk, "rb").read())
    print("checkpoint loaded; stepping", flush=True)
    integ.step(20)
    print("OK", flush=True)
