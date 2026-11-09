"""Energy-only periodic MM embedding without QM self-image polarization.

PySCF QMMMSCF uses get_qm_ewald_pot in both get_veff and energy_ewald.
Replacing that potential removes QM self-image terms at the SCF-functional
level. The stock periodic analytic gradient has additional QM-image terms
and is deliberately unavailable for this experiment-specific reference.
"""
import types

import numpy as np


def zero_qm_image_potential(self, mol, dm, qm_ewald_hess=None):
    return (np.zeros(mol.natm), np.zeros((mol.natm, 3)), np.zeros((mol.natm, 3, 3)))


def no_periodic_gradient(self, *args, **kwargs):
    raise NotImplementedError("energy-only MM-periodic reference: analytic gradients are not validated")


def remove_qm_images(mf):
    from pyscf.qmmm.pbc.itrf import QMMMSCF
    if not isinstance(mf, QMMMSCF):
        raise TypeError("expected a PySCF periodic QMMM SCF object")
    mf.get_qm_ewald_pot = types.MethodType(zero_qm_image_potential, mf)
    mf.nuc_grad_method = types.MethodType(no_periodic_gradient, mf)
    return mf
