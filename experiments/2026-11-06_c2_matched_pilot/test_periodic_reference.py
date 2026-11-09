"""Physics controls for the experiment's energy-only reference (no ORCA)."""
import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.qmmm.pbc import itrf

from periodic_reference import remove_qm_images


def molecule():
    return gto.M(atom="O 0 0 0; H .7586 0 .5043; H -.7586 0 .5043",
                 basis="sto-3g", unit="Angstrom", verbose=0)


def periodic(mol, charges):
    coords = [[3, 1, 1], [3, -1, -1]]
    if np.all(np.asarray(charges) == 0):
        # Coincident opposite charges give exactly zero external potential,
        # while avoiding PySCF's undefined Ewald mesh for sum(q**2) == 0.
        coords = [[3, 1, 1], [3, 1, 1]]
        charges = [.2, -.2]
    mf = itrf.add_mm_charges(scf.RHF(mol), coords,
                            np.eye(3)*20, np.array(charges), rcut_hcore=6,
                            rcut_ewald=10, unit="Angstrom")
    mf.conv_tol = 1e-12
    return mf


def test_zero_external_charge_recovers_isolated_energy_and_density():
    mol = molecule()
    isolated = scf.RHF(mol)
    isolated.conv_tol = 1e-12
    e0 = isolated.kernel()
    native = periodic(mol, [0., 0.])
    enative = native.kernel()
    aligned = remove_qm_images(periodic(mol, [0., 0.]))
    ealigned = aligned.kernel()
    assert isolated.converged and native.converged and aligned.converged
    assert abs(ealigned-e0) < 1e-10
    assert np.max(np.abs(aligned.make_rdm1()-isolated.make_rdm1())) < 1e-8
    # Negative control: the original reference polarizes against QM images.
    assert abs(enative-e0) > 1e-7


def test_external_periodic_field_retained_and_energy_variation_matches_fock():
    mol = molecule()
    mf = remove_qm_images(periodic(mol, [-.2, .2]))
    energy = mf.kernel()
    assert mf.converged
    isolated = scf.RHF(mol).kernel()
    assert abs(energy-isolated) > 1e-5
    dm = mf.make_rdm1()
    direction = np.zeros_like(dm)
    direction[0, 1] = direction[1, 0] = .1
    h = 1e-4
    numerical = (mf.energy_tot(dm+h*direction)-mf.energy_tot(dm-h*direction))/(2*h)
    analytic = np.einsum("ij,ji", mf.get_hcore()+mf.get_veff(dm=dm), direction)
    assert abs(numerical-analytic) < 1e-7


def test_unvalidated_gradient_and_wrong_reference_rejected():
    with pytest.raises(TypeError):
        remove_qm_images(scf.RHF(molecule()))
    aligned = remove_qm_images(periodic(molecule(), [0., 0.]))
    with pytest.raises(NotImplementedError, match="energy-only"):
        aligned.nuc_grad_method()
