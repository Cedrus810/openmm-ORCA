# Changelog

English | [简体中文](CHANGELOG.zh-CN.md)

Versions follow the milestones of `docs/plans/2026-09-26-openmm-orca-implementation-plan.md` (M2 = v0.1, M3 = v0.2, M4 = v0.3, M5 = v0.4).

## 0.4.0 — 2026-10-01 (M5: periodic MM + cutoff embedding)

### Added
- **Periodic QM/MM with cutoff embedding** (approximate, not PME-consistent): `createMixedSystem(..., embeddingCutoff=1.2 * nanometer)` for PME/LJPME/Ewald Systems. Each step the QM region and link-atom m1 are re-imaged (`openmmorca.qmmm.imaging`: `qm_bond_graph`, `make_qm_whole`, `minimum_image`) and whole MM residues within the cutoff are embedded at their nearest image (`openmmorca.qmmm.embedding`: `CutoffEmbedding`, `groups_from_topology`, `check_cutoff_against_box`, `perpendicular_widths`).
- `QMRequest.diagnostics`; timing-log columns `n_embed_groups`, `embed_changed`; DEBUG log line when groups cross the cutoff.
- `make_python_force(..., periodic=False)`; `QMMMCallback(..., embedding=None, qm_graph=None)`.

### Changed
- **Non-integer QM-region charges are now corrected** instead of only warned about: for every residue cut by a boundary, δ = x − round(x) (x: force-field charge of its QM part) is added to that residue's M2 atoms, so the embedding carries an integer net charge. `MixedSystemParts.qm_formal_charge`; warnings for ambiguous (near half-integer) residues and for a `charge` that differs from the force-field formal charge. The 0.3.0 "non-integer" warning is gone.
- Periodic Systems are no longer rejected with `NotImplementedError`; `CutoffPeriodic` raises `ValueError`.

### Added (application)
- Example `examples/enzyme_qmmm.py`: haloalkane dehalogenase DhlA with 1,2-dichloroethane (raw PDB `examples/data/2DHC.pdb`), PDBFixer + ff14SB + OpenFF Sage/NAGL for DCE + TIP3P, MM equilibration, periodic QM/MM NVT with link atoms and cutoff embedding; QM regions A and C; acceptance check at the end.

### Validation
- DhlA (PDB 2DHC, 31,610 atoms, PME), QM region A (DCE + Asp124 side chain, 15 atoms incl. 1 link H, charge −1), r2SCAN-3c on 40 MPI processes, Langevin NVT 300 K, 0.5 fs: 2000 steps (1 ps) without intervention, 5.53 s/step (ORCA 5.45 s; write 0.009 s, read 0.012 s), mean 152 embedded residue groups (~1,800 point charges), 286 steps with groups crossing the cutoff, 0 fresh-SCF retries; last 500 steps 298.6 ± 1.1 K; largest QM bond deviation 15.9 % (C1–Cl1 during a transient SN2 approach, O–C1 down to 0.232 nm at step ~1907, then recrossing).
- Periodic fake-backend finite differences < 1e-2 kJ/mol/nm; lattice translation and an atom wrapped across the box leave the ORCA energy unchanged (< 1e-7 Eh).

## 0.3.0 — 2026-09-29 (M4: link atoms)

### Added
- **Covalent QM/MM boundaries with H link atoms.** `ORCAPotential.createMixedSystem(..., boundaryPairs=[(q1, m1), ...], linkRatios=None)`: each declared bond gets a hydrogen cap at `R_q1 + g (R_m1 − R_q1)`; its force is redistributed onto q1/m1 by the chain rule (`F_q1 += (1 − g) F_L`, `F_m1 += g F_L`). Default `g` for C–C (1.09/1.526) and C–N (1.09/1.449); other element pairs need explicit `linkRatios`. `charge`/`multiplicity` describe the QM atoms plus link H.
- `openmmorca.qmmm.linkatoms`: `BoundaryPair`, `default_link_ratio`, `make_boundary_pairs`, `check_boundary_pairs` (q1 in QM, m1 in MM, bonded, one boundary per q1, no undeclared cut bonds), `LinkAtomManager`.
- `openmmorca.qmmm.charges.ChargeShift`: the m1 charge is removed from the embedding and spread evenly over m1's MM neighbours (M2). OpenMM's MM–MM electrostatics are unchanged. This is a *simplified* charge shift — no dipole-restoring point-charge pairs as in the literature scheme.
- `UserWarning` when the QM region's force-field charges sum to a non-integer value (residual > 0.05 e); no correction is applied.
- `check_whole_molecules(..., allowed_bonds=())`, `build_mixed_system(..., boundary_pairs=())`, `MixedSystemParts.boundary_pairs`, `QMMMCallback(..., link_manager=None)`.
- Example `examples/link_atom_dipeptide.py` (ACE-ALA-NME, QM = ALA methyl side chain, HF/def2-SVP, 200 steps NVT); test fixture `tests/data/ace_ala_nme.pdb`.

### Changed
- `MixedSystemParts.mm_charges_e` holds the charge-shifted embedding charges when boundaries are declared (identical to the force-field charges otherwise); an M2 atom that had zero charge is added to the embedding.
- The `check_whole_molecules` error message now points to `boundaryPairs`.

### Validation
- Fake backend: finite differences through one and two link atoms (C–C and C–N) agree with the analytic forces to < 1e-3 kJ/mol/nm.
- ORCA (HF/def2-SVP TightSCF): finite-difference error ≤ 0.042 kJ/mol/nm on Q1, M1 and M2 (forces 300–870 kJ/mol/nm); translating by 1 nm changes the energy by 5e-12 Eh; ‖ΣF‖/max‖F_i‖ = 1.4e-10. The dipeptide example keeps CA–CB within 0.149–0.158 nm over 200 steps at 720 ms/step.

### Known limitations
- Single-bond boundaries only, one link atom per q1; the non-integer QM charge residual is only warned about; the ONIOM model region must still be whole molecules; non-periodic systems only (M5).

## 0.2.2 — 2026-09-29

### Fixed
- **ONIOM: the low-level model-region evaluation used the full system's charge/multiplicity.** `E_low(model)` now runs the low configuration with the model-region (`high`) charge/multiplicity. Previously a charged or open-shell low-layer-only region silently gave a wrong `E_low(model)` or was rejected by the electron-count check; the "low-layer-only atoms must carry zero net charge" restriction is gone. **Behaviour change.**

### Added
- `UserWarning` when an ONIOM topology has no bonds (the whole-molecule check cannot see a cut molecule).
- `ORCAPotential._create_backend(config=None)`.

### Changed
- `ONIOMPotential.summarize_timings()` no longer lists every backend twice when `high` and `low` are the same object.

### Validation
- ONIOM NVE (5-water cluster, HF/STO-3G : xTB, 0.25 fs, 1 ps): drift +0.0067 kJ/mol/ps, std dev 0.0231 kJ/mol, 732 ms/step — within the QM/MM v0.1 thresholds.
- New ORCA identity test with a charged low-only region (neutral model water + Na⁺); the heterogeneous-level test now checks every backend's MO restart chain.

## 0.2.1 — 2026-09-29

### Added
- **Two-layer subtractive ONIOM (QM:QM)**: `ONIOMPotential(high=..., low=...).createONIOMSystem(topology, atoms)`, `E = E_high(model) + E_low(full) − E_low(model)`, mechanical coupling, three independent backends per System. Example `examples/oniom_water_cluster.py`.
- Shared helpers `validate_atom_indices`, `topology_masses`, `make_python_force(name=...)`.

## 0.2.0 — 2026-09-27 (M3: restart and robustness)

### Added
- SCF restart state machine (MORead from the last good `.gbw`, one fresh-SCF retry, never stale forces), failure bundles, per-step timing log and `summarize_timings()`, MPI tuning for `nprocs > 1`.

### Validation
- Restart vs. no-restart trajectories agree to < 1e-6 Eh over the first 50 steps; 1000 consecutive steps without intervention.

## 0.1 — 2026-09-27 (M2: non-periodic electronic embedding)

### Added
- `ORCAPotential.createSystem` (full QM) and `createMixedSystem` (QM/MM with electrostatic embedding through `%pointcharges` / `.pcgrad`), TIP4P-type virtual-site support.

### Validation
- QM water + MM water cluster NVE (HF/STO-3G, 0.25 fs, 1 ps): drift +0.0056 kJ/mol/ps, std dev 0.0163 kJ/mol.
