#!/usr/bin/env python3
"""Generate and validate the complete molecular H8 HF reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import scipy

CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from h8hf.circuits import (  # noqa: E402
    abstract_gate_lines,
    fit_projector_to_diamond,
    measurement_rotation,
    native_gate_lines,
    swap_network_matchings,
    verify_native_givens,
)
from h8hf.data_processing import (  # noqa: E402
    analyze_gamma,
    reconstruct_gamma,
)
from h8hf.gaussian import (  # noqa: E402
    parse_scf_energy_file,
    render_gaussian_input,
)
from h8hf.integrals import (  # noqa: E402
    ANGSTROM_TO_BOHR,
    H_STO3G_COEFFICIENTS,
    H_STO3G_EXPONENTS,
    build_sto3g_integrals,
    linear_h_chain,
)
from h8hf.rhf import (  # noqa: E402
    generalized_eigh,
    rhf_energy_from_gamma,
    run_rhf,
    transform_integrals,
)
from h8hf.sampling import (  # noqa: E402
    bootstrap_energies,
    multinomial_counts,
    plan_shots,
    slater_probabilities,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows supplied for {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",", fmt="%.16e")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_provenance() -> list[dict[str, str]]:
    paper = "https://arxiv.org/abs/2004.04174"
    google = "https://quantumai.google/cirq/experiments/hfvqe"
    return [
        {
            "parameter": "molecule",
            "value": "linear H8",
            "classification": "model choice",
            "derivation_or_source": "Eight hydrogen nuclei on the z axis; H8 is one of the benchmark chains in Arute et al.",
            "source": paper,
            "code_location": "h8hf/integrals.py:linear_h_chain",
        },
        {
            "parameter": "nearest-neighbor spacing",
            "value": "1.3000 angstrom",
            "classification": "geometry",
            "derivation_or_source": "Fixed benchmark geometry used in the prior project and explicitly illustrated for the hydrogen-chain experiment.",
            "source": paper,
            "code_location": "run_all.py:build_reproduction",
        },
        {
            "parameter": "charge / multiplicity",
            "value": "0 / 1",
            "classification": "electronic state",
            "derivation_or_source": "Neutral H8 has 8 electrons; the closed-shell benchmark is a singlet.",
            "source": paper,
            "code_location": "gaussian input and run_all.py",
        },
        {
            "parameter": "electronic-structure method",
            "value": "RHF",
            "classification": "model choice",
            "derivation_or_source": "Alpha and beta spatial orbitals are constrained to be identical, so only one spin sector is encoded.",
            "source": paper,
            "code_location": "h8hf/rhf.py",
        },
        {
            "parameter": "atomic basis",
            "value": "STO-3G",
            "classification": "model choice",
            "derivation_or_source": "The paper states that the original one-body integrals are represented in the STO-3G atomic basis.",
            "source": paper,
            "code_location": "h8hf/integrals.py",
        },
        {
            "parameter": "STO-3G primitive exponents",
            "value": "; ".join(f"{x:.8f}" for x in H_STO3G_EXPONENTS),
            "classification": "basis data",
            "derivation_or_source": "Standard H 1s STO-3G contraction.",
            "source": "Hehre, Stewart, Pople, J. Chem. Phys. 51, 2657 (1969)",
            "code_location": "h8hf/integrals.py:H_STO3G_EXPONENTS",
        },
        {
            "parameter": "STO-3G contraction coefficients",
            "value": "; ".join(f"{x:.8f}" for x in H_STO3G_COEFFICIENTS),
            "classification": "basis data",
            "derivation_or_source": "Standard H 1s STO-3G contraction; primitive normalization is applied explicitly.",
            "source": "Hehre, Stewart, Pople, J. Chem. Phys. 51, 2657 (1969)",
            "code_location": "h8hf/integrals.py:H_STO3G_COEFFICIENTS",
        },
        {
            "parameter": "angstrom-to-bohr",
            "value": f"{ANGSTROM_TO_BOHR:.16f}",
            "classification": "physical constant",
            "derivation_or_source": "CODATA-compatible atomic-unit conversion used for all analytic Gaussian integrals.",
            "source": "NIST Atomic Units",
            "code_location": "h8hf/integrals.py:ANGSTROM_TO_BOHR",
        },
        {
            "parameter": "number of spatial modes / qubits",
            "value": "8",
            "classification": "encoding",
            "derivation_or_source": "One STO-3G 1s function per H atom and one explicitly simulated spin sector.",
            "source": paper,
            "code_location": "run_all.py",
        },
        {
            "parameter": "particles in simulated spin sector",
            "value": "4",
            "classification": "encoding",
            "derivation_or_source": "RHF singlet has 4 alpha and 4 beta electrons; alpha and beta 1-RDMs are identical.",
            "source": paper,
            "code_location": "run_all.py",
        },
        {
            "parameter": "initial orbital basis",
            "value": "core orbitals",
            "classification": "basis transformation",
            "derivation_or_source": "Generalized eigenvectors of h_core C = S C epsilon, matching the paper's non-interacting initial orbitals.",
            "source": paper,
            "code_location": "h8hf/rhf.py:generalized_eigh",
        },
        {
            "parameter": "SCF energy / density tolerances",
            "value": "1e-12 Ha / 1e-10 RMS",
            "classification": "numerical control",
            "derivation_or_source": "Chosen tighter than the mHartree accuracy target so reference convergence is negligible.",
            "source": "analysis choice",
            "code_location": "h8hf/rhf.py:run_rhf",
        },
        {
            "parameter": "SCF maximum iterations / DIIS size",
            "value": "256 / 8",
            "classification": "numerical control",
            "derivation_or_source": "Convergence safeguards; neither changes the converged stationary solution.",
            "source": "analysis choice",
            "code_location": "h8hf/rhf.py:run_rhf",
        },
        {
            "parameter": "Givens layout",
            "value": "7-layer diamond, 16 rotations",
            "classification": "circuit ansatz",
            "derivation_or_source": "At half filling the Grassmann manifold has Nocc*Nvirt=4*4=16 real dimensions; the paper uses the nearest-neighbor diamond.",
            "source": paper,
            "code_location": "h8hf/circuits.py:diamond_layout",
        },
        {
            "parameter": "Givens parameter fit",
            "value": "nonlinear least squares on ||gamma(theta)-gamma_target||_F",
            "classification": "parameter derivation",
            "derivation_or_source": "Every angle is fitted from the molecular RHF projector; no angle is hand-entered.",
            "source": google,
            "code_location": "h8hf/circuits.py:fit_projector_to_diamond",
        },
        {
            "parameter": "fit seed / starts / tolerance",
            "value": "20260729+setting / 8 / max error <1e-11",
            "classification": "numerical control",
            "derivation_or_source": "Deterministic multi-start removes optimizer ambiguity; closure is checked against the target 1-RDM.",
            "source": "analysis choice",
            "code_location": "run_all.py and h8hf/circuits.py",
        },
        {
            "parameter": "native gates per Givens",
            "value": "2 SQISWAP + 3 RZ",
            "classification": "hardware compilation",
            "derivation_or_source": "Gate identity is evaluated explicitly under the PyQPanda RZ and SqiSWAP matrices, including its -i sign.",
            "source": "https://pyqpanda-tutorial-en.readthedocs.io/en/latest/chapter2/index.html",
            "code_location": "h8hf/circuits.py:native_gate_lines",
        },
        {
            "parameter": "RZ angle unit",
            "value": "radian",
            "classification": "file convention",
            "derivation_or_source": "All pi multiples are numerically expanded; this prevents the earlier normalized-pi ambiguity.",
            "source": "project convention",
            "code_location": "circuits/native/*.qcir",
        },
        {
            "parameter": "measurement settings",
            "value": "9 = 1 diagonal + 8 swap-network layers",
            "classification": "measurement design",
            "derivation_or_source": "The N+1 particle-number-preserving 1-RDM protocol in Appendix C; 28 unordered off-diagonal pairs appear exactly once.",
            "source": paper,
            "code_location": "h8hf/circuits.py:swap_network_matchings",
        },
        {
            "parameter": "off-diagonal analyzer",
            "value": "G(pi/4)",
            "classification": "measurement design",
            "derivation_or_source": "After the analyzer, gamma_pq=(n_q-n_p)/2 for the ordered pair (p,q).",
            "source": paper,
            "code_location": "h8hf/circuits.py:measurement_rotation",
        },
        {
            "parameter": "reported bit order",
            "value": "q[7]...q[0]",
            "classification": "data convention",
            "derivation_or_source": "Matches the earlier project convention; the parser also exposes the alternative explicitly.",
            "source": "project convention",
            "code_location": "h8hf/data_processing.py:bitstring_to_occupations",
        },
        {
            "parameter": "finite-shot demonstration",
            "value": "1000 shots per setting, seed 20260729",
            "classification": "sampling choice",
            "derivation_or_source": "Matches the user's simulator repetition count; it is a demonstration, not a Gaussian parameter.",
            "source": "user/project choice",
            "code_location": "run_all.py",
        },
        {
            "parameter": "post-selection",
            "value": "Hamming weight = 4",
            "classification": "error mitigation",
            "derivation_or_source": "All basis rotations conserve the particle number in the explicitly encoded spin sector.",
            "source": paper,
            "code_location": "h8hf/data_processing.py:postselect_distribution",
        },
        {
            "parameter": "pure-state projection",
            "value": "four largest natural occupations -> 1; remaining -> 0",
            "classification": "error mitigation",
            "derivation_or_source": "Closest rank-4 orthogonal projector in Frobenius norm; equivalent eigenvector limit of McWeeny purification when it converges.",
            "source": paper,
            "code_location": "h8hf/data_processing.py:rank_n_projector",
        },
        {
            "parameter": "bootstrap samples / seed",
            "value": "2000 / 20260730",
            "classification": "uncertainty analysis",
            "derivation_or_source": "Nonparametric multinomial resampling of each observed setting.",
            "source": "analysis choice",
            "code_location": "h8hf/sampling.py:bootstrap_energies",
        },
        {
            "parameter": "Gaussian route",
            "value": "RHF/STO-3G SCF=(Tight,XQC,MaxCycle=512) NoSymm Pop=Full",
            "classification": "classical cross-check",
            "derivation_or_source": "Same molecular model; Tight sets a stricter SCF threshold, XQC is only a convergence fallback.",
            "source": "https://gaussian.com/scf/",
            "code_location": "h8hf/gaussian.py:render_gaussian_input",
        },
    ]


def formula_provenance() -> list[dict[str, str]]:
    paper = "https://arxiv.org/abs/2004.04174"
    return [
        {
            "operation": "Gaussian product center",
            "formula": "P=(alpha*A+beta*B)/(alpha+beta)",
            "why_valid": "Gaussian product theorem for two s primitives.",
            "source": "standard Gaussian integral theory",
            "code_location": "h8hf/integrals.py:_gaussian_product",
        },
        {
            "operation": "AO overlap",
            "formula": "S_ab=N_a N_b (pi/p)^(3/2) exp[-mu R_AB^2]",
            "why_valid": "Analytic integral of two normalized s Gaussians.",
            "source": "standard Gaussian integral theory",
            "code_location": "h8hf/integrals.py:primitive_overlap",
        },
        {
            "operation": "kinetic integral",
            "formula": "T_ab=mu(3-2 mu R_AB^2) S_ab",
            "why_valid": "Applying -1/2 nabla^2 to an s Gaussian and integrating analytically.",
            "source": "standard Gaussian integral theory",
            "code_location": "h8hf/integrals.py:primitive_kinetic",
        },
        {
            "operation": "nuclear attraction",
            "formula": "V_ab(C)=-Z 2pi/p K_AB F0(p R_PC^2)",
            "why_valid": "Analytic Coulomb attraction integral reduced to the Boys F0 function.",
            "source": "standard Gaussian integral theory",
            "code_location": "h8hf/integrals.py:primitive_nuclear_attraction",
        },
        {
            "operation": "electron repulsion",
            "formula": "(ab|cd)=2 pi^(5/2)/(pq sqrt(p+q)) K_AB K_CD F0(rho R_PQ^2)",
            "why_valid": "Analytic four-center s-Gaussian Coulomb integral.",
            "source": "standard Gaussian integral theory",
            "code_location": "h8hf/integrals.py:primitive_eri",
        },
        {
            "operation": "Roothaan equations",
            "formula": "F C = S C epsilon",
            "why_valid": "Stationarity of the RHF energy under S-orthonormal orbital variations.",
            "source": "Hartree-Fock theory",
            "code_location": "h8hf/rhf.py:generalized_eigh",
        },
        {
            "operation": "RHF Fock matrix",
            "formula": "F_pq=h_pq+sum_rs P_rs[2(pq|rs)-(pr|qs)]",
            "why_valid": "Closed-shell Coulomb and same-spin exchange for one-spin density P.",
            "source": "Hartree-Fock theory",
            "code_location": "h8hf/rhf.py:build_fock",
        },
        {
            "operation": "core-orbital transform",
            "formula": "U_core->HF=C_core^T S C_HF",
            "why_valid": "C_core^T S is the inverse of an S-orthonormal square coefficient matrix.",
            "source": paper,
            "code_location": "run_all.py:build_reproduction",
        },
        {
            "operation": "Slater 1-RDM",
            "formula": "gamma=C_occ C_occ^T, gamma^2=gamma, Tr(gamma)=4",
            "why_valid": "One-particle projector of a four-particle Slater determinant.",
            "source": paper,
            "code_location": "run_all.py:build_reproduction",
        },
        {
            "operation": "Givens action",
            "formula": "G(theta)=[[cos,-sin],[sin,cos]]",
            "why_valid": "Exact SO(2) orbital rotation generated by a_p^dag a_q-a_q^dag a_p.",
            "source": paper,
            "code_location": "h8hf/circuits.py:givens_matrix",
        },
        {
            "operation": "circuit parameter extraction",
            "formula": "min_theta ||gamma(theta)-gamma_RHF||_F^2",
            "why_valid": "A Slater state is determined by its occupied-subspace projector; the 16-parameter diamond spans the half-filled manifold.",
            "source": paper,
            "code_location": "h8hf/circuits.py:fit_projector_to_diamond",
        },
        {
            "operation": "native compilation",
            "formula": "G(theta) ~ RZ(pi), SQISWAP, RZ(pi-theta)RZ(+theta), SQISWAP",
            "why_valid": "Direct 4x4 matrix multiplication gives -G(theta), differing only by global phase.",
            "source": paper,
            "code_location": "h8hf/circuits.py:verify_native_givens",
        },
        {
            "operation": "Slater bitstring probability",
            "formula": "Pr(S)=|det(C_occ[S,:])|^2",
            "why_valid": "Coefficient of a Fock determinant is the corresponding occupied-orbital minor.",
            "source": "Slater determinant expansion",
            "code_location": "h8hf/sampling.py:slater_probabilities",
        },
        {
            "operation": "diagonal 1-RDM",
            "formula": "gamma_pp=<n_p>=sum_x p(x) x_p",
            "why_valid": "Jordan-Wigner maps number to (I-Z_p)/2.",
            "source": paper,
            "code_location": "h8hf/data_processing.py:occupations_from_distribution",
        },
        {
            "operation": "real off-diagonal 1-RDM",
            "formula": "gamma_pq=(<n_q>'-<n_p>')/2 after G_pq(pi/4)",
            "why_valid": "The pi/4 number-conserving analyzer diagonalizes a_p^dag a_q+a_q^dag a_p.",
            "source": paper,
            "code_location": "h8hf/data_processing.py:reconstruct_gamma_from_weights",
        },
        {
            "operation": "two-spin RHF energy",
            "formula": "E=2 h.gamma + 2 gamma.gamma.(pq|rs) - gamma.gamma.(pr|qs) + E_nuc",
            "why_valid": "Wick factorization of the 2-RDM for a closed-shell Slater determinant.",
            "source": paper,
            "code_location": "h8hf/rhf.py:rhf_energy_from_gamma",
        },
        {
            "operation": "rank-4 projection",
            "formula": "gamma_proj=V diag(1,1,1,1,0,0,0,0) V^T",
            "why_valid": "Eckart-Young eigenvalue rounding gives the nearest rank-4 projector in Frobenius norm.",
            "source": paper,
            "code_location": "h8hf/data_processing.py:rank_n_projector",
        },
        {
            "operation": "Slater fidelity",
            "formula": "F=|det(C_ref^T C_meas)|^2",
            "why_valid": "Overlap of two equal-particle Slater determinants is the determinant of the orbital-overlap matrix.",
            "source": paper,
            "code_location": "h8hf/data_processing.py:slater_fidelity",
        },
    ]


def data_lineage() -> list[dict[str, str]]:
    return [
        {
            "stage": "1",
            "inputs": "R=1.3 Å; 8 centered Cartesian H coordinates; STO-3G constants",
            "operation": "analytic contracted Gaussian integrals",
            "outputs": "reference/integrals_and_orbitals.npz; AO CSV matrices",
            "code": "code/h8hf/integrals.py",
        },
        {
            "stage": "2",
            "inputs": "S, h_core, (pq|rs), 8 electrons",
            "operation": "RHF Roothaan iterations with DIIS",
            "outputs": "reference/rhf_reference.json; C_HF; E_RHF",
            "code": "code/h8hf/rhf.py",
        },
        {
            "stage": "3",
            "inputs": "C_core, S, C_HF",
            "operation": "U=C_core^T S C_HF and gamma=U_occ U_occ^T",
            "outputs": "reference/orbital_rotation_core_to_hf.csv; gamma_reference.csv",
            "code": "code/run_all.py",
        },
        {
            "stage": "4",
            "inputs": "gamma_reference and eight pi/4 analysis bases",
            "operation": "fit 16-angle nearest-neighbor diamond per setting",
            "outputs": "circuits/manifest.json; abstract/native qcir files",
            "code": "code/h8hf/circuits.py",
        },
        {
            "stage": "5",
            "inputs": "fitted occupied orbitals",
            "operation": "Pr(S)=|det(C_occ[S,:])|^2 and multinomial sampling",
            "outputs": "data/exact_probabilities; data/sample_1000_shots",
            "code": "code/h8hf/sampling.py",
        },
        {
            "stage": "6",
            "inputs": "nine bitstring distributions and manifest pair map",
            "operation": "normalize; weight-4 post-select; occupations; 1-RDM",
            "outputs": "raw gamma and per-setting diagnostics",
            "code": "code/h8hf/data_processing.py",
        },
        {
            "stage": "7",
            "inputs": "raw gamma; core-basis molecular integrals",
            "operation": "rank-4 projection; Wick/RHF energy; fidelity",
            "outputs": "results/exact and results/sample_1000_shots",
            "code": "code/h8hf/data_processing.py; code/h8hf/rhf.py",
        },
        {
            "stage": "8",
            "inputs": "previous Gaussian log/out, when supplied",
            "operation": "parse final SCF Done energy",
            "outputs": "results/gaussian_comparison.json",
            "code": "code/h8hf/gaussian.py",
        },
    ]


def build_reproduction(
    root: Path,
    shots: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    n_modes = 8
    n_electrons = 8
    n_occupied = 4
    spacing = 1.3
    sample_seed = 20260729
    bootstrap_seed = 20260730
    bit_order = "q[n-1]...q[0]"

    geometry = linear_h_chain(n_modes, spacing, centered=True)
    integrals = build_sto3g_integrals(geometry)
    rhf = run_rhf(
        integrals.h_core,
        integrals.overlap,
        integrals.eri,
        n_electrons,
        integrals.nuclear_repulsion,
    )
    if not rhf.converged:
        raise RuntimeError("H8 RHF did not converge.")

    core_energies, core_coeff, _ = generalized_eigh(
        integrals.h_core, integrals.overlap
    )
    orbital_rotation = (
        core_coeff.T @ integrals.overlap @ rhf.mo_coeff_ao
    )
    determinant_before_fix = float(np.linalg.det(orbital_rotation))
    virtual_sign_flip = False
    if determinant_before_fix < 0:
        orbital_rotation[:, -1] *= -1.0
        virtual_sign_flip = True
    orthogonality_error = float(
        np.max(np.abs(orbital_rotation.T @ orbital_rotation - np.eye(n_modes)))
    )
    reference_gamma = (
        orbital_rotation[:, :n_occupied]
        @ orbital_rotation[:, :n_occupied].T
    )
    h_core_basis, eri_core_basis = transform_integrals(
        integrals.h_core, integrals.eri, core_coeff
    )
    reference_energy = rhf_energy_from_gamma(
        reference_gamma,
        h_core_basis,
        eri_core_basis,
        integrals.nuclear_repulsion,
    )

    reference_dir = root / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        reference_dir / "integrals_and_orbitals.npz",
        geometry_angstrom=geometry,
        geometry_bohr=integrals.coordinates_bohr,
        overlap_ao=integrals.overlap,
        kinetic_ao=integrals.kinetic,
        nuclear_attraction_ao=integrals.nuclear_attraction,
        h_core_ao=integrals.h_core,
        eri_ao=integrals.eri,
        core_orbital_coeff_ao=core_coeff,
        h_core_basis=h_core_basis,
        eri_core_basis=eri_core_basis,
        hf_mo_coeff_ao=rhf.mo_coeff_ao,
        orbital_rotation_core_to_hf=orbital_rotation,
        gamma_reference=reference_gamma,
        orbital_energies_hf=rhf.orbital_energies,
        orbital_energies_core=core_energies,
        nuclear_repulsion=np.array(integrals.nuclear_repulsion),
    )
    save_matrix_csv(reference_dir / "overlap_ao.csv", integrals.overlap)
    save_matrix_csv(reference_dir / "h_core_ao.csv", integrals.h_core)
    save_matrix_csv(reference_dir / "core_orbital_coeff_ao.csv", core_coeff)
    save_matrix_csv(
        reference_dir / "orbital_rotation_core_to_hf.csv",
        orbital_rotation,
    )
    save_matrix_csv(reference_dir / "gamma_reference.csv", reference_gamma)
    write_json(
        reference_dir / "rhf_reference.json",
        {
            "geometry_angstrom": geometry.tolist(),
            "charge": 0,
            "multiplicity": 1,
            "method": "RHF",
            "basis": "STO-3G",
            "n_electrons": n_electrons,
            "n_modes_one_spin": n_modes,
            "n_particles_one_spin": n_occupied,
            "converged": rhf.converged,
            "iterations": rhf.iterations,
            "scf_total_energy_hartree": rhf.total_energy,
            "energy_from_reference_gamma": reference_energy,
            "energy_closure_error_hartree": (
                reference_energy["total"] - rhf.total_energy
            ),
            "nuclear_repulsion_hartree": integrals.nuclear_repulsion,
            "core_to_hf_orthogonality_max_error": orthogonality_error,
            "core_to_hf_determinant_before_virtual_sign_fix": determinant_before_fix,
            "virtual_column_sign_flip_applied": virtual_sign_flip,
            "scf_energy_history_hartree": list(rhf.energy_history),
            "scf_density_rms_history": list(rhf.density_rms_history),
        },
    )

    # Gaussian is a cross-check of the same model.  Existing .log/.out files
    # placed in gaussian/outputs are parsed automatically.
    gaussian_input_dir = root / "gaussian" / "inputs"
    gaussian_output_dir = root / "gaussian" / "outputs"
    gaussian_input_dir.mkdir(parents=True, exist_ok=True)
    gaussian_output_dir.mkdir(parents=True, exist_ok=True)
    (gaussian_input_dir / "H8_R1p3000_RHF_STO3G.gjf").write_text(
        render_gaussian_input(geometry), encoding="utf-8"
    )
    placeholder = gaussian_output_dir / "COPY_PREVIOUS_GAUSSIAN_LOG_HERE.md"
    if not placeholder.exists():
        placeholder.write_text(
            "把此前已经运行完成的 H8 `.log` 或 `.out` 复制到本目录，"
            "再次运行 `python code/run_all.py` 即会自动解析最后一条 "
            "`SCF Done` 并与量子/RHF 结果比较。\n",
            encoding="utf-8",
        )
    gaussian_logs = sorted(gaussian_output_dir.glob("*.log")) + sorted(
        gaussian_output_dir.glob("*.out")
    )
    gaussian_comparison: dict[str, Any] = {
        "status": "no_log_present",
        "message": "Previous Gaussian output was not available in this workspace.",
        "analytic_rhf_reference_hartree": rhf.total_energy,
    }
    if gaussian_logs:
        energy = parse_scf_energy_file(gaussian_logs[-1])
        gaussian_comparison = {
            "status": "parsed",
            "file": gaussian_logs[-1].name,
            "gaussian_scf_energy_hartree": energy,
            "analytic_rhf_reference_hartree": rhf.total_energy,
            "gaussian_minus_reference_hartree": energy - rhf.total_energy,
        }
    write_json(
        root / "results" / "gaussian_comparison.json",
        gaussian_comparison,
    )

    matchings = swap_network_matchings(n_modes)
    setting_specs = [
        {
            "id": "setting_00_diagonal",
            "kind": "diagonal",
            "pairs": [],
            "analyzer_angle_radian": None,
        }
    ]
    setting_specs.extend(
        {
            "id": f"setting_{index + 1:02d}_offdiag",
            "kind": "off_diagonal",
            "pairs": [list(pair) for pair in pairs],
            "analyzer_angle_radian": float(np.pi / 4.0),
            "swap_network_layer": index,
        }
        for index, pairs in enumerate(matchings)
    )

    settings: list[dict[str, Any]] = []
    exact_maps: dict[str, dict[str, float]] = {}
    sample_maps: dict[str, dict[str, int]] = {}
    rng = np.random.default_rng(sample_seed)
    for setting_index, setting in enumerate(setting_specs):
        pairs = [tuple(pair) for pair in setting["pairs"]]
        analyzer = (
            np.eye(n_modes)
            if setting["kind"] == "diagonal"
            else measurement_rotation(n_modes, pairs)
        )
        target_gamma = analyzer @ reference_gamma @ analyzer.T
        fit = fit_projector_to_diamond(
            target_gamma,
            n_occupied,
            seed=sample_seed + setting_index,
            starts=8,
            tolerance=1.0e-11,
        )
        if fit.max_projector_error > 1.0e-10:
            raise RuntimeError(
                f"{setting['id']} circuit did not close: "
                f"{fit.max_projector_error:.3e}"
            )
        abstract_rel = Path("abstract") / f"{setting['id']}.qcir"
        native_rel = Path("native") / f"{setting['id']}.qcir"
        abstract_path = root / "circuits" / abstract_rel
        native_path = root / "circuits" / native_rel
        abstract_path.parent.mkdir(parents=True, exist_ok=True)
        native_path.parent.mkdir(parents=True, exist_ok=True)
        abstract_path.write_text(
            "\n".join(abstract_gate_lines(fit.gates, n_occupied)) + "\n",
            encoding="utf-8",
        )
        native_path.write_text(
            "\n".join(native_gate_lines(fit.gates, n_occupied)) + "\n",
            encoding="utf-8",
        )
        exact = slater_probabilities(
            fit.orbital_rotation[:, :n_occupied], bit_order
        )
        sample = multinomial_counts(exact, shots, rng)
        exact_maps[setting["id"]] = exact
        sample_maps[setting["id"]] = sample
        exact_path = (
            root / "data" / "exact_probabilities" / f"{setting['id']}.json"
        )
        sample_path = (
            root
            / "data"
            / f"sample_{shots}_shots"
            / f"{setting['id']}.json"
        )
        write_json(
            exact_path,
            {
                "setting_id": setting["id"],
                "bit_order": bit_order,
                "probabilities": exact,
            },
        )
        write_json(
            sample_path,
            {
                "setting_id": setting["id"],
                "shots": shots,
                "seed": sample_seed,
                "bit_order": bit_order,
                "counts": sample,
            },
        )
        settings.append(
            {
                **setting,
                "abstract_circuit": str(abstract_rel),
                "native_circuit": str(native_rel),
                "givens_count": len(fit.gates),
                "sqiswap_count": 2 * len(fit.gates),
                "rz_count": 3 * len(fit.gates),
                "x_count": n_occupied,
                "fit_max_projector_error": fit.max_projector_error,
                "fit_frobenius_projector_error": fit.frobenius_projector_error,
                "fit_optimizer_cost": fit.optimizer_cost,
                "fit_optimizer_evaluations": fit.optimizer_evaluations,
                "givens": [
                    {
                        "index": index,
                        "layer": gate.layer,
                        "mode_a": gate.mode_a,
                        "mode_b": gate.mode_b,
                        "theta_radian": gate.theta_radian,
                        "theta_over_pi": gate.theta_radian / np.pi,
                    }
                    for index, gate in enumerate(fit.gates)
                ],
            }
        )

    manifest = {
        "schema_version": 1,
        "system": {
            "name": "linear_H8_R1p3000_RHF_STO3G",
            "n_modes": n_modes,
            "n_electrons_total": n_electrons,
            "n_particles_one_spin": n_occupied,
            "encoded_spin_sector": "alpha; beta is identical under RHF",
            "initial_occupied_modes": list(range(n_occupied)),
        },
        "gate_conventions": {
            "givens_block": "[[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]",
            "rz": "exp(-i*phi*Z/2), phi in radians",
            "sqiswap_single_excitation_block": "(I-iX)/sqrt(2), PyQPanda/OriginQ convention",
            "native_identity_max_error": max(
                verify_native_givens(value)
                for value in (-2.1, -0.2, 0.0, 0.7, 2.4)
            ),
        },
        "measurement": {
            "bit_order": bit_order,
            "settings_total": len(settings),
            "off_diagonal_reconstruction": "gamma[p,q]=(occupation[q]-occupation[p])/2 for each ordered pair",
            "postselection_hamming_weight": n_occupied,
        },
        "settings": settings,
    }
    write_json(root / "circuits" / "manifest.json", manifest)

    exact_files = {
        setting["id"]: root
        / "data"
        / "exact_probabilities"
        / f"{setting['id']}.json"
        for setting in settings
    }
    sample_files = {
        setting["id"]: root
        / "data"
        / f"sample_{shots}_shots"
        / f"{setting['id']}.json"
        for setting in settings
    }
    gamma_exact, exact_diagnostics = reconstruct_gamma(
        exact_files, manifest
    )
    gamma_sample, sample_diagnostics = reconstruct_gamma(
        sample_files, manifest
    )
    exact_analysis = analyze_gamma(
        gamma_exact,
        reference_gamma,
        h_core_basis,
        eri_core_basis,
        integrals.nuclear_repulsion,
        n_occupied,
    )
    sample_analysis = analyze_gamma(
        gamma_sample,
        reference_gamma,
        h_core_basis,
        eri_core_basis,
        integrals.nuclear_repulsion,
        n_occupied,
    )
    exact_analysis["settings"] = exact_diagnostics
    sample_analysis["settings"] = sample_diagnostics
    write_json(root / "results" / "exact" / "summary.json", exact_analysis)
    write_json(
        root / "results" / f"sample_{shots}_shots" / "summary.json",
        sample_analysis,
    )
    save_matrix_csv(
        root / "results" / "exact" / "gamma_reconstructed.csv",
        gamma_exact,
    )
    save_matrix_csv(
        root
        / "results"
        / f"sample_{shots}_shots"
        / "gamma_reconstructed_raw.csv",
        gamma_sample,
    )
    save_matrix_csv(
        root
        / "results"
        / f"sample_{shots}_shots"
        / "gamma_reconstructed_projected.csv",
        np.asarray(sample_analysis["gamma_projected"]),
    )

    bootstrap = bootstrap_energies(
        sample_maps,
        manifest,
        reference_gamma,
        h_core_basis,
        eri_core_basis,
        integrals.nuclear_repulsion,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    write_json(
        root / "results" / f"sample_{shots}_shots" / "bootstrap.json",
        bootstrap,
    )
    shot_plan = plan_shots(
        exact_maps,
        manifest,
        reference_gamma,
        h_core_basis,
        eri_core_basis,
        integrals.nuclear_repulsion,
        shot_levels=[1000, 5000, 10000, 25000],
        repetitions=500,
        seed=20260731,
    )
    write_csv(root / "results" / "shot_planning.csv", shot_plan)

    write_csv(root / "PARAMETER_PROVENANCE.csv", parameter_provenance())
    write_csv(root / "FORMULA_PROVENANCE.csv", formula_provenance())
    write_csv(root / "DATA_LINEAGE.csv", data_lineage())
    write_json(
        root / "RUN_METADATA.json",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "shots_per_setting": shots,
            "sampling_seed": sample_seed,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "shot_planning_repetitions_per_level": 500,
            "shot_planning_seed": 20260731,
        },
    )

    key_summary = {
        "reference_total_energy_hartree": rhf.total_energy,
        "exact_reconstructed_total_energy_hartree": exact_analysis[
            "energy_projected"
        ]["total"],
        "exact_energy_error_hartree": exact_analysis[
            "energy_error_projected_hartree"
        ],
        f"sample_{shots}_projected_total_energy_hartree": sample_analysis[
            "energy_projected"
        ]["total"],
        f"sample_{shots}_projected_error_millihartree": 1000.0
        * sample_analysis["energy_error_projected_hartree"],
        f"sample_{shots}_projected_fidelity": sample_analysis[
            "slater_fidelity_projected"
        ],
        "bootstrap_projected_energy_95_percent_hartree": [
            bootstrap["projected_total_energy_hartree"]["percentile_2p5"],
            bootstrap["projected_total_energy_hartree"]["percentile_97p5"],
        ],
        "settings": len(settings),
        "givens_per_setting": 16,
        "sqiswap_per_setting": 32,
        "rz_per_setting": 48,
        "gaussian": gaussian_comparison,
    }
    write_json(root / "results" / "KEY_RESULTS.json", key_summary)
    return key_summary


def write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    write_csv(root / "SHA256SUMS.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=PACKAGE_ROOT,
        help="Package root to populate (default: repository package root).",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    summary = build_reproduction(
        args.root.resolve(), args.shots, args.bootstrap
    )
    write_manifest(args.root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
