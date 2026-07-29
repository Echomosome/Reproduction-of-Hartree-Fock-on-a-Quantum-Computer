#!/usr/bin/env python3
"""Build the H2 RHF/STO-3G same-basis molecular reference with PySCF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from h2_core import (
    givens_angle_from_orbital,
    phase_fix_real_vector,
    rhf_energy_components,
    symmetric_inverse_sqrt,
    write_json,
)


def build_reference(
    bond_length_angstrom: float,
) -> dict[str, np.ndarray | float | str]:
    try:
        from pyscf import ao2mo, gto, scf
    except ImportError as error:
        raise SystemExit(
            "PySCF is required for this step. Install "
            "requirements-chemistry.txt first."
        ) from error

    half = bond_length_angstrom / 2.0
    molecule = gto.M(
        atom=[
            ("H", (0.0, 0.0, -half)),
            ("H", (0.0, 0.0, half)),
        ],
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    # Some restricted containers do not expose /proc.  Keeping the tiny H2
    # integral tensor in memory also makes the calculation deterministic.
    molecule.incore_anyway = True

    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-10
    rhf_energy = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError("PySCF RHF did not converge.")

    overlap_ao = mean_field.get_ovlp()
    orthogonalizer = symmetric_inverse_sqrt(overlap_ao)
    h1_orth = orthogonalizer.T @ mean_field.get_hcore() @ orthogonalizer
    eri_orth = ao2mo.restore(
        1,
        ao2mo.kernel(molecule, orthogonalizer),
        molecule.nao_nr(),
    )

    # chi_orth = chi_AO X, so C_orth = X^{-1} C_AO.
    mo_coeff_orth = np.linalg.solve(
        orthogonalizer, mean_field.mo_coeff
    )
    occupied_orbital = phase_fix_real_vector(mo_coeff_orth[:, 0])
    gamma_one_spin = np.outer(occupied_orbital, occupied_orbital)
    theta = givens_angle_from_orbital(occupied_orbital)

    components = rhf_energy_components(
        gamma_one_spin,
        h1_orth,
        eri_orth,
        molecule.energy_nuc(),
    )
    closure_error = components["total_hartree"] - rhf_energy
    if abs(closure_error) > 1e-10:
        raise RuntimeError(
            "1-RDM energy functional did not reproduce RHF: "
            f"{components['total_hartree']:.12f} vs {rhf_energy:.12f}."
        )

    return {
        "bond_length_angstrom": float(bond_length_angstrom),
        "basis": "STO-3G",
        "method": "RHF",
        "charge": 0.0,
        "multiplicity": 1.0,
        "nuclear_repulsion": float(molecule.energy_nuc()),
        "rhf_energy": rhf_energy,
        "overlap_ao": overlap_ao,
        "orthogonalizer_ao_to_orth": orthogonalizer,
        "h1_orth": h1_orth,
        "eri_orth": eri_orth,
        "mo_coeff_ao": mean_field.mo_coeff,
        "mo_coeff_orth": mo_coeff_orth,
        "occupied_orbital_orth": occupied_orbital,
        "gamma_one_spin_orth": gamma_one_spin,
        "ao_density_spin_summed": mean_field.make_rdm1(),
        "givens_theta_radians": theta,
        "givens_theta_over_pi": theta / np.pi,
        "energy_closure_error_hartree": float(closure_error),
        "energy_components": components,
        "pyscf_version": __import__("pyscf").__version__,
    }


def save_reference(
    reference: dict[str, np.ndarray | float | str],
    output_npz: Path,
    output_json: Path,
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        **{
            key: value
            for key, value in reference.items()
            if key != "energy_components"
        },
    )

    serializable: dict = {}
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        else:
            serializable[key] = value
    write_json(output_json, serializable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond-length", type=float, default=0.7414)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("reference/h2_r0.7414_sto3g_reference.npz"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reference/h2_r0.7414_sto3g_reference.json"),
    )
    arguments = parser.parse_args()

    reference = build_reference(arguments.bond_length)
    save_reference(reference, arguments.output_npz, arguments.output_json)
    print(
        json.dumps(
            {
                "bond_length_angstrom": reference[
                    "bond_length_angstrom"
                ],
                "rhf_energy_hartree": reference["rhf_energy"],
                "givens_theta_radians": reference[
                    "givens_theta_radians"
                ],
                "energy_closure_error_hartree": reference[
                    "energy_closure_error_hartree"
                ],
                "saved_npz": str(arguments.output_npz),
                "saved_json": str(arguments.output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
