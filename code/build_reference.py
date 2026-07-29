#!/usr/bin/env python3
"""Build the linear-H4 RHF/STO-3G reference and same-basis integrals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from h4_core import (
    rhf_energy_components,
    symmetric_inverse_sqrt,
    write_json,
)


def linear_h4_coordinates(spacing_angstrom: float) -> np.ndarray:
    """Return a centered linear chain with nearest-neighbour spacing R."""
    return np.asarray(
        [
            [0.0, 0.0, (index - 1.5) * spacing_angstrom]
            for index in range(4)
        ],
        dtype=float,
    )


def build_reference(
    spacing_angstrom: float,
) -> dict[str, np.ndarray | float | str | dict]:
    try:
        import pyscf
        from pyscf import ao2mo, gto, scf
    except ImportError as error:
        raise SystemExit(
            "PySCF is required for this step. Install "
            "requirements-chemistry.txt first."
        ) from error

    coordinates = linear_h4_coordinates(spacing_angstrom)
    molecule = gto.M(
        atom=[
            ("H", tuple(coordinates[index])) for index in range(4)
        ],
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    molecule.incore_anyway = True
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-12
    mean_field.conv_tol_grad = 1e-10
    mean_field.max_cycle = 512
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

    # chi_orth = chi_AO X, therefore C_orth = X^{-1} C_AO.
    mo_coeff_orth = np.linalg.solve(
        orthogonalizer, mean_field.mo_coeff
    )
    if np.linalg.norm(
        mo_coeff_orth.T @ mo_coeff_orth - np.eye(4)
    ) > 1e-9:
        raise RuntimeError("Orthogonal-basis MO coefficients are not unitary.")
    occupied_orbitals = mo_coeff_orth[:, :2]
    gamma_one_spin = occupied_orbitals @ occupied_orbitals.T

    components = rhf_energy_components(
        gamma_one_spin,
        h1_orth,
        eri_orth,
        molecule.energy_nuc(),
    )
    closure_error = components["total_hartree"] - rhf_energy
    if abs(closure_error) > 1e-9:
        raise RuntimeError(
            "1-RDM energy functional failed to reproduce RHF: "
            f"{components['total_hartree']:.12f} vs {rhf_energy:.12f}."
        )

    return {
        "molecule": "linear H4",
        "spacing_angstrom": float(spacing_angstrom),
        "geometry_angstrom": coordinates,
        "basis": "STO-3G",
        "method": "RHF",
        "charge": 0.0,
        "multiplicity": 1.0,
        "n_spatial_orbitals": 4.0,
        "n_electrons": 4.0,
        "n_particles_per_spin": 2.0,
        "nuclear_repulsion": float(molecule.energy_nuc()),
        "rhf_energy": rhf_energy,
        "overlap_ao": overlap_ao,
        "orthogonalizer_ao_to_orth": orthogonalizer,
        "h1_orth": h1_orth,
        "eri_orth": eri_orth,
        "mo_coeff_ao": mean_field.mo_coeff,
        "mo_coeff_orth": mo_coeff_orth,
        "mo_energy": mean_field.mo_energy,
        "mo_occupations_spin_summed": mean_field.mo_occ,
        "occupied_orbitals_orth": occupied_orbitals,
        "gamma_one_spin_orth": gamma_one_spin,
        "ao_density_spin_summed": mean_field.make_rdm1(),
        "energy_closure_error_hartree": float(closure_error),
        "energy_components": components,
        "pyscf_version": pyscf.__version__,
    }


def save_reference(
    reference: dict[str, np.ndarray | float | str | dict],
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
    parser.add_argument("--spacing", type=float, default=1.3)
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("reference/h4_r1.3000_sto3g_reference.npz"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reference/h4_r1.3000_sto3g_reference.json"),
    )
    arguments = parser.parse_args()
    if arguments.spacing <= 0:
        raise ValueError("--spacing must be positive.")

    reference = build_reference(arguments.spacing)
    save_reference(reference, arguments.output_npz, arguments.output_json)
    print(
        json.dumps(
            {
                "spacing_angstrom": reference["spacing_angstrom"],
                "rhf_energy_hartree": reference["rhf_energy"],
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
