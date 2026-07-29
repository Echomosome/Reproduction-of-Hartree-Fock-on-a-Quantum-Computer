#!/usr/bin/env python3
"""Build full and 10-mode frozen-core RHF references for diazene."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from diazene_core import (
    N_FROZEN_ORBITALS,
    N_FULL_MODES,
    N_FULL_PARTICLES_PER_SPIN,
    N_MODES,
    N_PARTICLES_PER_SPIN,
    active_fock,
    cartesian_array,
    geometry_metrics,
    load_geometries,
    rhf_energy_components,
    write_json,
)


def canonicalize_column_signs(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def rhf_energy_ao(
    density_spin_summed: np.ndarray,
    hcore_ao: np.ndarray,
    j_ao: np.ndarray,
    k_ao: np.ndarray,
    nuclear_repulsion: float,
) -> float:
    fock = hcore_ao + j_ao - 0.5 * k_ao
    return float(
        nuclear_repulsion
        + 0.5
        * np.einsum(
            "pq,qp->",
            density_spin_summed,
            hcore_ao + fock,
            optimize=True,
        )
    )


def preliminary_two_cycle_scf(
    mean_field,
    overlap_ao: np.ndarray,
    hcore_ao: np.ndarray,
    n_occupied: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Perform exactly two undamped, no-DIIS canonical RHF updates."""
    from scipy.linalg import eigh

    molecule = mean_field.mol
    density = np.asarray(
        mean_field.get_init_guess(molecule, key="minao"), dtype=float
    )
    history: list[dict] = []
    for cycle in range(1, 3):
        coulomb, exchange = mean_field.get_jk(molecule, density)
        fock = hcore_ao + coulomb - 0.5 * exchange
        orbital_energies, coefficients = eigh(fock, overlap_ao)
        coefficients = canonicalize_column_signs(coefficients)
        occupied = coefficients[:, :n_occupied]
        density = 2.0 * occupied @ occupied.T

        # Rebuild J and K from the updated density so the recorded energy is
        # the energy of that density, not the mixed old/new SCF expression.
        coulomb_new, exchange_new = mean_field.get_jk(molecule, density)
        energy = rhf_energy_ao(
            density,
            hcore_ao,
            coulomb_new,
            exchange_new,
            molecule.energy_nuc(),
        )
        commutator = (
            (hcore_ao + coulomb_new - 0.5 * exchange_new)
            @ density
            @ overlap_ao
            - overlap_ao
            @ density
            @ (hcore_ao + coulomb_new - 0.5 * exchange_new)
        )
        history.append(
            {
                "cycle": cycle,
                "energy_hartree": energy,
                "commutator_frobenius": float(
                    np.linalg.norm(commutator)
                ),
            }
        )
    return coefficients, orbital_energies, history


def frozen_core_integrals(
    h1_pre_mo: np.ndarray,
    eri_pre_mo: np.ndarray,
    nuclear_repulsion: float,
    n_frozen: int = N_FROZEN_ORBITALS,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """Integrate out the first ``n_frozen`` doubly occupied orbitals."""
    n_total = h1_pre_mo.shape[0]
    frozen = list(range(n_frozen))
    active = list(range(n_frozen, n_total))

    frozen_one = 2.0 * sum(h1_pre_mo[index, index] for index in frozen)
    frozen_coulomb = sum(
        2.0 * eri_pre_mo[i, i, j, j]
        for i in frozen
        for j in frozen
    )
    frozen_exchange = sum(
        eri_pre_mo[i, j, i, j]
        for i in frozen
        for j in frozen
    )
    constant = float(
        nuclear_repulsion
        + frozen_one
        + frozen_coulomb
        - frozen_exchange
    )

    h1_active = h1_pre_mo[np.ix_(active, active)].copy()
    for p_active, p_full in enumerate(active):
        for q_active, q_full in enumerate(active):
            core_shift = 0.0
            for frozen_index in frozen:
                core_shift += (
                    2.0
                    * eri_pre_mo[
                        p_full, q_full, frozen_index, frozen_index
                    ]
                    - eri_pre_mo[
                        p_full, frozen_index, q_full, frozen_index
                    ]
                )
            h1_active[p_active, q_active] += core_shift
    eri_active = eri_pre_mo[np.ix_(active, active, active, active)]
    components = {
        "nuclear_repulsion_hartree": float(nuclear_repulsion),
        "frozen_one_electron_hartree": float(frozen_one),
        "frozen_coulomb_hartree": float(frozen_coulomb),
        "frozen_exchange_hartree": float(frozen_exchange),
        "constant_offset_hartree": constant,
    }
    return constant, h1_active, eri_active, components


def active_rhf(
    h1_active: np.ndarray,
    eri_active: np.ndarray,
    constant_offset: float,
    n_particles: int = N_PARTICLES_PER_SPIN,
    density_tolerance: float = 1e-12,
    energy_tolerance: float = 1e-13,
    max_cycle: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Solve the frozen-core RHF problem in the orthonormal active basis."""
    n_modes = h1_active.shape[0]
    gamma = np.diag(
        [1.0] * n_particles + [0.0] * (n_modes - n_particles)
    )
    previous_energy: float | None = None
    fock_history: list[np.ndarray] = []
    error_history: list[np.ndarray] = []
    records: list[dict] = []

    for cycle in range(1, max_cycle + 1):
        fock = active_fock(gamma, h1_active, eri_active)
        error = fock @ gamma - gamma @ fock
        fock_history.append(fock.copy())
        error_history.append(error.reshape(-1).copy())
        if len(fock_history) > 8:
            fock_history.pop(0)
            error_history.pop(0)

        fock_effective = fock
        if cycle >= 3 and len(fock_history) >= 2:
            size = len(fock_history)
            pulay = np.empty((size + 1, size + 1), dtype=float)
            pulay[:size, :size] = np.asarray(
                [
                    [
                        np.dot(error_history[i], error_history[j])
                        for j in range(size)
                    ]
                    for i in range(size)
                ]
            )
            pulay[:size, size] = -1.0
            pulay[size, :size] = -1.0
            pulay[size, size] = 0.0
            rhs = np.zeros(size + 1)
            rhs[size] = -1.0
            try:
                coefficients = np.linalg.solve(pulay, rhs)[:size]
                fock_effective = sum(
                    coefficient * stored
                    for coefficient, stored in zip(
                        coefficients, fock_history
                    )
                )
            except np.linalg.LinAlgError:
                fock_effective = fock

        orbital_energies, orbitals = np.linalg.eigh(fock_effective)
        orbitals = canonicalize_column_signs(orbitals)
        occupied = orbitals[:, :n_particles]
        gamma_new = occupied @ occupied.T
        density_change = float(np.linalg.norm(gamma_new - gamma))
        energy = rhf_energy_components(
            gamma_new, h1_active, eri_active, constant_offset
        )["total_hartree"]
        energy_change = (
            float("inf")
            if previous_energy is None
            else abs(energy - previous_energy)
        )
        records.append(
            {
                "cycle": cycle,
                "energy_hartree": energy,
                "energy_change_hartree": energy_change,
                "density_change_frobenius": density_change,
                "commutator_frobenius": float(np.linalg.norm(error)),
            }
        )
        gamma = gamma_new
        previous_energy = energy
        if (
            density_change < density_tolerance
            and energy_change < energy_tolerance
        ):
            # Re-diagonalize the self-consistent Fock matrix to store a
            # mutually consistent final coefficient/energy pair.
            fock_final = active_fock(gamma, h1_active, eri_active)
            orbital_energies, orbitals = np.linalg.eigh(fock_final)
            orbitals = canonicalize_column_signs(orbitals)
            occupied = orbitals[:, :n_particles]
            gamma = occupied @ occupied.T
            return gamma, orbitals, orbital_energies, records

    raise RuntimeError(
        f"Active RHF failed to converge in {max_cycle} cycles."
    )


def build_one_reference(geometry: dict) -> dict:
    try:
        import pyscf
        from pyscf import ao2mo, gto, scf
    except ImportError as error:
        raise SystemExit(
            "PySCF and SciPy are required. Install "
            "requirements-chemistry.txt."
        ) from error

    atoms = [
        (atom["element"], tuple(atom["xyz_angstrom"]))
        for atom in geometry["atoms"]
    ]
    molecule = gto.M(
        atom=atoms,
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    molecule.incore_anyway = True
    if molecule.nao_nr() != N_FULL_MODES:
        raise RuntimeError(
            f"Expected {N_FULL_MODES} STO-3G AOs, "
            f"found {molecule.nao_nr()}."
        )
    if molecule.nelectron != 2 * N_FULL_PARTICLES_PER_SPIN:
        raise RuntimeError(
            f"Expected {2 * N_FULL_PARTICLES_PER_SPIN} electrons, "
            f"found {molecule.nelectron}."
        )

    full_rhf = scf.RHF(molecule)
    full_rhf.conv_tol = 1e-12
    full_rhf.conv_tol_grad = 1e-10
    full_rhf.max_cycle = 512
    full_energy = float(full_rhf.kernel())
    if not full_rhf.converged:
        raise RuntimeError(f"{geometry['id']}: full RHF did not converge.")

    overlap_ao = np.asarray(full_rhf.get_ovlp(), dtype=float)
    hcore_ao = np.asarray(full_rhf.get_hcore(), dtype=float)
    preliminary_coefficients, preliminary_energies, preliminary_history = (
        preliminary_two_cycle_scf(
            full_rhf,
            overlap_ao,
            hcore_ao,
            N_FULL_PARTICLES_PER_SPIN,
        )
    )
    orthogonality_error = np.linalg.norm(
        preliminary_coefficients.T
        @ overlap_ao
        @ preliminary_coefficients
        - np.eye(N_FULL_MODES)
    )
    if orthogonality_error > 1e-9:
        raise RuntimeError(
            "Preliminary orbitals are not S-orthonormal: "
            f"{orthogonality_error:.3e}."
        )

    h1_pre_mo = (
        preliminary_coefficients.T
        @ hcore_ao
        @ preliminary_coefficients
    )
    eri_pre_mo = ao2mo.restore(
        1,
        ao2mo.kernel(molecule, preliminary_coefficients),
        N_FULL_MODES,
    )
    (
        constant_offset,
        h1_active,
        eri_active,
        frozen_components,
    ) = frozen_core_integrals(
        h1_pre_mo,
        eri_pre_mo,
        molecule.energy_nuc(),
        N_FROZEN_ORBITALS,
    )
    (
        gamma_active,
        active_mo_coeff,
        active_mo_energy,
        active_history,
    ) = active_rhf(
        h1_active,
        eri_active,
        constant_offset,
        N_PARTICLES_PER_SPIN,
    )
    active_components = rhf_energy_components(
        gamma_active, h1_active, eri_active, constant_offset
    )
    active_energy = active_components["total_hartree"]
    occupied_orbitals = active_mo_coeff[:, :N_PARTICLES_PER_SPIN]
    closure_error = float(
        active_energy
        - rhf_energy_components(
            occupied_orbitals @ occupied_orbitals.T,
            h1_active,
            eri_active,
            constant_offset,
        )["total_hartree"]
    )
    if abs(closure_error) > 1e-10:
        raise RuntimeError("Active energy closure failed.")

    return {
        "geometry_id": geometry["id"],
        "pathway": geometry["pathway"],
        "path_index": int(geometry["path_index"]),
        "reaction_coordinate_deg": float(
            geometry["reaction_coordinate_deg"]
        ),
        "role": geometry["role"],
        "geometry_angstrom": cartesian_array(geometry),
        "geometry_metrics": geometry_metrics(geometry),
        "method": "RHF",
        "basis": "STO-3G",
        "charge": 0,
        "multiplicity": 1,
        "n_full_spatial_modes": N_FULL_MODES,
        "n_full_electrons": 2 * N_FULL_PARTICLES_PER_SPIN,
        "n_frozen_spatial_orbitals": N_FROZEN_ORBITALS,
        "n_active_spatial_modes": N_MODES,
        "n_active_particles_per_spin": N_PARTICLES_PER_SPIN,
        "full_rhf_energy_hartree": full_energy,
        "active_rhf_energy_hartree": active_energy,
        "frozen_core_bias_hartree": active_energy - full_energy,
        "nuclear_repulsion_hartree": float(molecule.energy_nuc()),
        "overlap_ao": overlap_ao,
        "hcore_ao": hcore_ao,
        "full_mo_coeff_ao": canonicalize_column_signs(
            full_rhf.mo_coeff
        ),
        "full_mo_energy": np.asarray(full_rhf.mo_energy, dtype=float),
        "full_density_ao_spin_summed": np.asarray(
            full_rhf.make_rdm1(), dtype=float
        ),
        "preliminary_mo_coeff_ao": preliminary_coefficients,
        "preliminary_mo_energy": preliminary_energies,
        "h1_preliminary_mo": h1_pre_mo,
        "eri_preliminary_mo": eri_pre_mo,
        "constant_offset_hartree": constant_offset,
        "h1_active": h1_active,
        "eri_active": eri_active,
        "active_mo_coeff": active_mo_coeff,
        "active_mo_energy": active_mo_energy,
        "occupied_orbitals_active": occupied_orbitals,
        "gamma_active_one_spin": gamma_active,
        "preliminary_scf_history": preliminary_history,
        "active_scf_history": active_history,
        "frozen_core_components": frozen_components,
        "active_energy_components": active_components,
        "active_energy_closure_error_hartree": closure_error,
        "pyscf_version": pyscf.__version__,
    }


def serializable(reference: dict) -> dict:
    result: dict = {}
    for key, value in reference.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, np.integer):
            result[key] = int(value)
        else:
            result[key] = value
    return result


def save_reference(reference: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_id = str(reference["geometry_id"])
    arrays = {
        key: value
        for key, value in reference.items()
        if isinstance(value, np.ndarray)
    }
    scalars = {
        key: value
        for key, value in reference.items()
        if isinstance(value, (str, int, float, np.integer, np.floating))
    }
    np.savez(output_dir / f"{geometry_id}.npz", **arrays, **scalars)
    write_json(
        output_dir / f"{geometry_id}.json",
        serializable(reference),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometries",
        type=Path,
        default=Path("data/geometries_paper.json"),
    )
    parser.add_argument(
        "--geometry-id",
        default=None,
        help="Build only one geometry; default builds all 18.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reference")
    )
    arguments = parser.parse_args()

    geometries = load_geometries(arguments.geometries)
    if arguments.geometry_id is not None:
        geometries = [
            geometry
            for geometry in geometries
            if geometry["id"] == arguments.geometry_id
        ]
        if not geometries:
            raise KeyError(
                f"Unknown geometry id {arguments.geometry_id!r}."
            )

    rows: list[dict] = []
    for geometry in geometries:
        reference = build_one_reference(geometry)
        save_reference(reference, arguments.output_dir)
        row = {
            "geometry_id": reference["geometry_id"],
            "pathway": reference["pathway"],
            "path_index": reference["path_index"],
            "reaction_coordinate_deg": reference[
                "reaction_coordinate_deg"
            ],
            "full_rhf_energy_hartree": reference[
                "full_rhf_energy_hartree"
            ],
            "active_rhf_energy_hartree": reference[
                "active_rhf_energy_hartree"
            ],
            "frozen_core_bias_millihartree": 1000.0
            * reference["frozen_core_bias_hartree"],
            "preliminary_cycle_1_energy_hartree": reference[
                "preliminary_scf_history"
            ][0]["energy_hartree"],
            "preliminary_cycle_2_energy_hartree": reference[
                "preliminary_scf_history"
            ][1]["energy_hartree"],
            "active_scf_cycles": len(reference["active_scf_history"]),
        }
        rows.append(row)
        print(
            f"{reference['geometry_id']}: "
            f"E_full={reference['full_rhf_energy_hartree']:.12f} Ha, "
            f"E_active={reference['active_rhf_energy_hartree']:.12f} Ha, "
            f"bias={row['frozen_core_bias_millihartree']:.6f} mHa"
        )

    rows.sort(key=lambda row: (row["pathway"], row["path_index"]))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "reference_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        arguments.output_dir / "reference_summary.json",
        {
            "model": {
                "method": "RHF",
                "basis": "STO-3G",
                "full_spatial_modes": N_FULL_MODES,
                "full_electrons": 2 * N_FULL_PARTICLES_PER_SPIN,
                "preliminary_scf_cycles": 2,
                "frozen_spatial_orbitals": N_FROZEN_ORBITALS,
                "active_spatial_modes": N_MODES,
                "active_particles_per_spin": N_PARTICLES_PER_SPIN,
            },
            "geometries": rows,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
