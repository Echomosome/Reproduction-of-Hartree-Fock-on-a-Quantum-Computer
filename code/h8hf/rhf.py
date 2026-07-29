"""Restricted Hartree--Fock and the one-spin 1-RDM energy functional."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RHFResult:
    converged: bool
    iterations: int
    total_energy: float
    electronic_energy: float
    nuclear_repulsion: float
    orbital_energies: np.ndarray
    mo_coeff_ao: np.ndarray
    density_ao_one_spin: np.ndarray
    fock_ao: np.ndarray
    energy_history: tuple[float, ...]
    density_rms_history: tuple[float, ...]


def symmetric_orthogonalizer(overlap: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if np.min(eigenvalues) < 1.0e-10:
        raise ValueError("The AO overlap matrix is linearly dependent.")
    return (eigenvectors * eigenvalues**-0.5) @ eigenvectors.T


def generalized_eigh(
    matrix: np.ndarray, overlap: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve F C = S C e and return e, AO C, orthonormal-basis C."""
    x = symmetric_orthogonalizer(overlap)
    transformed = x.T @ matrix @ x
    eigenvalues, coeff_orth = np.linalg.eigh(transformed)
    coeff_ao = x @ coeff_orth
    return eigenvalues, coeff_ao, coeff_orth


def build_fock(
    h_core: np.ndarray, eri: np.ndarray, density_one_spin: np.ndarray
) -> np.ndarray:
    coulomb = np.einsum(
        "rs,pqrs->pq", density_one_spin, eri, optimize=True
    )
    exchange = np.einsum(
        "rs,prqs->pq", density_one_spin, eri, optimize=True
    )
    return h_core + 2.0 * coulomb - exchange


def ao_rhf_energy(
    h_core: np.ndarray,
    fock: np.ndarray,
    density_one_spin: np.ndarray,
    nuclear_repulsion: float,
) -> tuple[float, float]:
    electronic = float(np.einsum("pq,pq", density_one_spin, h_core + fock))
    return electronic + nuclear_repulsion, electronic


class _DIIS:
    def __init__(self, max_vectors: int = 8) -> None:
        self.max_vectors = max_vectors
        self.focks: list[np.ndarray] = []
        self.errors: list[np.ndarray] = []

    def add(self, fock: np.ndarray, error: np.ndarray) -> None:
        self.focks.append(fock.copy())
        self.errors.append(error.ravel().copy())
        if len(self.focks) > self.max_vectors:
            self.focks.pop(0)
            self.errors.pop(0)

    def extrapolate(self) -> np.ndarray:
        n = len(self.focks)
        if n < 2:
            return self.focks[-1]
        b_matrix = np.empty((n + 1, n + 1))
        b_matrix[:n, :n] = np.array(
            [[np.dot(a, b) for b in self.errors] for a in self.errors]
        )
        b_matrix[:n, n] = -1.0
        b_matrix[n, :n] = -1.0
        b_matrix[n, n] = 0.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        # A tiny diagonal regularizer only handles nearly duplicate DIIS errors.
        b_matrix[:n, :n] += np.eye(n) * 1.0e-14
        try:
            coefficients = np.linalg.solve(b_matrix, rhs)[:n]
        except np.linalg.LinAlgError:
            return self.focks[-1]
        return sum(
            coefficient * fock
            for coefficient, fock in zip(coefficients, self.focks)
        )


def run_rhf(
    h_core: np.ndarray,
    overlap: np.ndarray,
    eri: np.ndarray,
    n_electrons: int,
    nuclear_repulsion: float,
    *,
    max_iterations: int = 256,
    energy_tolerance: float = 1.0e-12,
    density_tolerance: float = 1.0e-10,
    diis_start: int = 2,
    diis_size: int = 8,
) -> RHFResult:
    if n_electrons % 2:
        raise ValueError("RHF requires an even number of electrons here.")
    n_occupied = n_electrons // 2
    _, core_coeff, _ = generalized_eigh(h_core, overlap)
    occupied = core_coeff[:, :n_occupied]
    density = occupied @ occupied.T
    diis = _DIIS(diis_size)
    energy_history: list[float] = []
    density_history: list[float] = []
    previous_energy: float | None = None

    converged = False
    final_coeff = core_coeff
    final_orbital_energies = np.zeros(h_core.shape[0])
    final_fock = build_fock(h_core, eri, density)

    for iteration in range(1, max_iterations + 1):
        fock = build_fock(h_core, eri, density)
        commutator = fock @ density @ overlap - overlap @ density @ fock
        diis.add(fock, commutator)
        effective_fock = (
            diis.extrapolate() if iteration >= diis_start else fock
        )
        orbital_energies, coeff, _ = generalized_eigh(
            effective_fock, overlap
        )
        occupied = coeff[:, :n_occupied]
        new_density = occupied @ occupied.T

        # Mild early damping is deterministic and prevents initial oscillation.
        if iteration <= 3:
            new_density = 0.8 * new_density + 0.2 * density

        new_fock = build_fock(h_core, eri, new_density)
        total_energy, _ = ao_rhf_energy(
            h_core, new_fock, new_density, nuclear_repulsion
        )
        density_rms = float(np.sqrt(np.mean((new_density - density) ** 2)))
        energy_history.append(total_energy)
        density_history.append(density_rms)
        energy_change = (
            np.inf
            if previous_energy is None
            else abs(total_energy - previous_energy)
        )

        density = new_density
        previous_energy = total_energy
        final_coeff = coeff
        final_orbital_energies = orbital_energies
        final_fock = new_fock
        if (
            energy_change < energy_tolerance
            and density_rms < density_tolerance
        ):
            converged = True
            break

    # Remove any residual damping/non-idempotency by diagonalizing final F once.
    final_orbital_energies, final_coeff, _ = generalized_eigh(
        build_fock(h_core, eri, density), overlap
    )
    final_occupied = final_coeff[:, :n_occupied]
    final_density = final_occupied @ final_occupied.T
    final_fock = build_fock(h_core, eri, final_density)
    total_energy, electronic_energy = ao_rhf_energy(
        h_core, final_fock, final_density, nuclear_repulsion
    )

    return RHFResult(
        converged=converged,
        iterations=iteration,
        total_energy=total_energy,
        electronic_energy=electronic_energy,
        nuclear_repulsion=nuclear_repulsion,
        orbital_energies=final_orbital_energies,
        mo_coeff_ao=final_coeff,
        density_ao_one_spin=final_density,
        fock_ao=final_fock,
        energy_history=tuple(energy_history),
        density_rms_history=tuple(density_history),
    )


def transform_integrals(
    h_ao: np.ndarray, eri_ao: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h_new = coefficients.T @ h_ao @ coefficients
    eri_new = np.einsum(
        "mp,nq,kr,ls,mnkl->pqrs",
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        eri_ao,
        optimize=True,
    )
    return h_new, eri_new


def rhf_energy_from_gamma(
    gamma_one_spin: np.ndarray,
    h_one_body: np.ndarray,
    eri_chemist: np.ndarray,
    nuclear_repulsion: float,
) -> dict[str, float]:
    """Evaluate RHF energy from a one-spin spatial-orbital 1-RDM.

    E = 2 Tr[h gamma]
        + 2 sum gamma_pq gamma_rs (pq|rs)
        -   sum gamma_pq gamma_rs (pr|qs)
        + E_nuc.
    """
    gamma = np.asarray(gamma_one_spin, dtype=float)
    one_electron = 2.0 * np.einsum(
        "pq,pq", gamma, h_one_body, optimize=True
    )
    coulomb = 2.0 * np.einsum(
        "pq,rs,pqrs", gamma, gamma, eri_chemist, optimize=True
    )
    exchange = -np.einsum(
        "pq,rs,prqs", gamma, gamma, eri_chemist, optimize=True
    )
    electronic = float(one_electron + coulomb + exchange)
    total = electronic + nuclear_repulsion
    return {
        "one_electron": float(one_electron),
        "coulomb": float(coulomb),
        "exchange": float(exchange),
        "electronic": electronic,
        "nuclear_repulsion": float(nuclear_repulsion),
        "total": float(total),
    }
