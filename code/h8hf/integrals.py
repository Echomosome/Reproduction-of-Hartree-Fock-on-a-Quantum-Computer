"""Analytic s-type Gaussian integrals for a minimal H/STO-3G basis.

The implementation is intentionally small and explicit.  It is not intended to
replace a production electronic-structure package; it provides an independently
auditable reference for the fixed H8/STO-3G calculation in this repository.
All internal distances are in bohr and all energies are in hartree.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt
from typing import Iterable

import numpy as np


ANGSTROM_TO_BOHR = 1.8897261254578281

# Hydrogen 1s STO-3G contraction (Hehre, Stewart, Pople convention).
H_STO3G_EXPONENTS = np.array([3.42525091, 0.62391373, 0.16885540])
H_STO3G_COEFFICIENTS = np.array([0.15432897, 0.53532814, 0.44463454])


@dataclass(frozen=True)
class IntegralBundle:
    overlap: np.ndarray
    kinetic: np.ndarray
    nuclear_attraction: np.ndarray
    h_core: np.ndarray
    eri: np.ndarray
    nuclear_repulsion: float
    coordinates_bohr: np.ndarray
    charges: np.ndarray


def linear_h_chain(
    n_atoms: int = 8,
    spacing_angstrom: float = 1.3,
    centered: bool = True,
) -> np.ndarray:
    """Return a linear H_n geometry along z, in angstrom."""
    z = np.arange(n_atoms, dtype=float) * spacing_angstrom
    if centered:
        z -= np.mean(z)
    xyz = np.zeros((n_atoms, 3), dtype=float)
    xyz[:, 2] = z
    return xyz


def _primitive_norm(alpha: float) -> float:
    return (2.0 * alpha / pi) ** 0.75


def boys0(t: float) -> float:
    """Boys function F_0(t), including a stable small-t expansion."""
    if t < 1.0e-10:
        return 1.0 - t / 3.0 + t * t / 10.0 - t**3 / 42.0
    root_t = sqrt(t)
    return 0.5 * sqrt(pi / t) * erf(root_t)


def _gaussian_product(
    alpha: float,
    center_a: np.ndarray,
    beta: float,
    center_b: np.ndarray,
) -> tuple[float, float, np.ndarray, float]:
    p = alpha + beta
    mu = alpha * beta / p
    rab2 = float(np.dot(center_a - center_b, center_a - center_b))
    product_center = (alpha * center_a + beta * center_b) / p
    kab = np.exp(-mu * rab2)
    return p, mu, product_center, float(kab)


def primitive_overlap(
    alpha: float,
    center_a: np.ndarray,
    beta: float,
    center_b: np.ndarray,
) -> float:
    p, _, _, kab = _gaussian_product(alpha, center_a, beta, center_b)
    return _primitive_norm(alpha) * _primitive_norm(beta) * (pi / p) ** 1.5 * kab


def primitive_kinetic(
    alpha: float,
    center_a: np.ndarray,
    beta: float,
    center_b: np.ndarray,
) -> float:
    p, mu, _, kab = _gaussian_product(alpha, center_a, beta, center_b)
    rab2 = float(np.dot(center_a - center_b, center_a - center_b))
    unnormalized_overlap = (pi / p) ** 1.5 * kab
    prefactor = mu * (3.0 - 2.0 * mu * rab2)
    return (
        _primitive_norm(alpha)
        * _primitive_norm(beta)
        * prefactor
        * unnormalized_overlap
    )


def primitive_nuclear_attraction(
    alpha: float,
    center_a: np.ndarray,
    beta: float,
    center_b: np.ndarray,
    nucleus: np.ndarray,
    charge: float,
) -> float:
    p, _, product_center, kab = _gaussian_product(
        alpha, center_a, beta, center_b
    )
    rpc2 = float(np.dot(product_center - nucleus, product_center - nucleus))
    value = -charge * 2.0 * pi / p * kab * boys0(p * rpc2)
    return _primitive_norm(alpha) * _primitive_norm(beta) * value


def primitive_eri(
    alpha: float,
    center_a: np.ndarray,
    beta: float,
    center_b: np.ndarray,
    gamma: float,
    center_c: np.ndarray,
    delta: float,
    center_d: np.ndarray,
) -> float:
    p, _, product_p, kab = _gaussian_product(
        alpha, center_a, beta, center_b
    )
    q, _, product_q, kcd = _gaussian_product(
        gamma, center_c, delta, center_d
    )
    rpq2 = float(np.dot(product_p - product_q, product_p - product_q))
    rho = p * q / (p + q)
    prefactor = 2.0 * pi**2.5 / (p * q * sqrt(p + q))
    normalization = (
        _primitive_norm(alpha)
        * _primitive_norm(beta)
        * _primitive_norm(gamma)
        * _primitive_norm(delta)
    )
    return normalization * prefactor * kab * kcd * boys0(rho * rpq2)


def _contracted_pair_integrals(
    center_a: np.ndarray,
    center_b: np.ndarray,
    coordinates: np.ndarray,
    charges: np.ndarray,
) -> tuple[float, float, float]:
    overlap = 0.0
    kinetic = 0.0
    attraction = 0.0
    for alpha, coeff_a in zip(H_STO3G_EXPONENTS, H_STO3G_COEFFICIENTS):
        for beta, coeff_b in zip(H_STO3G_EXPONENTS, H_STO3G_COEFFICIENTS):
            weight = float(coeff_a * coeff_b)
            overlap += weight * primitive_overlap(
                alpha, center_a, beta, center_b
            )
            kinetic += weight * primitive_kinetic(
                alpha, center_a, beta, center_b
            )
            for nucleus, charge in zip(coordinates, charges):
                attraction += weight * primitive_nuclear_attraction(
                    alpha,
                    center_a,
                    beta,
                    center_b,
                    nucleus,
                    float(charge),
                )
    return overlap, kinetic, attraction


def _contracted_eri(
    center_a: np.ndarray,
    center_b: np.ndarray,
    center_c: np.ndarray,
    center_d: np.ndarray,
) -> float:
    value = 0.0
    for alpha, coeff_a in zip(H_STO3G_EXPONENTS, H_STO3G_COEFFICIENTS):
        for beta, coeff_b in zip(H_STO3G_EXPONENTS, H_STO3G_COEFFICIENTS):
            for gamma, coeff_c in zip(
                H_STO3G_EXPONENTS, H_STO3G_COEFFICIENTS
            ):
                for delta, coeff_d in zip(
                    H_STO3G_EXPONENTS, H_STO3G_COEFFICIENTS
                ):
                    value += (
                        float(coeff_a * coeff_b * coeff_c * coeff_d)
                        * primitive_eri(
                            alpha,
                            center_a,
                            beta,
                            center_b,
                            gamma,
                            center_c,
                            delta,
                            center_d,
                        )
                    )
    return value


def nuclear_repulsion(
    coordinates_bohr: np.ndarray, charges: Iterable[float]
) -> float:
    charges_array = np.asarray(list(charges), dtype=float)
    value = 0.0
    for atom_a in range(len(charges_array)):
        for atom_b in range(atom_a):
            distance = np.linalg.norm(
                coordinates_bohr[atom_a] - coordinates_bohr[atom_b]
            )
            value += charges_array[atom_a] * charges_array[atom_b] / distance
    return float(value)


def build_sto3g_integrals(
    coordinates_angstrom: np.ndarray,
    charges: Iterable[float] | None = None,
) -> IntegralBundle:
    """Build AO integrals in chemist notation (pq|rs)."""
    coordinates_angstrom = np.asarray(coordinates_angstrom, dtype=float)
    n_basis = coordinates_angstrom.shape[0]
    if charges is None:
        charges_array = np.ones(n_basis, dtype=float)
    else:
        charges_array = np.asarray(list(charges), dtype=float)
    if len(charges_array) != n_basis:
        raise ValueError("One charge is required for every hydrogen center.")

    coordinates = coordinates_angstrom * ANGSTROM_TO_BOHR
    overlap = np.zeros((n_basis, n_basis))
    kinetic = np.zeros_like(overlap)
    attraction = np.zeros_like(overlap)

    for mu in range(n_basis):
        for nu in range(mu + 1):
            s_mn, t_mn, v_mn = _contracted_pair_integrals(
                coordinates[mu],
                coordinates[nu],
                coordinates,
                charges_array,
            )
            overlap[mu, nu] = overlap[nu, mu] = s_mn
            kinetic[mu, nu] = kinetic[nu, mu] = t_mn
            attraction[mu, nu] = attraction[nu, mu] = v_mn

    eri = np.zeros((n_basis, n_basis, n_basis, n_basis))
    # Eightfold permutational symmetry is used only to avoid repeated work.
    for mu in range(n_basis):
        for nu in range(mu + 1):
            pair_mn = mu * (mu + 1) // 2 + nu
            for lam in range(n_basis):
                for sig in range(lam + 1):
                    pair_ls = lam * (lam + 1) // 2 + sig
                    if pair_ls > pair_mn:
                        continue
                    value = _contracted_eri(
                        coordinates[mu],
                        coordinates[nu],
                        coordinates[lam],
                        coordinates[sig],
                    )
                    for a, b, c, d in {
                        (mu, nu, lam, sig),
                        (nu, mu, lam, sig),
                        (mu, nu, sig, lam),
                        (nu, mu, sig, lam),
                        (lam, sig, mu, nu),
                        (sig, lam, mu, nu),
                        (lam, sig, nu, mu),
                        (sig, lam, nu, mu),
                    }:
                        eri[a, b, c, d] = value

    h_core = kinetic + attraction
    return IntegralBundle(
        overlap=overlap,
        kinetic=kinetic,
        nuclear_attraction=attraction,
        h_core=h_core,
        eri=eri,
        nuclear_repulsion=nuclear_repulsion(coordinates, charges_array),
        coordinates_bohr=coordinates,
        charges=charges_array,
    )
