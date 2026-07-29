"""Givens-network construction and X/RZ/SQISWAP compilation."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class GivensGate:
    mode_a: int
    mode_b: int
    theta_radian: float
    layer: int


@dataclass(frozen=True)
class FitResult:
    angles: np.ndarray
    gates: tuple[GivensGate, ...]
    orbital_rotation: np.ndarray
    projector: np.ndarray
    max_projector_error: float
    frobenius_projector_error: float
    optimizer_cost: float
    optimizer_evaluations: int


def givens_matrix(
    n_modes: int, mode_a: int, mode_b: int, theta: float
) -> np.ndarray:
    """Real Givens matrix with block [[cos,-sin],[sin,cos]]."""
    if mode_a == mode_b:
        raise ValueError("A Givens rotation needs two distinct modes.")
    matrix = np.eye(n_modes)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    matrix[mode_a, mode_a] = cosine
    matrix[mode_a, mode_b] = -sine
    matrix[mode_b, mode_a] = sine
    matrix[mode_b, mode_b] = cosine
    return matrix


def diamond_layout(n_modes: int, n_occupied: int) -> list[list[tuple[int, int]]]:
    """Half-filled nearest-neighbor diamond used in the HFVQE experiment."""
    if n_modes != 2 * n_occupied:
        raise ValueError("This transparent diamond generator assumes half filling.")
    layers: list[list[tuple[int, int]]] = []
    for step in range(n_modes - 1):
        width = min(step + 1, n_modes - 1 - step, n_occupied)
        # Equivalent explicit pattern for N=8:
        # (3,4); (2,3)(4,5); ...; (3,4).
        if step < n_occupied:
            start = n_occupied - 1 - step
        else:
            start = step - n_occupied + 1
        pairs = [(start + 2 * index, start + 2 * index + 1) for index in range(width)]
        layers.append(pairs)
    return layers


def flattened_layout(
    n_modes: int, n_occupied: int
) -> list[tuple[int, int, int]]:
    flattened: list[tuple[int, int, int]] = []
    for layer, pairs in enumerate(diamond_layout(n_modes, n_occupied)):
        for mode_a, mode_b in pairs:
            flattened.append((mode_a, mode_b, layer))
    return flattened


def orbital_rotation_from_angles(
    angles: Iterable[float], n_modes: int, n_occupied: int
) -> np.ndarray:
    rotation = np.eye(n_modes)
    layout = flattened_layout(n_modes, n_occupied)
    angle_array = np.asarray(list(angles), dtype=float)
    if len(angle_array) != len(layout):
        raise ValueError(f"Expected {len(layout)} angles, got {len(angle_array)}.")
    # Listed circuit order is chronological; each new gate left-multiplies.
    for theta, (mode_a, mode_b, _) in zip(angle_array, layout):
        rotation = givens_matrix(n_modes, mode_a, mode_b, theta) @ rotation
    return rotation


def projector_from_angles(
    angles: Iterable[float], n_modes: int, n_occupied: int
) -> np.ndarray:
    rotation = orbital_rotation_from_angles(angles, n_modes, n_occupied)
    occupied = rotation[:, :n_occupied]
    return occupied @ occupied.T


def fit_projector_to_diamond(
    target_projector: np.ndarray,
    n_occupied: int,
    *,
    seed: int = 20260729,
    starts: int = 8,
    tolerance: float = 1.0e-11,
) -> FitResult:
    """Obtain every Givens angle by an explicit nonlinear least-squares fit."""
    target = 0.5 * (target_projector + target_projector.T)
    n_modes = target.shape[0]
    layout = flattened_layout(n_modes, n_occupied)
    n_parameters = len(layout)

    # The full symmetric residual makes the objective exactly the Frobenius norm.
    def residual(angles: np.ndarray) -> np.ndarray:
        return (
            projector_from_angles(angles, n_modes, n_occupied) - target
        ).ravel()

    rng = np.random.default_rng(seed)
    initial_guesses = [np.zeros(n_parameters)]
    initial_guesses.extend(
        rng.normal(0.0, 0.35, size=n_parameters) for _ in range(starts - 1)
    )
    best = None
    for initial in initial_guesses:
        candidate = least_squares(
            residual,
            initial,
            method="trf",
            ftol=1.0e-14,
            xtol=1.0e-14,
            gtol=1.0e-14,
            max_nfev=20000,
        )
        if best is None or candidate.cost < best.cost:
            best = candidate
        if np.max(np.abs(residual(candidate.x))) < tolerance:
            break
    assert best is not None

    rotation = orbital_rotation_from_angles(best.x, n_modes, n_occupied)
    projector = rotation[:, :n_occupied] @ rotation[:, :n_occupied].T
    difference = projector - target
    gates = tuple(
        GivensGate(a, b, float(theta), layer)
        for theta, (a, b, layer) in zip(best.x, layout)
    )
    return FitResult(
        angles=np.asarray(best.x),
        gates=gates,
        orbital_rotation=rotation,
        projector=projector,
        max_projector_error=float(np.max(np.abs(difference))),
        frobenius_projector_error=float(np.linalg.norm(difference)),
        optimizer_cost=float(best.cost),
        optimizer_evaluations=int(best.nfev),
    )


def swap_network_matchings(n_modes: int) -> list[list[tuple[int, int]]]:
    """N alternating swap-network layers covering every K_N edge once."""
    order = list(range(n_modes))
    layers: list[list[tuple[int, int]]] = []
    for layer in range(n_modes):
        start = layer % 2
        positions = list(range(start, n_modes - 1, 2))
        pairs = [(order[position], order[position + 1]) for position in positions]
        layers.append(pairs)
        for position in positions:
            order[position], order[position + 1] = (
                order[position + 1],
                order[position],
            )
    unordered = [tuple(sorted(pair)) for layer in layers for pair in layer]
    expected = n_modes * (n_modes - 1) // 2
    if len(unordered) != expected or len(set(unordered)) != expected:
        raise AssertionError("Swap-network matchings did not cover K_N exactly.")
    return layers


def measurement_rotation(
    n_modes: int, pairs: Iterable[tuple[int, int]]
) -> np.ndarray:
    """Apply pi/4 number-conserving analyzers to disjoint mode pairs."""
    result = np.eye(n_modes)
    touched: set[int] = set()
    for mode_a, mode_b in pairs:
        if mode_a in touched or mode_b in touched:
            raise ValueError("Measurement pairs must be disjoint.")
        touched.update((mode_a, mode_b))
        result = givens_matrix(n_modes, mode_a, mode_b, pi / 4.0) @ result
    return result


def native_gate_lines(gates: Iterable[GivensGate], n_occupied: int) -> list[str]:
    """Compile G(theta) to two SQISWAP and three RZ gates.

    Conventions:
      SQISWAP single-excitation block = (I - i X) / sqrt(2)
      RZ(phi) = exp(-i phi Z / 2)

    This is the PyQPanda/OriginQ sign convention.  The listed sequence equals
    -G(theta); the minus sign is a global phase.
    """
    lines = [f"X q[{mode}]" for mode in range(n_occupied)]
    for gate in gates:
        mode_a, mode_b, theta = (
            gate.mode_a,
            gate.mode_b,
            gate.theta_radian,
        )
        lines.extend(
            [
                f"RZ q[{mode_a}],({pi:.15f})",
                f"SQISWAP q[{mode_a}],q[{mode_b}]",
                f"RZ q[{mode_a}],({pi - theta:.15f})",
                f"RZ q[{mode_b}],({theta:.15f})",
                f"SQISWAP q[{mode_a}],q[{mode_b}]",
            ]
        )
    return lines


def abstract_gate_lines(
    gates: Iterable[GivensGate], n_occupied: int
) -> list[str]:
    lines = [f"X q[{mode}]" for mode in range(n_occupied)]
    lines.extend(
        f"GIVENS q[{gate.mode_a}],q[{gate.mode_b}],({gate.theta_radian:.15f})"
        for gate in gates
    )
    return lines


def verify_native_givens(theta: float) -> float:
    """Return max error between the native block and -G(theta)."""
    def rz(angle: float) -> np.ndarray:
        return np.diag(
            [np.exp(-0.5j * angle), np.exp(0.5j * angle)]
        )

    sqiswap = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(2), -1j / np.sqrt(2), 0],
            [0, -1j / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )
    chronological = [
        np.kron(rz(pi), np.eye(2)),
        sqiswap,
        np.kron(rz(pi - theta), rz(theta)),
        sqiswap,
    ]
    total = np.eye(4, dtype=complex)
    for operation in chronological:
        total = operation @ total
    cosine, sine = np.cos(theta), np.sin(theta)
    target = np.array(
        [
            [1, 0, 0, 0],
            [0, cosine, -sine, 0],
            [0, sine, cosine, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )
    return float(np.max(np.abs(total + target)))
