#!/usr/bin/env python3
"""Core mathematics and I/O for the linear-H4 RHF quantum reproduction.

Conventions
-----------

* Four orthonormal spatial orbitals are encoded in q[0] ... q[3].
* One spin sector contains two electrons.  Closed-shell RHF duplicates the
  same spatial 1-RDM for alpha and beta spin.
* Returned bitstrings are q[3]q[2]q[1]q[0] by default.
* RZ parameters are radians.
* SQISWAP has the +i one-particle block

      1/sqrt(2) [[1, i],
                 [i, 1]].

The density matrix convention is D[p,q] = <a_q^dagger a_p>.  All molecular
orbitals and target density matrices in this project are real.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np


N_MODES = 4
N_PARTICLES_PER_SPIN = 2
DEFAULT_BIT_ORDER = "q3q2q1q0"


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    angle: float | None = None


@dataclass(frozen=True)
class Distribution:
    probabilities: dict[str, float]
    shots: int | None
    value_type: str
    raw_total: float
    payload: dict


def symmetric_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric inverse square root of a positive matrix."""
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    if np.any(values <= 0):
        raise ValueError("Matrix must be positive definite.")
    return (vectors * values**-0.5) @ vectors.T


def rhf_energy_components(
    gamma_one_spin: np.ndarray,
    h1_orth: np.ndarray,
    eri_orth: np.ndarray,
    nuclear_repulsion: float,
) -> dict[str, float]:
    """Evaluate the closed-shell RHF functional in an orthonormal basis."""
    gamma = np.asarray(gamma_one_spin, dtype=float)
    h1 = np.asarray(h1_orth, dtype=float)
    eri = np.asarray(eri_orth, dtype=float)
    one_body = 2.0 * np.einsum(
        "pq,qp->", h1, gamma, optimize=True
    )
    coulomb = 2.0 * np.einsum(
        "pq,rs,pqrs->", gamma, gamma, eri, optimize=True
    )
    exchange = np.einsum(
        "pq,rs,prqs->", gamma, gamma, eri, optimize=True
    )
    electronic = one_body + coulomb - exchange
    total = float(nuclear_repulsion + electronic)
    return {
        "nuclear_repulsion_hartree": float(nuclear_repulsion),
        "one_electron_hartree": float(one_body),
        "coulomb_hartree": float(coulomb),
        "exchange_hartree": float(exchange),
        "electronic_hartree": float(electronic),
        "total_hartree": total,
    }


def nearest_neighbor_givens_decomposition(
    orthogonal: np.ndarray,
    tolerance: float = 1e-12,
) -> list[tuple[int, int, float]]:
    """Decompose a real orthogonal matrix into adjacent real Givens gates.

    The returned list is in circuit time order.  Applying each

        G(theta) = [[cos(theta), -sin(theta)],
                    [sin(theta),  cos(theta)]]

    to the specified adjacent rows produces ``orthogonal`` up to independent
    input-column signs.  Those signs cannot change a Slater projector.
    """
    matrix = np.asarray(orthogonal, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square matrix.")
    size = matrix.shape[0]
    orthogonality_error = np.linalg.norm(matrix.T @ matrix - np.eye(size))
    if orthogonality_error > 1e-9:
        raise ValueError(
            f"Matrix is not orthogonal; error={orthogonality_error:.3e}."
        )

    work = matrix.copy()
    eliminations: list[tuple[int, int, float]] = []
    for column in range(size):
        for row in range(size - 1, column, -1):
            upper = float(work[row - 1, column])
            lower = float(work[row, column])
            angle = math.atan2(-lower, upper)
            cosine = math.cos(angle)
            sine = math.sin(angle)
            rotation = np.eye(size)
            rotation[np.ix_([row - 1, row], [row - 1, row])] = [
                [cosine, -sine],
                [sine, cosine],
            ]
            work = rotation @ work
            eliminations.append((row - 1, row, angle))

    off_diagonal = work - np.diag(np.diag(work))
    if np.linalg.norm(off_diagonal) > 1e-9:
        raise RuntimeError("Givens elimination did not diagonalize the matrix.")
    if np.max(np.abs(np.abs(np.diag(work)) - 1.0)) > 1e-9:
        raise RuntimeError("Givens elimination produced a non-sign diagonal.")

    circuit = [
        (left, right, -angle)
        for left, right, angle in reversed(eliminations)
        if abs(math.sin(angle)) > tolerance
    ]
    return circuit


def native_givens_lines(
    first: int, second: int, theta: float
) -> list[str]:
    """Compile a real Givens rotation into +i SQISWAP and RZ gates."""
    return [
        f"SQISWAP q[{second}],q[{first}]",
        f"RZ q[{first}],({math.pi + theta:.12f})",
        f"RZ q[{second}],({-theta:.12f})",
        f"SQISWAP q[{second}],q[{first}]",
        f"RZ q[{first}],({math.pi:.12f})",
    ]


def analyzer_matrix(
    logical_pairs: list[tuple[int, int]],
    n_modes: int = N_MODES,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Build a basis analyzer that sends two logical pairs to adjacent rows.

    For each logical pair (i,j), output rows (left,right) are

        (e_i + e_j)/sqrt(2), (-e_i + e_j)/sqrt(2),

    hence <n_left>-<n_right> = 2 Re(gamma_ij).
    """
    if len(logical_pairs) * 2 != n_modes:
        raise ValueError("The analyzer requires a perfect matching.")
    flattened = [mode for pair in logical_pairs for mode in pair]
    if sorted(flattened) != list(range(n_modes)):
        raise ValueError("Logical pairs must partition all modes.")

    analyzer = np.zeros((n_modes, n_modes), dtype=float)
    measurement_map: list[dict[str, int]] = []
    factor = 1.0 / math.sqrt(2.0)
    for pair_index, (logical_i, logical_j) in enumerate(logical_pairs):
        physical_left = 2 * pair_index
        physical_right = physical_left + 1
        analyzer[physical_left, logical_i] = factor
        analyzer[physical_left, logical_j] = factor
        analyzer[physical_right, logical_i] = -factor
        analyzer[physical_right, logical_j] = factor
        measurement_map.append(
            {
                "physical_left": physical_left,
                "physical_right": physical_right,
                "logical_i": logical_i,
                "logical_j": logical_j,
            }
        )
    if np.linalg.norm(analyzer @ analyzer.T - np.eye(n_modes)) > 1e-12:
        raise RuntimeError("Constructed analyzer is not orthogonal.")
    return analyzer, measurement_map


def parse_native_circuit(text: str) -> list[Gate]:
    """Parse the gate-body or full OriginIR syntax used by this project."""
    gates: list[Gate] = []
    x_pattern = re.compile(r"^X\s+q\[(\d+)\]\s*$", re.IGNORECASE)
    rz_pattern = re.compile(
        r"^RZ\s+q\[(\d+)\]\s*,\s*\(([-+0-9.eE]+)\)\s*$",
        re.IGNORECASE,
    )
    sq_pattern = re.compile(
        r"^SQISWAP\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]\s*$",
        re.IGNORECASE,
    )
    ignored = ("QINIT", "CREG", "MEASURE", "#", "//")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.upper().startswith(ignored):
            continue
        if match := x_pattern.match(line):
            gates.append(Gate("X", (int(match.group(1)),)))
        elif match := rz_pattern.match(line):
            gates.append(
                Gate("RZ", (int(match.group(1)),), float(match.group(2)))
            )
        elif match := sq_pattern.match(line):
            gates.append(
                Gate(
                    "SQISWAP",
                    (int(match.group(1)), int(match.group(2))),
                )
            )
        else:
            raise ValueError(
                f"Cannot parse line {line_number}: {raw_line!r}"
            )
    return gates


def occupied_modes_from_x_gates(
    gates: Iterable[Gate], n_modes: int = N_MODES
) -> list[int]:
    occupied: list[int] = []
    for gate in gates:
        if gate.name == "X":
            mode = gate.qubits[0]
            if mode < 0 or mode >= n_modes:
                raise ValueError(f"Invalid qubit index {mode}.")
            occupied.append(mode)
    if len(set(occupied)) != len(occupied):
        raise ValueError("The same input mode is initialized twice.")
    return occupied


def single_particle_unitary(
    gates: Iterable[Gate], n_modes: int = N_MODES
) -> np.ndarray:
    """Simulate all number-conserving gates in one-particle space."""
    unitary = np.eye(n_modes, dtype=complex)
    sqrt_iswap = np.asarray(
        [[1.0, 1.0j], [1.0j, 1.0]], dtype=complex
    ) / math.sqrt(2.0)
    for gate in gates:
        if gate.name == "X":
            continue
        operation = np.eye(n_modes, dtype=complex)
        if gate.name == "RZ":
            if gate.angle is None:
                raise ValueError("RZ gate is missing an angle.")
            operation[gate.qubits[0], gate.qubits[0]] = np.exp(
                1.0j * gate.angle
            )
        elif gate.name == "SQISWAP":
            first, second = gate.qubits
            operation[np.ix_([first, second], [first, second])] = sqrt_iswap
        else:
            raise ValueError(f"Unsupported gate {gate.name!r}.")
        unitary = operation @ unitary
    return unitary


def gamma_from_native_circuit(
    gates: Iterable[Gate], n_modes: int = N_MODES
) -> np.ndarray:
    gate_list = list(gates)
    occupied = occupied_modes_from_x_gates(gate_list, n_modes)
    if len(occupied) != N_PARTICLES_PER_SPIN:
        raise ValueError(
            f"Expected {N_PARTICLES_PER_SPIN} initialized particles."
        )
    unitary = single_particle_unitary(gate_list, n_modes)
    orbitals = unitary[:, occupied]
    return orbitals @ orbitals.conj().T


def bitstring_from_occupied_modes(
    occupied_modes: Iterable[int],
    n_modes: int = N_MODES,
    bit_order: str = DEFAULT_BIT_ORDER,
) -> str:
    bits_by_mode = ["0"] * n_modes
    for mode in occupied_modes:
        bits_by_mode[mode] = "1"
    if bit_order == "q3q2q1q0":
        return "".join(reversed(bits_by_mode))
    if bit_order == "q0q1q2q3":
        return "".join(bits_by_mode)
    raise ValueError(f"Unsupported bit order {bit_order!r}.")


def occupied_modes_from_bitstring(
    bitstring: str,
    bit_order: str = DEFAULT_BIT_ORDER,
) -> list[int]:
    validate_bitstring(bitstring)
    if bit_order == "q3q2q1q0":
        bits_by_mode = list(reversed(bitstring))
    elif bit_order == "q0q1q2q3":
        bits_by_mode = list(bitstring)
    else:
        raise ValueError(f"Unsupported bit order {bit_order!r}.")
    return [index for index, bit in enumerate(bits_by_mode) if bit == "1"]


def slater_probabilities_from_circuit(
    gates: Iterable[Gate],
    n_modes: int = N_MODES,
    bit_order: str = DEFAULT_BIT_ORDER,
) -> dict[str, float]:
    """Return exact Z probabilities using Slater determinant amplitudes."""
    gate_list = list(gates)
    occupied_input = occupied_modes_from_x_gates(gate_list, n_modes)
    if len(occupied_input) != N_PARTICLES_PER_SPIN:
        raise ValueError("The H4 circuit must initialize two particles.")
    unitary = single_particle_unitary(gate_list, n_modes)
    occupied_orbitals = unitary[:, occupied_input]
    probabilities: dict[str, float] = {}
    for output_modes in combinations(range(n_modes), len(occupied_input)):
        amplitude = np.linalg.det(occupied_orbitals[list(output_modes), :])
        bitstring = bitstring_from_occupied_modes(
            output_modes, n_modes, bit_order
        )
        probabilities[bitstring] = float(abs(amplitude) ** 2)
    return normalize_probabilities(probabilities)


def validate_bitstring(bitstring: str, n_bits: int = N_MODES) -> None:
    if len(bitstring) != n_bits or set(bitstring) - {"0", "1"}:
        raise ValueError(
            f"Expected a {n_bits}-character binary string, got {bitstring!r}."
        )


def normalize_probabilities(
    probabilities: dict[str, float],
) -> dict[str, float]:
    total = float(sum(probabilities.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Distribution has a non-positive total.")
    return {key: float(value) / total for key, value in probabilities.items()}


def infer_shots_from_probability_grid(
    values: Iterable[float], maximum: int = 100_000
) -> int | None:
    array = np.asarray([float(value) for value in values if value > 0])
    if not len(array):
        return None
    for candidate in range(1, maximum + 1):
        residual = np.max(
            np.abs(array * candidate - np.rint(array * candidate))
        )
        if residual <= 2.1e-4:
            return candidate
    return None


def load_distribution(path: Path, n_bits: int = N_MODES) -> Distribution:
    """Read counts or probabilities from common quantum-platform JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status")
    if status is not None and str(status).lower() != "completed":
        raise ValueError(f"{path.name}: status is not Completed.")

    if isinstance(payload.get("counts"), dict):
        raw = payload["counts"]
        hint = "counts"
    elif isinstance(payload.get("probabilities"), dict):
        raw = payload["probabilities"]
        hint = "probabilities"
    else:
        result = payload.get("result", payload)
        if isinstance(result, dict) and isinstance(result.get("counts"), dict):
            raw = result["counts"]
            hint = "counts"
        elif (
            isinstance(result, dict)
            and isinstance(result.get("probabilities"), dict)
        ):
            raw = result["probabilities"]
            hint = "probabilities"
        elif (
            isinstance(result, dict)
            and isinstance(result.get("key"), list)
            and isinstance(result.get("value"), list)
        ):
            if len(result["key"]) != len(result["value"]):
                raise ValueError(f"{path.name}: key/value lengths differ.")
            raw = dict(zip(result["key"], result["value"]))
            hint = str(
                result.get("value_type", payload.get("value_type", "auto"))
            ).lower()
        else:
            raise ValueError(
                f"{path.name}: no supported counts/probabilities layout."
            )
    if not raw:
        raise ValueError(f"{path.name}: distribution is empty.")

    converted: dict[str, float] = {}
    for bitstring, value in raw.items():
        validate_bitstring(str(bitstring), n_bits)
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{path.name}: invalid value {value!r}.")
        converted[str(bitstring)] = number
    raw_total = float(sum(converted.values()))
    if raw_total <= 0:
        raise ValueError(f"{path.name}: total is non-positive.")

    metadata_shots = payload.get("shots")
    if metadata_shots is None and isinstance(payload.get("result"), dict):
        metadata_shots = payload["result"].get("shots")
    shots: int | None = (
        int(metadata_shots) if metadata_shots is not None else None
    )

    all_integer = all(
        abs(value - round(value)) <= 1e-9 for value in converted.values()
    )
    if hint == "counts" or (hint == "auto" and raw_total > 1.5 and all_integer):
        value_type = "counts"
        if shots is None:
            shots = int(round(raw_total))
    else:
        value_type = "probabilities"
        if shots is None:
            shots = infer_shots_from_probability_grid(converted.values())

    return Distribution(
        probabilities=normalize_probabilities(converted),
        shots=shots,
        value_type=value_type,
        raw_total=raw_total,
        payload=payload,
    )


def postselect_particle_number(
    probabilities: dict[str, float],
    particle_number: int = N_PARTICLES_PER_SPIN,
) -> tuple[dict[str, float], float]:
    selected = {
        bitstring: probability
        for bitstring, probability in probabilities.items()
        if bitstring.count("1") == particle_number
    }
    pass_probability = float(sum(selected.values()))
    if pass_probability <= 0:
        raise ValueError("No probability remains after particle postselection.")
    return normalize_probabilities(selected), pass_probability


def occupation_expectations(
    probabilities: dict[str, float],
    bit_order: str = DEFAULT_BIT_ORDER,
) -> np.ndarray:
    occupations = np.zeros(N_MODES, dtype=float)
    for bitstring, probability in probabilities.items():
        for mode in occupied_modes_from_bitstring(bitstring, bit_order):
            occupations[mode] += probability
    return occupations


def reconstruct_gamma(
    distributions: dict[str, dict[str, float]],
    manifest: dict,
    bit_order: str = DEFAULT_BIT_ORDER,
) -> np.ndarray:
    """Reconstruct the real symmetric one-spin 1-RDM."""
    gamma = np.zeros((N_MODES, N_MODES), dtype=float)
    filled = np.zeros((N_MODES, N_MODES), dtype=bool)
    for circuit in manifest["circuits"]:
        setting = circuit["setting"]
        if setting not in distributions:
            raise KeyError(f"Missing distribution for setting {setting!r}.")
        occupations = occupation_expectations(
            distributions[setting], bit_order
        )
        if circuit["kind"] == "diagonal":
            for mode in range(N_MODES):
                gamma[mode, mode] = occupations[mode]
                filled[mode, mode] = True
        elif circuit["kind"] == "offdiagonal_matching":
            for mapping in circuit["measurement_map"]:
                left = int(mapping["physical_left"])
                right = int(mapping["physical_right"])
                logical_i = int(mapping["logical_i"])
                logical_j = int(mapping["logical_j"])
                value = 0.5 * (occupations[left] - occupations[right])
                gamma[logical_i, logical_j] = value
                gamma[logical_j, logical_i] = value
                filled[logical_i, logical_j] = True
                filled[logical_j, logical_i] = True
        else:
            raise ValueError(f"Unknown circuit kind {circuit['kind']!r}.")
    if not np.all(filled):
        missing = np.argwhere(~filled).tolist()
        raise RuntimeError(f"1-RDM reconstruction is incomplete: {missing}.")
    return gamma


def rank_projector(matrix: np.ndarray, rank: int) -> np.ndarray:
    hermitian = 0.5 * (
        np.asarray(matrix, dtype=complex)
        + np.asarray(matrix, dtype=complex).conj().T
    )
    _, vectors = np.linalg.eigh(hermitian)
    occupied = vectors[:, -rank:]
    projected = occupied @ occupied.conj().T
    return np.real_if_close(projected).astype(float)


def gamma_metrics(
    matrix: np.ndarray, target: np.ndarray | None = None
) -> dict[str, float | list[float]]:
    gamma = np.asarray(matrix, dtype=float)
    metrics: dict[str, float | list[float]] = {
        "trace": float(np.trace(gamma)),
        "hermiticity_frobenius": float(
            np.linalg.norm(gamma - gamma.T)
        ),
        "idempotency_frobenius": float(
            np.linalg.norm(gamma @ gamma - gamma)
        ),
        "natural_occupations_ascending": [
            float(value) for value in np.linalg.eigvalsh(gamma)
        ],
    }
    if target is not None:
        difference = gamma - np.asarray(target, dtype=float)
        metrics["frobenius_error"] = float(np.linalg.norm(difference))
        metrics["max_abs_error"] = float(np.max(np.abs(difference)))
    return metrics


def slater_subspace_fidelity(
    projected: np.ndarray, target: np.ndarray, rank: int
) -> float:
    _, trial_vectors = np.linalg.eigh(np.asarray(projected, dtype=float))
    _, target_vectors = np.linalg.eigh(np.asarray(target, dtype=float))
    trial_occupied = trial_vectors[:, -rank:]
    target_occupied = target_vectors[:, -rank:]
    overlap = target_occupied.T @ trial_occupied
    return float(abs(np.linalg.det(overlap)) ** 2)


def total_variation_distance(
    first: dict[str, float], second: dict[str, float]
) -> float:
    keys = set(first) | set(second)
    return 0.5 * sum(
        abs(first.get(key, 0.0) - second.get(key, 0.0)) for key in keys
    )


def load_reference(path: Path) -> dict[str, np.ndarray | float | str]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def parse_gaussian_scf_energies(text: str) -> list[float]:
    pattern = re.compile(
        r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+\.\d+(?:[DEde][-+]?\d+)?)"
    )
    return [
        float(match.group(1).replace("D", "E").replace("d", "e"))
        for match in pattern.finditer(text)
    ]


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
