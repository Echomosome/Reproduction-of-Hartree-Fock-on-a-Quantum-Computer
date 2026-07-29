#!/usr/bin/env python3
"""Core mathematics and I/O for the diazene RHF quantum reproduction.

The project follows the single-spin, frozen-core representation used for the
diazene experiment in arXiv:2004.04174:

* full STO-3G model: 12 orthonormal spatial modes, 8 electrons per spin;
* two frozen doubly occupied orbitals;
* active quantum model: 10 modes, 6 particles per spin;
* closed-shell alpha and beta density matrices are identical;
* bitstrings are q[9]...q[0] unless explicitly overridden;
* RZ parameters are radians;
* SQISWAP uses the +i one-particle block

      1/sqrt(2) [[1, i],
                 [i, 1]].

The one-spin density convention is gamma[p,q] = <a_q^dagger a_p>.
All orbitals and density matrices are real in this project.
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


N_FULL_MODES = 12
N_FULL_PARTICLES_PER_SPIN = 8
N_FROZEN_ORBITALS = 2
N_MODES = N_FULL_MODES - N_FROZEN_ORBITALS
N_PARTICLES_PER_SPIN = (
    N_FULL_PARTICLES_PER_SPIN - N_FROZEN_ORBITALS
)
DEFAULT_BIT_ORDER = "q9q8q7q6q5q4q3q2q1q0"


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


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_geometries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometries = payload.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise ValueError(f"{path}: no non-empty 'geometries' list.")
    required = {"id", "pathway", "reaction_coordinate_deg", "atoms"}
    seen: set[str] = set()
    for geometry in geometries:
        missing = required - set(geometry)
        if missing:
            raise ValueError(
                f"Geometry is missing keys {sorted(missing)}: {geometry}."
            )
        if geometry["id"] in seen:
            raise ValueError(f"Duplicate geometry id {geometry['id']!r}.")
        seen.add(geometry["id"])
        atoms = geometry["atoms"]
        if len(atoms) != 4:
            raise ValueError(f"{geometry['id']}: expected four atoms.")
        if [atom["element"] for atom in atoms] != ["H", "N", "N", "H"]:
            raise ValueError(
                f"{geometry['id']}: atom order must be H,N,N,H."
            )
        for atom in atoms:
            xyz = atom.get("xyz_angstrom")
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise ValueError(
                    f"{geometry['id']}: invalid xyz for {atom}."
                )
    return geometries


def geometry_by_id(geometries: list[dict], geometry_id: str) -> dict:
    matches = [entry for entry in geometries if entry["id"] == geometry_id]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one geometry with id {geometry_id!r}, "
            f"found {len(matches)}."
        )
    return matches[0]


def cartesian_array(geometry: dict) -> np.ndarray:
    return np.asarray(
        [atom["xyz_angstrom"] for atom in geometry["atoms"]],
        dtype=float,
    )


def distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def bond_angle(
    first: np.ndarray, vertex: np.ndarray, third: np.ndarray
) -> float:
    left = np.asarray(first, dtype=float) - np.asarray(vertex, dtype=float)
    right = np.asarray(third, dtype=float) - np.asarray(vertex, dtype=float)
    cosine = float(
        np.dot(left, right)
        / (np.linalg.norm(left) * np.linalg.norm(right))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def signed_dihedral(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
) -> float:
    """Return the signed first-second-third-fourth dihedral in degrees."""
    p0, p1, p2, p3 = [
        np.asarray(point, dtype=float)
        for point in (first, second, third, fourth)
    ]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def geometry_metrics(geometry: dict) -> dict[str, float]:
    """Recompute all chemically meaningful internal coordinates."""
    xyz = cartesian_array(geometry)
    dihedral_signed = signed_dihedral(xyz[0], xyz[1], xyz[2], xyz[3])
    return {
        "r_h1_n1_angstrom": distance(xyz[0], xyz[1]),
        "r_n1_n2_angstrom": distance(xyz[1], xyz[2]),
        "r_n2_h2_angstrom": distance(xyz[2], xyz[3]),
        "angle_h1_n1_n2_deg": bond_angle(xyz[0], xyz[1], xyz[2]),
        "angle_n1_n2_h2_deg": bond_angle(xyz[1], xyz[2], xyz[3]),
        "dihedral_h1_n1_n2_h2_signed_deg": dihedral_signed,
        "dihedral_h1_n1_n2_h2_0_to_360_deg": (
            dihedral_signed % 360.0
        ),
    }


def rhf_energy_components(
    gamma_one_spin: np.ndarray,
    h1: np.ndarray,
    eri: np.ndarray,
    constant_offset: float,
) -> dict[str, float]:
    """Evaluate a closed-shell RHF functional in an orthonormal basis.

    ``constant_offset`` contains nuclear repulsion and, for the active-space
    model, the frozen-core energy.  ``h1`` is therefore the corresponding
    effective one-electron tensor.
    """
    gamma = np.asarray(gamma_one_spin, dtype=float)
    h1 = np.asarray(h1, dtype=float)
    eri = np.asarray(eri, dtype=float)
    one_body = 2.0 * np.einsum("pq,qp->", h1, gamma, optimize=True)
    coulomb = 2.0 * np.einsum(
        "pq,rs,pqrs->", gamma, gamma, eri, optimize=True
    )
    exchange = np.einsum(
        "pq,rs,prqs->", gamma, gamma, eri, optimize=True
    )
    electronic_active = float(one_body + coulomb - exchange)
    return {
        "constant_offset_hartree": float(constant_offset),
        "one_electron_hartree": float(one_body),
        "coulomb_hartree": float(coulomb),
        "exchange_hartree": float(exchange),
        "active_electronic_hartree": electronic_active,
        "total_hartree": float(constant_offset + electronic_active),
    }


def active_fock(
    gamma_one_spin: np.ndarray, h1: np.ndarray, eri: np.ndarray
) -> np.ndarray:
    """Return F = h + 2J - K for a one-spin RHF density matrix."""
    gamma = np.asarray(gamma_one_spin, dtype=float)
    coulomb = np.einsum(
        "rs,pqrs->pq", gamma, eri, optimize=True
    )
    exchange = np.einsum(
        "rs,prqs->pq", gamma, eri, optimize=True
    )
    return np.asarray(h1, dtype=float) + 2.0 * coulomb - exchange


def real_givens_matrix_elements(
    left: float, right: float, zero: str = "right"
) -> np.ndarray:
    """Return the real Givens matrix used by OpenFermion's QR scheme."""
    tolerance = 1e-15
    a = float(left)
    b = float(right)
    if abs(a) < tolerance:
        cosine, sine, phase = 1.0, 0.0, 1.0
    elif abs(b) < tolerance:
        cosine, sine, phase = 0.0, 1.0, 1.0
    else:
        denominator = math.hypot(a, b)
        cosine = abs(b) / denominator
        sine = abs(a) / denominator
        phase = math.copysign(1.0, a * b)
    if zero == "left":
        return np.asarray(
            [[cosine, -phase * sine], [phase * sine, cosine]],
            dtype=float,
        )
    if zero == "right":
        return np.asarray(
            [[sine, phase * cosine], [-phase * cosine, sine]],
            dtype=float,
        )
    raise ValueError("zero must be 'left' or 'right'.")


def rotate_rows(
    matrix: np.ndarray, rotation: np.ndarray, first: int, second: int
) -> None:
    row_first = matrix[first].copy()
    row_second = matrix[second].copy()
    matrix[first] = rotation[0, 0] * row_first + rotation[0, 1] * row_second
    matrix[second] = (
        rotation[1, 0] * row_first + rotation[1, 1] * row_second
    )


def rotate_columns(
    matrix: np.ndarray, rotation: np.ndarray, first: int, second: int
) -> None:
    column_first = matrix[:, first].copy()
    column_second = matrix[:, second].copy()
    matrix[:, first] = (
        rotation[0, 0] * column_first
        + rotation[0, 1] * column_second
    )
    matrix[:, second] = (
        rotation[1, 0] * column_first
        + rotation[1, 1] * column_second
    )


def slater_givens_decomposition(
    occupied_orbitals: np.ndarray,
    tolerance: float = 1e-14,
) -> list[list[tuple[int, int, float]]]:
    """Compile an occupied subspace into the optimal adjacent Givens network.

    ``occupied_orbitals`` has shape (n_modes, n_particles), with orthonormal
    columns.  The returned nested list is in circuit time order; rotations in
    one inner list act on disjoint pairs and may run in parallel.  Applying

        G(theta) = [[cos(theta), -sin(theta)],
                    [sin(theta),  cos(theta)]]

    to the first ``n_particles`` occupied computational modes prepares the
    target projector.  The algorithm is the real specialization of
    OpenFermion's ``givens_decomposition``.
    """
    orbitals = np.asarray(occupied_orbitals, dtype=float)
    if orbitals.ndim != 2:
        raise ValueError("occupied_orbitals must be a matrix.")
    n_modes, n_particles = orbitals.shape
    if n_particles > n_modes:
        raise ValueError("n_particles cannot exceed n_modes.")
    orthogonality_error = np.linalg.norm(
        orbitals.T @ orbitals - np.eye(n_particles)
    )
    if orthogonality_error > 1e-9:
        raise ValueError(
            "Occupied columns are not orthonormal; "
            f"error={orthogonality_error:.3e}."
        )

    # OpenFermion's convention uses occupied orbitals as orthonormal rows.
    current = orbitals.T.copy()
    m, n = current.shape

    # Occupied-occupied rotations are a gauge transformation of a Slater
    # determinant.  Remove them on the left so they need not enter the circuit.
    for column in reversed(range(n - m + 1, n)):
        for row in range(m - n + column):
            if abs(current[row, column]) <= tolerance:
                continue
            rotation = real_givens_matrix_elements(
                current[row, column],
                current[row + 1, column],
                zero="left",
            )
            rotate_rows(current, rotation, row, row + 1)

    elimination_layers: list[list[tuple[int, int, float]]] = []
    max_parallel = min(m, n - m)
    for layer_index in range(n - 1):
        if layer_index < max_parallel - 1:
            start_row = 0
            end_row = layer_index + 1
            start_column = n - m - layer_index
            end_column = start_column + 2 * (layer_index + 1)
        elif layer_index > n - 1 - max_parallel:
            count = n - 1 - layer_index
            start_row = m - count
            end_row = m
            start_column = m - count + 1
            end_column = start_column + 2 * count
        elif max_parallel == m:
            start_row = 0
            end_row = m
            start_column = n - m - layer_index
            end_column = start_column + 2 * m
        else:
            start_row = layer_index + 1 - max_parallel
            end_row = layer_index + 1
            start_column = start_row + 1
            end_column = start_column + 2 * max_parallel

        operations: list[tuple[int, int, float]] = []
        for row, column in zip(
            range(start_row, end_row),
            range(start_column, end_column, 2),
        ):
            right_element = float(current[row, column])
            if abs(right_element) <= tolerance:
                continue
            left_element = float(current[row, column - 1])
            rotation = real_givens_matrix_elements(
                left_element, right_element, zero="right"
            )
            elimination_theta = math.asin(
                float(np.clip(rotation[1, 0], -1.0, 1.0))
            )
            # Preparation reverses the QR eliminations.  Our G(theta)
            # convention is the inverse of OpenFermion's state gate.
            preparation_theta = -elimination_theta
            operations.append(
                (column - 1, column, preparation_theta)
            )
            rotate_columns(
                current, rotation, column - 1, column
            )
        if operations:
            elimination_layers.append(operations)

    residual = current.copy()
    residual[:, :m] -= np.diag(np.diag(residual[:, :m]))
    if np.max(np.abs(residual)) > 1e-8:
        raise RuntimeError(
            "Givens elimination residual is too large: "
            f"{np.max(np.abs(residual)):.3e}."
        )
    return list(reversed(elimination_layers))


def native_givens_lines(
    first: int, second: int, theta: float
) -> list[str]:
    """Compile a real Givens rotation into +i SQISWAP and RZ gates."""
    if second != first + 1:
        raise ValueError("Native compiler requires adjacent modes.")
    return [
        f"SQISWAP q[{second}],q[{first}]",
        f"RZ q[{first}],({math.pi + theta:.15f})",
        f"RZ q[{second}],({-theta:.15f})",
        f"SQISWAP q[{second}],q[{first}]",
        f"RZ q[{first}],({math.pi:.15f})",
    ]


def complete_graph_matchings(
    n_modes: int = N_MODES,
) -> list[list[tuple[int, int]]]:
    """Return an edge-coloring of K_n for even n using n-1 matchings."""
    if n_modes < 2 or n_modes % 2:
        raise ValueError("n_modes must be an even integer >= 2.")
    players = list(range(n_modes))
    matchings: list[list[tuple[int, int]]] = []
    for _ in range(n_modes - 1):
        pairs = [
            tuple(sorted((players[index], players[-1 - index])))
            for index in range(n_modes // 2)
        ]
        matchings.append(pairs)
        players = [players[0], players[-1], *players[1:-1]]
    edges = [pair for matching in matchings for pair in matching]
    expected = {
        tuple(pair) for pair in combinations(range(n_modes), 2)
    }
    if set(edges) != expected or len(edges) != len(expected):
        raise RuntimeError("Round-robin matching construction failed.")
    return matchings


def analyzer_matrix(
    logical_pairs: list[tuple[int, int]],
    n_modes: int = N_MODES,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Map each logical pair to symmetric/antisymmetric output modes."""
    if len(logical_pairs) * 2 != n_modes:
        raise ValueError("Analyzer requires a perfect matching.")
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
        raise RuntimeError("Analyzer is not orthogonal.")
    return analyzer, measurement_map


def parse_native_circuit(text: str) -> list[Gate]:
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
        if gate.name != "X":
            continue
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
            operation[np.ix_([first, second], [first, second])] = (
                sqrt_iswap
            )
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
    descending = "".join(f"q{index}" for index in reversed(range(n_modes)))
    ascending = "".join(f"q{index}" for index in range(n_modes))
    if bit_order == descending:
        return "".join(reversed(bits_by_mode))
    if bit_order == ascending:
        return "".join(bits_by_mode)
    raise ValueError(f"Unsupported bit order {bit_order!r}.")


def validate_bitstring(bitstring: str, n_bits: int = N_MODES) -> None:
    if len(bitstring) != n_bits or set(bitstring) - {"0", "1"}:
        raise ValueError(
            f"Expected a {n_bits}-character binary string, "
            f"got {bitstring!r}."
        )


def occupied_modes_from_bitstring(
    bitstring: str,
    bit_order: str = DEFAULT_BIT_ORDER,
    n_modes: int = N_MODES,
) -> list[int]:
    validate_bitstring(bitstring, n_modes)
    descending = "".join(f"q{index}" for index in reversed(range(n_modes)))
    ascending = "".join(f"q{index}" for index in range(n_modes))
    if bit_order == descending:
        bits_by_mode = list(reversed(bitstring))
    elif bit_order == ascending:
        bits_by_mode = list(bitstring)
    else:
        raise ValueError(f"Unsupported bit order {bit_order!r}.")
    return [
        index for index, bit in enumerate(bits_by_mode) if bit == "1"
    ]


def normalize_probabilities(
    probabilities: dict[str, float],
) -> dict[str, float]:
    total = float(sum(probabilities.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Distribution has a non-positive total.")
    return {
        key: float(value) / total for key, value in probabilities.items()
    }


def slater_probabilities_from_circuit(
    gates: Iterable[Gate],
    n_modes: int = N_MODES,
    bit_order: str = DEFAULT_BIT_ORDER,
) -> dict[str, float]:
    gate_list = list(gates)
    occupied_input = occupied_modes_from_x_gates(gate_list, n_modes)
    if len(occupied_input) != N_PARTICLES_PER_SPIN:
        raise ValueError(
            f"Expected {N_PARTICLES_PER_SPIN} initialized particles."
        )
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
    shots = int(metadata_shots) if metadata_shots is not None else None
    all_integer = all(
        abs(value - round(value)) <= 1e-9 for value in converted.values()
    )
    if hint == "counts" or (
        hint == "auto" and raw_total > 1.5 and all_integer
    ):
        value_type = "counts"
        if shots is None:
            shots = int(round(raw_total))
    else:
        value_type = "probabilities"
        if shots is None:
            source_label = str(payload.get("source", "")).lower()
            if "exact" not in source_label:
                shots = infer_shots_from_probability_grid(
                    converted.values()
                )
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
        raise ValueError("No probability remains after postselection.")
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
    gamma = np.zeros((N_MODES, N_MODES), dtype=float)
    filled = np.zeros((N_MODES, N_MODES), dtype=bool)
    for circuit in manifest["circuits"]:
        setting = circuit["setting"]
        if setting not in distributions:
            raise KeyError(f"Missing distribution for {setting!r}.")
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
            raise ValueError(f"Unknown kind {circuit['kind']!r}.")
    if not np.all(filled):
        missing = np.argwhere(~filled).tolist()
        raise RuntimeError(f"Incomplete 1-RDM reconstruction: {missing}.")
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
    result: dict[str, float | list[float]] = {
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
        result["frobenius_error"] = float(np.linalg.norm(difference))
        result["max_abs_error"] = float(np.max(np.abs(difference)))
    return result


def slater_subspace_fidelity(
    projected: np.ndarray, target: np.ndarray, rank: int
) -> float:
    _, trial_vectors = np.linalg.eigh(np.asarray(projected, dtype=float))
    _, target_vectors = np.linalg.eigh(np.asarray(target, dtype=float))
    overlap = target_vectors[:, -rank:].T @ trial_vectors[:, -rank:]
    return float(abs(np.linalg.det(overlap)) ** 2)


def total_variation_distance(
    first: dict[str, float], second: dict[str, float]
) -> float:
    keys = set(first) | set(second)
    return 0.5 * sum(
        abs(first.get(key, 0.0) - second.get(key, 0.0))
        for key in keys
    )


def load_reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def parse_gaussian_scf_energies(text: str) -> list[float]:
    pattern = re.compile(
        r"SCF Done:\s+E\([^)]+\)\s*=\s*"
        r"([-+]?\d+\.\d+(?:[DEde][-+]?\d+)?)"
    )
    return [
        float(match.group(1).replace("D", "E").replace("d", "e"))
        for match in pattern.finditer(text)
    ]
