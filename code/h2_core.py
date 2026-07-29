#!/usr/bin/env python3
"""Core mathematics and I/O for the H2 RHF/1-RDM reproduction.

Conventions used throughout the project
---------------------------------------

* Two orthonormal spatial orbitals are encoded in q[0] and q[1].
* One spin sector contains one electron.  RHF duplicates the same spatial
  one-particle density matrix for alpha and beta spin.
* Returned bitstrings are written as ``q[1]q[0]`` by default, so ``"01"``
  means q[0] is occupied.
* Native RZ angles are radians.
* SQISWAP uses the +i single-particle block

      1/sqrt(2) [[1, i],
                 [i, 1]].

The density matrix used by the energy routine is the PySCF convention
``D[p,q] = <a_q^dagger a_p>``.  For the real RHF states studied here this is
identical to the transpose convention often used in physics.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


N_MODES = 2
N_PARTICLES_PER_SPIN = 1


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


def phase_fix_real_vector(vector: np.ndarray) -> np.ndarray:
    """Return a normalized, real vector with a deterministic overall sign."""
    result = np.real_if_close(np.asarray(vector)).astype(float)
    norm = float(np.linalg.norm(result))
    if norm <= 0:
        raise ValueError("Cannot normalize the zero vector.")
    result = result / norm
    largest = int(np.argmax(np.abs(result)))
    if result[largest] < 0:
        result = -result
    return result


def givens_angle_from_orbital(orbital: np.ndarray) -> float:
    """Angle theta such that G(theta)|10> = c0|10> + c1|01>."""
    c = phase_fix_real_vector(orbital)
    if c.shape != (2,):
        raise ValueError("The H2 minimal-basis orbital must have two entries.")
    return float(math.atan2(c[1], c[0]))


def native_givens_lines(
    first: int, second: int, theta: float
) -> list[str]:
    """Compile a real Givens rotation into SQISWAP and RZ gates.

    In the one-particle basis [first, second], this sequence realizes

        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]].
    """
    return [
        f"SQISWAP q[{second}],q[{first}]",
        f"RZ q[{first}],({math.pi + theta:.12f})",
        f"RZ q[{second}],({-theta:.12f})",
        f"SQISWAP q[{second}],q[{first}]",
        f"RZ q[{first}],({math.pi:.12f})",
    ]


def real_coherence_analyzer_lines(first: int, second: int) -> list[str]:
    """Map Re(D[first,second]) to an occupation-number difference."""
    return [
        f"RZ q[{first}],({math.pi / 4:.12f})",
        f"RZ q[{second}],({-math.pi / 4:.12f})",
        f"SQISWAP q[{second}],q[{first}]",
    ]


def parse_native_circuit(text: str) -> list[Gate]:
    """Parse the gate-body or full OriginIR files used in this package."""
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
    ignored_prefixes = ("QINIT", "CREG", "MEASURE", "#", "//")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.upper().startswith(ignored_prefixes):
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


def single_particle_unitary(
    gates: Iterable[Gate], n_modes: int = N_MODES
) -> np.ndarray:
    """Simulate number-conserving gates in the one-particle representation."""
    unitary = np.eye(n_modes, dtype=complex)
    sqrt_iswap = np.asarray(
        [[1.0, 1.0j], [1.0j, 1.0]], dtype=complex
    ) / math.sqrt(2.0)
    for gate in gates:
        if gate.name == "X":
            continue
        operation = np.eye(n_modes, dtype=complex)
        if gate.name == "RZ":
            qubit = gate.qubits[0]
            if gate.angle is None:
                raise ValueError("RZ gate is missing an angle.")
            operation[qubit, qubit] = np.exp(1.0j * gate.angle)
        elif gate.name == "SQISWAP":
            first, second = gate.qubits
            operation[np.ix_([first, second], [first, second])] = sqrt_iswap
        else:
            raise ValueError(f"Unsupported gate {gate.name!r}")
        unitary = operation @ unitary
    return unitary


def occupied_modes_from_x_gates(
    gates: Iterable[Gate], n_modes: int = N_MODES
) -> list[int]:
    """Read the initial determinant encoded by X gates."""
    occupied: list[int] = []
    for gate in gates:
        if gate.name == "X":
            qubit = gate.qubits[0]
            if qubit < 0 or qubit >= n_modes:
                raise ValueError(f"Invalid qubit index {qubit}.")
            occupied.append(qubit)
    if len(set(occupied)) != len(occupied):
        raise ValueError("The same qubit is initialized twice.")
    return occupied


def gamma_from_native_circuit(
    gates: Iterable[Gate], n_modes: int = N_MODES
) -> np.ndarray:
    """Return D[p,q]=<a_q^dagger a_p> after a native Slater circuit."""
    gate_list = list(gates)
    occupied = occupied_modes_from_x_gates(gate_list, n_modes)
    if not occupied:
        raise ValueError("Circuit contains no X gate / occupied input mode.")
    unitary = single_particle_unitary(gate_list, n_modes)
    occupied_orbitals = unitary[:, occupied]
    return occupied_orbitals @ occupied_orbitals.conj().T


def bitstring_from_occupied_modes(
    occupied_modes: Iterable[int], n_modes: int = N_MODES
) -> str:
    """Encode occupations as q[n-1]...q[0]."""
    bits = ["0"] * n_modes
    for mode in occupied_modes:
        bits[n_modes - 1 - mode] = "1"
    return "".join(bits)


def one_particle_probabilities_from_circuit(
    gates: Iterable[Gate], n_modes: int = N_MODES
) -> dict[str, float]:
    """Return exact Z-basis probabilities for a one-particle circuit."""
    gate_list = list(gates)
    occupied = occupied_modes_from_x_gates(gate_list, n_modes)
    if occupied != [0]:
        raise ValueError(
            "The two-qubit H2 primary circuit must initialize only q[0]."
        )
    state = single_particle_unitary(gate_list, n_modes)[:, 0]
    probabilities: dict[str, float] = {}
    for mode, amplitude in enumerate(state):
        probabilities[bitstring_from_occupied_modes([mode], n_modes)] = float(
            abs(amplitude) ** 2
        )
    return normalize_probabilities(probabilities)


def normalize_probabilities(
    probabilities: dict[str, float]
) -> dict[str, float]:
    total = float(sum(probabilities.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Distribution has a non-positive total.")
    return {key: float(value) / total for key, value in probabilities.items()}


def validate_bitstring(bitstring: str, n_bits: int = N_MODES) -> None:
    if len(bitstring) != n_bits or set(bitstring) - {"0", "1"}:
        raise ValueError(
            f"Expected a {n_bits}-character binary string, got {bitstring!r}."
        )


def infer_shots_from_probability_grid(
    values: Iterable[float], maximum: int = 100_000
) -> int | None:
    """Infer a plausible shot count from rounded probabilities."""
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
    """Load counts or probabilities from common platform JSON layouts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status")
    if status is not None and str(status).lower() != "completed":
        raise ValueError(f"{path.name}: status is not Completed.")

    if isinstance(payload.get("counts"), dict):
        raw = payload["counts"]
        value_type_hint = "counts"
    elif isinstance(payload.get("probabilities"), dict):
        raw = payload["probabilities"]
        value_type_hint = "probabilities"
    else:
        result = payload.get("result", payload)
        if isinstance(result, dict) and isinstance(result.get("counts"), dict):
            raw = result["counts"]
            value_type_hint = "counts"
        elif (
            isinstance(result, dict)
            and isinstance(result.get("probabilities"), dict)
        ):
            raw = result["probabilities"]
            value_type_hint = "probabilities"
        elif (
            isinstance(result, dict)
            and isinstance(result.get("key"), list)
            and isinstance(result.get("value"), list)
        ):
            if len(result["key"]) != len(result["value"]):
                raise ValueError(f"{path.name}: key/value lengths differ.")
            raw = dict(zip(result["key"], result["value"]))
            value_type_hint = str(
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

    counts_hint = value_type_hint.startswith("count")
    probability_hint = value_type_hint.startswith("prob")
    if counts_hint or (not probability_hint and raw_total > 1.5):
        rounded = {key: int(round(value)) for key, value in converted.items()}
        if any(abs(rounded[key] - converted[key]) > 1e-6 for key in converted):
            raise ValueError(f"{path.name}: non-integer count encountered.")
        count_total = int(sum(rounded.values()))
        if shots is not None and count_total != shots:
            raise ValueError(
                f"{path.name}: counts total {count_total} != shots {shots}."
            )
        shots = count_total
        probabilities = {
            key: value / count_total for key, value in rounded.items()
        }
        value_type = "counts"
    else:
        if abs(raw_total - 1.0) > 2e-3:
            raise ValueError(
                f"{path.name}: probability total {raw_total:.8f} is not 1."
            )
        probabilities = normalize_probabilities(converted)
        if shots is None:
            shots = infer_shots_from_probability_grid(converted.values())
        value_type = "probabilities"

    return Distribution(
        probabilities=probabilities,
        shots=shots,
        value_type=value_type,
        raw_total=raw_total,
        payload=payload,
    )


def postselect_particle_number(
    probabilities: dict[str, float], n_particles: int = 1
) -> tuple[dict[str, float], float]:
    selected = {
        key: value
        for key, value in probabilities.items()
        if key.count("1") == n_particles
    }
    pass_probability = float(sum(selected.values()))
    if pass_probability <= 0:
        raise ValueError("No shots survive particle-number post-selection.")
    return normalize_probabilities(selected), pass_probability


def occupation(
    probabilities: dict[str, float],
    qubit: int,
    bit_order: str = "q1q0",
) -> float:
    """Expectation value of n_qubit for a two-bit distribution."""
    if bit_order == "q1q0":
        character_index = N_MODES - 1 - qubit
    elif bit_order == "q0q1":
        character_index = qubit
    else:
        raise ValueError("bit_order must be 'q1q0' or 'q0q1'.")
    return float(
        sum(
            probability
            for bitstring, probability in probabilities.items()
            if bitstring[character_index] == "1"
        )
    )


def reconstruct_real_gamma(
    diagonal_probabilities: dict[str, float],
    real_probabilities: dict[str, float],
    bit_order: str = "q1q0",
) -> np.ndarray:
    """Reconstruct the real 2x2 one-spin 1-RDM from two settings."""
    n0 = occupation(diagonal_probabilities, 0, bit_order)
    n1 = occupation(diagonal_probabilities, 1, bit_order)
    real_n0 = occupation(real_probabilities, 0, bit_order)
    real_n1 = occupation(real_probabilities, 1, bit_order)
    coherence = 0.5 * (real_n0 - real_n1)
    return np.asarray([[n0, coherence], [coherence, n1]], dtype=float)


def rank_one_projector(matrix: np.ndarray) -> np.ndarray:
    """Project a Hermitian estimate onto the nearest rank-one projector."""
    hermitian = 0.5 * (
        np.asarray(matrix, dtype=complex)
        + np.asarray(matrix, dtype=complex).conj().T
    )
    values, vectors = np.linalg.eigh(hermitian)
    occupied = vectors[:, int(np.argmax(values))]
    projector = np.outer(occupied, occupied.conj())
    return np.real_if_close(projector).astype(float)


def rhf_energy_components(
    gamma_one_spin: np.ndarray,
    h1_orth: np.ndarray,
    eri_orth: np.ndarray,
    nuclear_repulsion: float,
) -> dict[str, float]:
    """Evaluate the closed-shell RHF energy from a one-spin spatial 1-RDM.

    For a real idempotent D with Tr(D)=N/2,

      E = E_nuc
          + 2 sum_pq h_pq D_pq
          + 2 sum_pqrs D_pq D_rs (pq|rs)
          -   sum_pqrs D_pq D_rs (pr|qs).
    """
    gamma = np.real(
        0.5
        * (
            np.asarray(gamma_one_spin, dtype=complex)
            + np.asarray(gamma_one_spin, dtype=complex).conj().T
        )
    )
    h1 = np.asarray(h1_orth, dtype=float)
    eri = np.asarray(eri_orth, dtype=float)
    one_electron = 2.0 * np.einsum(
        "pq,qp->", h1, gamma, optimize=True
    )
    coulomb = 2.0 * np.einsum(
        "pq,rs,pqrs->", gamma, gamma, eri, optimize=True
    )
    exchange = np.einsum(
        "pq,rs,prqs->", gamma, gamma, eri, optimize=True
    )
    electronic = float(one_electron + coulomb - exchange)
    total = float(nuclear_repulsion + electronic)
    return {
        "nuclear_repulsion_hartree": float(nuclear_repulsion),
        "one_electron_hartree": float(one_electron),
        "coulomb_hartree": float(coulomb),
        "exchange_hartree": float(exchange),
        "electronic_hartree": electronic,
        "total_hartree": total,
    }


def gamma_metrics(
    estimate: np.ndarray, target: np.ndarray | None = None
) -> dict[str, float | list[float]]:
    hermitian = 0.5 * (
        np.asarray(estimate, dtype=complex)
        + np.asarray(estimate, dtype=complex).conj().T
    )
    eigenvalues = np.linalg.eigvalsh(hermitian)
    metrics: dict[str, float | list[float]] = {
        "trace": float(np.trace(hermitian).real),
        "hermiticity_frobenius": float(
            np.linalg.norm(estimate - np.asarray(estimate).conj().T)
        ),
        "idempotency_frobenius": float(
            np.linalg.norm(hermitian @ hermitian - hermitian)
        ),
        "natural_occupations_ascending": [
            float(value) for value in eigenvalues
        ],
    }
    if target is not None:
        target_array = np.asarray(target, dtype=complex)
        metrics["frobenius_error"] = float(
            np.linalg.norm(hermitian - target_array)
        )
        metrics["max_abs_error"] = float(
            np.max(np.abs(hermitian - target_array))
        )
    return metrics


def rank_one_orbital_fidelities(
    projected: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Return one-spin and full closed-shell Slater determinant fidelities."""
    _, projected_vectors = np.linalg.eigh(projected)
    _, target_vectors = np.linalg.eigh(target)
    occupied_projected = projected_vectors[:, -1]
    occupied_target = target_vectors[:, -1]
    one_spin = float(abs(np.vdot(occupied_target, occupied_projected)) ** 2)
    return {
        "one_spin_orbital_fidelity": one_spin,
        "two_electron_rhf_determinant_fidelity": one_spin**2,
    }


def load_reference(path: Path) -> dict[str, np.ndarray | float]:
    archive = np.load(path)
    return {key: archive[key] for key in archive.files}


def parse_gaussian_scf_energies(text: str) -> list[float]:
    """Extract all Gaussian ``SCF Done`` energies from a log."""
    pattern = re.compile(
        r"SCF Done:\s+E\([^)]*\)\s*=\s*"
        r"([-+]?\d+\.\d+(?:[DEde][-+]?\d+)?)"
    )
    return [
        float(match.replace("D", "E").replace("d", "e"))
        for match in pattern.findall(text)
    ]


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
