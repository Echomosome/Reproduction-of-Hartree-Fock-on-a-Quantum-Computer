"""Counts -> post-selection -> 1-RDM -> rank projection -> RHF energy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .rhf import rhf_energy_from_gamma


BITSTRING = re.compile(r"^[01]+$")


@dataclass(frozen=True)
class Distribution:
    probabilities: dict[str, float]
    input_total: float
    accepted_total: float
    acceptance_rate: float


def _find_bitstring_mapping(node: Any, n_modes: int) -> Mapping[str, Any] | None:
    if isinstance(node, Mapping):
        if node and all(
            isinstance(key, str)
            and BITSTRING.fullmatch(key.replace(" ", ""))
            and len(key.replace(" ", "")) == n_modes
            and isinstance(value, (int, float))
            for key, value in node.items()
        ):
            return node
        for preferred in ("counts", "probabilities", "result", "data"):
            if preferred in node:
                found = _find_bitstring_mapping(node[preferred], n_modes)
                if found is not None:
                    return found
        for value in node.values():
            found = _find_bitstring_mapping(value, n_modes)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_bitstring_mapping(value, n_modes)
            if found is not None:
                return found
    return None


def load_distribution(path: str | Path, n_modes: int) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = _find_bitstring_mapping(payload, n_modes)
    if mapping is None:
        raise ValueError(f"No {n_modes}-bit distribution found in {path}.")
    clean = {
        key.replace(" ", ""): float(value)
        for key, value in mapping.items()
        if float(value) >= 0.0
    }
    if not clean or sum(clean.values()) <= 0.0:
        raise ValueError(f"Distribution in {path} has zero total weight.")
    return clean


def postselect_distribution(
    weights: Mapping[str, float], particle_number: int
) -> Distribution:
    total = float(sum(weights.values()))
    accepted = {
        bitstring: float(weight)
        for bitstring, weight in weights.items()
        if bitstring.count("1") == particle_number
    }
    accepted_total = float(sum(accepted.values()))
    if accepted_total <= 0.0:
        raise ValueError("No bitstrings survive particle-number post-selection.")
    probabilities = {
        bitstring: weight / accepted_total
        for bitstring, weight in accepted.items()
    }
    return Distribution(
        probabilities=probabilities,
        input_total=total,
        accepted_total=accepted_total,
        acceptance_rate=accepted_total / total,
    )


def bitstring_to_occupations(
    bitstring: str, n_modes: int, bit_order: str
) -> np.ndarray:
    if bit_order == "q[n-1]...q[0]":
        return np.array(
            [int(bitstring[n_modes - 1 - mode]) for mode in range(n_modes)],
            dtype=float,
        )
    if bit_order == "q[0]...q[n-1]":
        return np.array([int(value) for value in bitstring], dtype=float)
    raise ValueError(f"Unsupported bit order: {bit_order}")


def occupations_from_distribution(
    probabilities: Mapping[str, float],
    n_modes: int,
    bit_order: str,
) -> np.ndarray:
    occupations = np.zeros(n_modes)
    for bitstring, probability in probabilities.items():
        occupations += probability * bitstring_to_occupations(
            bitstring, n_modes, bit_order
        )
    return occupations


def reconstruct_gamma(
    setting_files: Mapping[str, str | Path],
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    weight_maps = {
        setting_id: load_distribution(path, int(manifest["system"]["n_modes"]))
        for setting_id, path in setting_files.items()
    }
    gamma, diagnostics = reconstruct_gamma_from_weights(weight_maps, manifest)
    for setting_id, path in setting_files.items():
        diagnostics[setting_id]["path"] = str(path)
    return gamma, diagnostics


def reconstruct_gamma_from_weights(
    setting_weights: Mapping[str, Mapping[str, float]],
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    n_modes = int(manifest["system"]["n_modes"])
    n_occupied = int(manifest["system"]["n_particles_one_spin"])
    bit_order = str(manifest["measurement"]["bit_order"])
    gamma = np.zeros((n_modes, n_modes))
    diagnostics: dict[str, Any] = {}

    setting_by_id = {
        setting["id"]: setting for setting in manifest["settings"]
    }
    for setting_id, weights in setting_weights.items():
        setting = setting_by_id[setting_id]
        postselected = postselect_distribution(weights, n_occupied)
        occupations = occupations_from_distribution(
            postselected.probabilities, n_modes, bit_order
        )
        diagnostics[setting_id] = {
            "input_total": postselected.input_total,
            "accepted_total": postselected.accepted_total,
            "acceptance_rate": postselected.acceptance_rate,
            "occupations": occupations.tolist(),
        }
        if setting["kind"] == "diagonal":
            np.fill_diagonal(gamma, occupations)
        else:
            for pair in setting["pairs"]:
                mode_a, mode_b = int(pair[0]), int(pair[1])
                value = 0.5 * (occupations[mode_b] - occupations[mode_a])
                gamma[mode_a, mode_b] = value
                gamma[mode_b, mode_a] = value

    expected = {setting["id"] for setting in manifest["settings"]}
    missing = expected - set(setting_weights)
    if missing:
        raise ValueError(f"Missing measurement settings: {sorted(missing)}")
    return 0.5 * (gamma + gamma.T), diagnostics


def rank_n_projector(
    gamma: np.ndarray, particle_number: int
) -> tuple[np.ndarray, np.ndarray]:
    """Closest rank-N orthogonal projector in Frobenius norm."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gamma + gamma.T))
    order = np.argsort(eigenvalues)[::-1]
    occupied_vectors = eigenvectors[:, order[:particle_number]]
    projector = occupied_vectors @ occupied_vectors.T
    return projector, eigenvalues


def mcweeny_purify(
    gamma: np.ndarray, iterations: int = 20
) -> np.ndarray:
    purified = 0.5 * (gamma + gamma.T)
    for _ in range(iterations):
        purified = 3.0 * purified @ purified - 2.0 * purified @ purified @ purified
    return 0.5 * (purified + purified.T)


def slater_fidelity(
    reference_projector: np.ndarray,
    measured_projector: np.ndarray,
    particle_number: int,
) -> float:
    ref_values, ref_vectors = np.linalg.eigh(reference_projector)
    measured_values, measured_vectors = np.linalg.eigh(measured_projector)
    ref_occ = ref_vectors[:, np.argsort(ref_values)[-particle_number:]]
    measured_occ = measured_vectors[
        :, np.argsort(measured_values)[-particle_number:]
    ]
    return float(abs(np.linalg.det(ref_occ.T @ measured_occ)) ** 2)


def analyze_gamma(
    gamma_raw: np.ndarray,
    reference_gamma: np.ndarray,
    h_one_body: np.ndarray,
    eri: np.ndarray,
    nuclear_repulsion: float,
    particle_number: int,
) -> dict[str, Any]:
    projected, raw_eigenvalues = rank_n_projector(
        gamma_raw, particle_number
    )
    raw_energy = rhf_energy_from_gamma(
        gamma_raw, h_one_body, eri, nuclear_repulsion
    )
    projected_energy = rhf_energy_from_gamma(
        projected, h_one_body, eri, nuclear_repulsion
    )
    reference_energy = rhf_energy_from_gamma(
        reference_gamma, h_one_body, eri, nuclear_repulsion
    )
    return {
        "gamma_raw": gamma_raw.tolist(),
        "gamma_projected": projected.tolist(),
        "natural_occupations_raw_ascending": raw_eigenvalues.tolist(),
        "trace_raw": float(np.trace(gamma_raw)),
        "idempotency_error_raw_frobenius": float(
            np.linalg.norm(gamma_raw @ gamma_raw - gamma_raw)
        ),
        "max_gamma_error_raw": float(
            np.max(np.abs(gamma_raw - reference_gamma))
        ),
        "max_gamma_error_projected": float(
            np.max(np.abs(projected - reference_gamma))
        ),
        "slater_fidelity_projected": slater_fidelity(
            reference_gamma, projected, particle_number
        ),
        "energy_raw": raw_energy,
        "energy_projected": projected_energy,
        "energy_reference": reference_energy,
        "energy_error_raw_hartree": float(
            raw_energy["total"] - reference_energy["total"]
        ),
        "energy_error_projected_hartree": float(
            projected_energy["total"] - reference_energy["total"]
        ),
    }
