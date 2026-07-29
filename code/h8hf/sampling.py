"""Exact Slater probabilities, finite-shot samples, and bootstrap analysis."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

import numpy as np

from .data_processing import (
    analyze_gamma,
    reconstruct_gamma_from_weights,
)


def occupation_bitstring(
    occupied_modes: tuple[int, ...], n_modes: int, bit_order: str
) -> str:
    occupations = [0] * n_modes
    for mode in occupied_modes:
        occupations[mode] = 1
    if bit_order == "q[n-1]...q[0]":
        return "".join(str(occupations[mode]) for mode in reversed(range(n_modes)))
    if bit_order == "q[0]...q[n-1]":
        return "".join(str(value) for value in occupations)
    raise ValueError(f"Unsupported bit order: {bit_order}")


def slater_probabilities(
    occupied_orbitals: np.ndarray,
    bit_order: str = "q[n-1]...q[0]",
) -> dict[str, float]:
    """Born probabilities |det(C_S)|^2 for an N-particle determinant."""
    n_modes, n_particles = occupied_orbitals.shape
    probabilities: dict[str, float] = {}
    for occupied_modes in combinations(range(n_modes), n_particles):
        amplitude = np.linalg.det(occupied_orbitals[list(occupied_modes), :])
        probability = float(abs(amplitude) ** 2)
        bitstring = occupation_bitstring(
            occupied_modes, n_modes, bit_order
        )
        probabilities[bitstring] = probability
    normalization = sum(probabilities.values())
    return {
        bitstring: probability / normalization
        for bitstring, probability in probabilities.items()
        if probability > 1.0e-18
    }


def multinomial_counts(
    probabilities: Mapping[str, float],
    shots: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    bitstrings = sorted(probabilities)
    values = np.array([probabilities[key] for key in bitstrings], dtype=float)
    values /= np.sum(values)
    sampled = rng.multinomial(shots, values)
    return {
        bitstring: int(count)
        for bitstring, count in zip(bitstrings, sampled)
        if count
    }


def bootstrap_energies(
    observed_counts: Mapping[str, Mapping[str, float]],
    manifest: Mapping[str, Any],
    reference_gamma: np.ndarray,
    h_one_body: np.ndarray,
    eri: np.ndarray,
    nuclear_repulsion: float,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    prepared: dict[str, tuple[list[str], np.ndarray, int]] = {}
    for setting_id, counts in observed_counts.items():
        bitstrings = sorted(counts)
        values = np.array([counts[key] for key in bitstrings], dtype=float)
        shots = int(round(float(np.sum(values))))
        probabilities = values / np.sum(values)
        prepared[setting_id] = (bitstrings, probabilities, shots)

    raw_energies = np.zeros(samples)
    projected_energies = np.zeros(samples)
    fidelities = np.zeros(samples)
    for sample in range(samples):
        resampled: dict[str, dict[str, int]] = {}
        for setting_id, (bitstrings, probabilities, shots) in prepared.items():
            values = rng.multinomial(shots, probabilities)
            resampled[setting_id] = {
                key: int(value)
                for key, value in zip(bitstrings, values)
                if value
            }
        gamma, _ = reconstruct_gamma_from_weights(resampled, manifest)
        analysis = analyze_gamma(
            gamma,
            reference_gamma,
            h_one_body,
            eri,
            nuclear_repulsion,
            int(manifest["system"]["n_particles_one_spin"]),
        )
        raw_energies[sample] = analysis["energy_raw"]["total"]
        projected_energies[sample] = analysis["energy_projected"]["total"]
        fidelities[sample] = analysis["slater_fidelity_projected"]

    def summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "percentile_2p5": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "percentile_97p5": float(np.quantile(values, 0.975)),
            "samples": values.tolist(),
        }

    return {
        "bootstrap_samples": samples,
        "seed": seed,
        "raw_total_energy_hartree": summary(raw_energies),
        "projected_total_energy_hartree": summary(projected_energies),
        "projected_slater_fidelity": summary(fidelities),
    }


def plan_shots(
    exact_probabilities: Mapping[str, Mapping[str, float]],
    manifest: Mapping[str, Any],
    reference_gamma: np.ndarray,
    h_one_body: np.ndarray,
    eri: np.ndarray,
    nuclear_repulsion: float,
    *,
    shot_levels: list[int],
    repetitions: int,
    seed: int,
    chemical_accuracy_millihartree: float = 1.594,
) -> list[dict[str, Any]]:
    """Independent ideal-statistics trials for practical shot planning."""
    rng = np.random.default_rng(seed)
    reference_energy = analyze_gamma(
        reference_gamma,
        reference_gamma,
        h_one_body,
        eri,
        nuclear_repulsion,
        int(manifest["system"]["n_particles_one_spin"]),
    )["energy_reference"]["total"]
    rows: list[dict[str, Any]] = []
    for shots in shot_levels:
        errors = np.zeros(repetitions)
        fidelities = np.zeros(repetitions)
        for repetition in range(repetitions):
            sampled = {
                setting_id: multinomial_counts(probabilities, shots, rng)
                for setting_id, probabilities in exact_probabilities.items()
            }
            gamma, _ = reconstruct_gamma_from_weights(sampled, manifest)
            analysis = analyze_gamma(
                gamma,
                reference_gamma,
                h_one_body,
                eri,
                nuclear_repulsion,
                int(manifest["system"]["n_particles_one_spin"]),
            )
            errors[repetition] = 1000.0 * (
                analysis["energy_projected"]["total"] - reference_energy
            )
            fidelities[repetition] = analysis["slater_fidelity_projected"]
        absolute = np.abs(errors)
        rows.append(
            {
                "shots_per_setting": shots,
                "settings": len(exact_probabilities),
                "total_shots": shots * len(exact_probabilities),
                "repetitions": repetitions,
                "seed": seed,
                "mean_error_millihartree": float(np.mean(errors)),
                "median_absolute_error_millihartree": float(
                    np.median(absolute)
                ),
                "rms_error_millihartree": float(
                    np.sqrt(np.mean(errors**2))
                ),
                "percentile95_absolute_error_millihartree": float(
                    np.quantile(absolute, 0.95)
                ),
                "fraction_within_chemical_accuracy": float(
                    np.mean(absolute <= chemical_accuracy_millihartree)
                ),
                "mean_projected_slater_fidelity": float(
                    np.mean(fidelities)
                ),
            }
        )
    return rows
