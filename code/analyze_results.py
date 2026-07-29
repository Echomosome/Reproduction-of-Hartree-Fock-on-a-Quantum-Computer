#!/usr/bin/env python3
"""Reconstruct the H2 one-spin 1-RDM and RHF energy from two JSON settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from h2_core import (
    gamma_metrics,
    load_distribution,
    load_reference,
    parse_gaussian_scf_energies,
    postselect_particle_number,
    rank_one_orbital_fidelities,
    rank_one_projector,
    reconstruct_real_gamma,
    rhf_energy_components,
    write_json,
)


CHEMICAL_ACCURACY_HARTREE = 1.593601e-3


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=float), delimiter=",", fmt="%.12f")


def resolve_shots(
    explicit: int | None, inferred: int | None, filename: str
) -> int:
    if explicit is not None:
        return explicit
    if inferred is None:
        raise ValueError(
            f"Cannot infer shots for {filename}; pass --shots explicitly."
        )
    return inferred


def multinomial_probabilities(
    probabilities: dict[str, float],
    shots: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    keys = sorted(probabilities)
    vector = np.asarray([probabilities[key] for key in keys], dtype=float)
    counts = rng.multinomial(shots, vector)
    return {
        key: float(count / shots) for key, count in zip(keys, counts)
    }


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "q025": float(q025),
        "median": float(median),
        "q975": float(q975),
    }


def bootstrap(
    diagonal: dict[str, float],
    real: dict[str, float],
    diagonal_shots: int,
    real_shots: int,
    bit_order: str,
    reference: dict,
    target_gamma: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    for index in range(repetitions):
        diagonal_sample = multinomial_probabilities(
            diagonal, diagonal_shots, rng
        )
        real_sample = multinomial_probabilities(real, real_shots, rng)
        gamma = reconstruct_real_gamma(
            diagonal_sample, real_sample, bit_order
        )
        projected = rank_one_projector(gamma)
        raw_energy = rhf_energy_components(
            gamma,
            reference["h1_orth"],
            reference["eri_orth"],
            float(reference["nuclear_repulsion"]),
        )["total_hartree"]
        projected_energy = rhf_energy_components(
            projected,
            reference["h1_orth"],
            reference["eri_orth"],
            float(reference["nuclear_repulsion"]),
        )["total_hartree"]
        records.append(
            {
                "bootstrap_index": float(index),
                "raw_energy_hartree": raw_energy,
                "projected_energy_hartree": projected_energy,
                "raw_energy_error_millihartree": 1000.0
                * (raw_energy - float(reference["rhf_energy"])),
                "projected_energy_error_millihartree": 1000.0
                * (projected_energy - float(reference["rhf_energy"])),
                "raw_gamma_frobenius_error": float(
                    np.linalg.norm(gamma - target_gamma)
                ),
                "projected_gamma_frobenius_error": float(
                    np.linalg.norm(projected - target_gamma)
                ),
            }
        )

    numeric_keys = [
        key for key in records[0] if key != "bootstrap_index"
    ]
    summary = {
        key: percentile_summary(
            np.asarray([record[key] for record in records], dtype=float)
        )
        for key in numeric_keys
    }
    return records, summary


def write_bootstrap_csv(path: Path, records: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_gamma_comparison(
    path: Path,
    target: np.ndarray,
    raw: np.ndarray,
    projected: np.ndarray,
) -> None:
    matrices = [target, raw, projected]
    titles = ["Classical RHF target", "Measured raw", "Rank-1 projected"]
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(matrix, vmin=-0.05, vmax=1.05, cmap="viridis")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.4f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] < 0.72 else "black",
                    fontsize=10,
                )
        axis.set_title(title, fontsize=10)
        axis.set_xticks([0, 1], ["q0", "q1"])
        axis.set_yticks([0, 1], ["q0", "q1"])
    figure.colorbar(image, ax=axes, shrink=0.78, label="1-RDM element")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_energy_comparison(
    path: Path,
    energies: dict[str, float],
    bootstrap_summary: dict[str, dict[str, float]],
) -> None:
    labels = list(energies)
    values = np.asarray([energies[label] for label in labels])
    reference = energies["PySCF RHF"]
    errors_mha = 1000.0 * (values - reference)
    figure, axis = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    colors = ["#404B69", "#4C78A8", "#F28E2B"]
    bars = axis.bar(labels, errors_mha, color=colors)
    if "Gaussian RHF" in energies:
        bars[-1].set_color("#59A14F")
    axis.axhspan(
        -1000 * CHEMICAL_ACCURACY_HARTREE,
        1000 * CHEMICAL_ACCURACY_HARTREE,
        color="#59A14F",
        alpha=0.13,
        label="±1 kcal/mol",
    )
    if "Measured raw" in labels:
        raw_index = labels.index("Measured raw")
        raw_ci = bootstrap_summary["raw_energy_error_millihartree"]
        axis.vlines(
            raw_index,
            raw_ci["q025"],
            raw_ci["q975"],
            color="black",
            linewidth=1.4,
        )
        axis.hlines(
            [raw_ci["q025"], raw_ci["q975"]],
            raw_index - 0.06,
            raw_index + 0.06,
            color="black",
            linewidth=1.4,
        )
    if "Rank-1 projected" in labels:
        projected_index = labels.index("Rank-1 projected")
        projected_ci = bootstrap_summary[
            "projected_energy_error_millihartree"
        ]
        axis.vlines(
            projected_index,
            projected_ci["q025"],
            projected_ci["q975"],
            color="black",
            linewidth=1.4,
        )
        axis.hlines(
            [projected_ci["q025"], projected_ci["q975"]],
            projected_index - 0.06,
            projected_index + 0.06,
            color="black",
            linewidth=1.4,
        )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_ylabel("Energy error relative to PySCF RHF (mHa)")
    axis.set_title("H₂ RHF/STO-3G energy closure")
    axis.legend(frameon=False)
    axis.tick_params(axis="x", rotation=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/example_ideal_counts"),
    )
    parser.add_argument(
        "--diagonal-file", default="z_diagonal.json"
    )
    parser.add_argument(
        "--real-file", default="real_offdiagonal.json"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reference/h2_r0.7414_sto3g_reference.npz"),
    )
    parser.add_argument(
        "--gaussian-log", type=Path, default=None
    )
    parser.add_argument(
        "--bit-order", choices=("q1q0", "q0q1"), default="q1q0"
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5210)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_demo")
    )
    arguments = parser.parse_args()
    if arguments.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive.")

    reference = load_reference(arguments.reference)
    target_gamma = np.asarray(
        reference["gamma_one_spin_orth"], dtype=float
    )

    diagonal_loaded = load_distribution(
        arguments.input_dir / arguments.diagonal_file
    )
    real_loaded = load_distribution(
        arguments.input_dir / arguments.real_file
    )
    diagonal, diagonal_pass = postselect_particle_number(
        diagonal_loaded.probabilities
    )
    real, real_pass = postselect_particle_number(
        real_loaded.probabilities
    )
    diagonal_shots = resolve_shots(
        arguments.shots,
        diagonal_loaded.shots,
        arguments.diagonal_file,
    )
    real_shots = resolve_shots(
        arguments.shots,
        real_loaded.shots,
        arguments.real_file,
    )
    diagonal_effective_shots = max(
        1, int(round(diagonal_shots * diagonal_pass))
    )
    real_effective_shots = max(1, int(round(real_shots * real_pass)))

    raw_gamma = reconstruct_real_gamma(
        diagonal, real, arguments.bit_order
    )
    projected_gamma = rank_one_projector(raw_gamma)
    raw_energy = rhf_energy_components(
        raw_gamma,
        reference["h1_orth"],
        reference["eri_orth"],
        float(reference["nuclear_repulsion"]),
    )
    projected_energy = rhf_energy_components(
        projected_gamma,
        reference["h1_orth"],
        reference["eri_orth"],
        float(reference["nuclear_repulsion"]),
    )
    target_energy = rhf_energy_components(
        target_gamma,
        reference["h1_orth"],
        reference["eri_orth"],
        float(reference["nuclear_repulsion"]),
    )
    pyscf_energy = float(reference["rhf_energy"])

    records, bootstrap_summary = bootstrap(
        diagonal,
        real,
        diagonal_effective_shots,
        real_effective_shots,
        arguments.bit_order,
        reference,
        target_gamma,
        arguments.bootstrap,
        arguments.seed,
    )

    gaussian_energy: float | None = None
    if arguments.gaussian_log is not None:
        energies = parse_gaussian_scf_energies(
            arguments.gaussian_log.read_text(
                encoding="utf-8", errors="replace"
            )
        )
        if not energies:
            raise ValueError(
                f"No 'SCF Done' energy found in {arguments.gaussian_log}."
            )
        gaussian_energy = energies[-1]

    raw_error = raw_energy["total_hartree"] - pyscf_energy
    projected_error = projected_energy["total_hartree"] - pyscf_energy
    fidelities = rank_one_orbital_fidelities(
        projected_gamma, target_gamma
    )
    circuit_level_pass = bool(
        min(diagonal_pass, real_pass) >= 0.95
        and fidelities["one_spin_orbital_fidelity"] >= 0.99
    )
    projected_energy_chemical_accuracy_pass = bool(
        abs(projected_error) <= CHEMICAL_ACCURACY_HARTREE
    )

    gaussian_comparison = None
    if gaussian_energy is not None:
        gaussian_comparison = {
            "gaussian_rhf_energy_hartree": gaussian_energy,
            "gaussian_minus_pyscf_hartree": gaussian_energy - pyscf_energy,
            "gaussian_classical_crosscheck_pass_1_microhartree": bool(
                abs(gaussian_energy - pyscf_energy) <= 1e-6
            ),
            "raw_quantum_minus_gaussian_hartree": (
                raw_energy["total_hartree"] - gaussian_energy
            ),
            "projected_quantum_minus_gaussian_hartree": (
                projected_energy["total_hartree"] - gaussian_energy
            ),
        }

    summary = {
        "model": {
            "molecule": "H2",
            "method": str(reference["method"]),
            "basis": str(reference["basis"]),
            "bond_length_angstrom": float(
                reference["bond_length_angstrom"]
            ),
            "encoding": "2 qubits, 1 particle, one RHF spin sector",
            "bitstring_order": arguments.bit_order,
        },
        "input": {
            "diagonal_file": arguments.diagonal_file,
            "real_file": arguments.real_file,
            "diagonal_value_type": diagonal_loaded.value_type,
            "real_value_type": real_loaded.value_type,
            "diagonal_shots": diagonal_shots,
            "real_shots": real_shots,
            "diagonal_particle_number_pass_probability": diagonal_pass,
            "real_particle_number_pass_probability": real_pass,
        },
        "gamma_raw": raw_gamma.tolist(),
        "gamma_projected_rank1": projected_gamma.tolist(),
        "gamma_reference": target_gamma.tolist(),
        "gamma_raw_metrics": gamma_metrics(raw_gamma, target_gamma),
        "gamma_projected_metrics": gamma_metrics(
            projected_gamma, target_gamma
        ),
        "fidelities": fidelities,
        "energy": {
            "pyscf_rhf_hartree": pyscf_energy,
            "functional_target_hartree": target_energy["total_hartree"],
            "functional_closure_error_hartree": (
                target_energy["total_hartree"] - pyscf_energy
            ),
            "raw_components": raw_energy,
            "projected_components": projected_energy,
            "raw_error_hartree": raw_error,
            "raw_error_millihartree": 1000.0 * raw_error,
            "projected_error_hartree": projected_error,
            "projected_error_millihartree": 1000.0 * projected_error,
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        },
        "bootstrap_95_percent": bootstrap_summary,
        "gaussian": gaussian_comparison,
        "acceptance": {
            "circuit_level_pass": circuit_level_pass,
            "circuit_thresholds": {
                "particle_number_pass_probability_min": 0.95,
                "one_spin_orbital_fidelity_min": 0.99,
            },
            "projected_energy_within_1_kcal_per_mol": (
                projected_energy_chemical_accuracy_pass
            ),
            "important_boundary": (
                "The Gaussian comparison is molecular-level only when geometry, "
                "basis, RHF definition, orbital basis transform, and nuclear "
                "repulsion are identical."
            ),
        },
    }

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output_dir / "summary.json", summary)
    save_matrix_csv(arguments.output_dir / "gamma_raw.csv", raw_gamma)
    save_matrix_csv(
        arguments.output_dir / "gamma_projected_rank1.csv",
        projected_gamma,
    )
    save_matrix_csv(
        arguments.output_dir / "gamma_reference.csv", target_gamma
    )
    write_bootstrap_csv(
        arguments.output_dir / "bootstrap_energy.csv", records
    )

    energy_rows = [
        ["quantity", "energy_hartree", "error_vs_pyscf_millihartree"],
        ["PySCF RHF", pyscf_energy, 0.0],
        [
            "Measured raw",
            raw_energy["total_hartree"],
            1000.0 * raw_error,
        ],
        [
            "Rank-1 projected",
            projected_energy["total_hartree"],
            1000.0 * projected_error,
        ],
    ]
    if gaussian_energy is not None:
        energy_rows.append(
            [
                "Gaussian RHF",
                gaussian_energy,
                1000.0 * (gaussian_energy - pyscf_energy),
            ]
        )
    with (arguments.output_dir / "energy_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(energy_rows)

    plot_gamma_comparison(
        arguments.output_dir / "gamma_comparison.png",
        target_gamma,
        raw_gamma,
        projected_gamma,
    )
    plotted_energies = {
        "PySCF RHF": pyscf_energy,
        "Measured raw": raw_energy["total_hartree"],
        "Rank-1 projected": projected_energy["total_hartree"],
    }
    if gaussian_energy is not None:
        plotted_energies["Gaussian RHF"] = gaussian_energy
    plot_energy_comparison(
        arguments.output_dir / "energy_comparison.png",
        plotted_energies,
        bootstrap_summary,
    )

    concise = {
        "pyscf_rhf_hartree": pyscf_energy,
        "raw_quantum_hartree": raw_energy["total_hartree"],
        "projected_quantum_hartree": projected_energy["total_hartree"],
        "raw_error_millihartree": 1000.0 * raw_error,
        "projected_error_millihartree": 1000.0 * projected_error,
        "one_spin_orbital_fidelity": fidelities[
            "one_spin_orbital_fidelity"
        ],
        "circuit_level_pass": circuit_level_pass,
        "projected_energy_within_1_kcal_per_mol": (
            projected_energy_chemical_accuracy_pass
        ),
        "output_dir": str(arguments.output_dir),
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
