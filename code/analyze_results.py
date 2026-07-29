#!/usr/bin/env python3
"""Reconstruct H4's one-spin 1-RDM and RHF energy from four JSON files."""

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

from h4_core import (
    N_PARTICLES_PER_SPIN,
    gamma_metrics,
    load_distribution,
    load_reference,
    parse_gaussian_scf_energies,
    parse_native_circuit,
    postselect_particle_number,
    rank_projector,
    reconstruct_gamma,
    rhf_energy_components,
    slater_probabilities_from_circuit,
    slater_subspace_fidelity,
    total_variation_distance,
    write_json,
)


CHEMICAL_ACCURACY_HARTREE = 1.593601e-3


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=float), delimiter=",", fmt="%.12f")


def write_records_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def resolve_shots(
    explicit: int | None, inferred: int | None, filename: str
) -> int | None:
    if explicit is not None:
        return explicit
    if inferred is not None:
        return inferred
    return None


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


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"q025": math.nan, "median": math.nan, "q975": math.nan}
    q025, median, q975 = np.quantile(array, [0.025, 0.5, 0.975])
    return {
        "q025": float(q025),
        "median": float(median),
        "q975": float(q975),
    }


def evaluate_gamma(
    gamma: np.ndarray,
    target_gamma: np.ndarray,
    reference: dict,
) -> dict[str, float]:
    projected = rank_projector(gamma, N_PARTICLES_PER_SPIN)
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
    rhf_energy = float(reference["rhf_energy"])
    fidelity = slater_subspace_fidelity(
        projected, target_gamma, N_PARTICLES_PER_SPIN
    )
    return {
        "raw_energy_hartree": raw_energy,
        "projected_energy_hartree": projected_energy,
        "raw_energy_error_millihartree": 1000.0
        * (raw_energy - rhf_energy),
        "projected_energy_error_millihartree": 1000.0
        * (projected_energy - rhf_energy),
        "raw_gamma_frobenius_error": float(
            np.linalg.norm(gamma - target_gamma)
        ),
        "projected_gamma_frobenius_error": float(
            np.linalg.norm(projected - target_gamma)
        ),
        "raw_idempotency_frobenius": float(
            np.linalg.norm(gamma @ gamma - gamma)
        ),
        "one_spin_slater_fidelity": fidelity,
        "full_rhf_determinant_fidelity": fidelity**2,
    }


def bootstrap_model(
    source_distributions: dict[str, dict[str, float]],
    shots_by_setting: dict[str, int],
    manifest: dict,
    target_gamma: np.ndarray,
    reference: dict,
    ideal_distributions: dict[str, dict[str, float]],
    repetitions: int,
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for bootstrap_index in range(repetitions):
        postselected_samples: dict[str, dict[str, float]] = {}
        tv_values: list[float] = []
        valid = True
        for circuit in manifest["circuits"]:
            setting = circuit["setting"]
            sampled_full = multinomial_probabilities(
                source_distributions[setting],
                shots_by_setting[setting],
                rng,
            )
            try:
                sampled_selected, _ = postselect_particle_number(
                    sampled_full
                )
            except ValueError:
                valid = False
                break
            postselected_samples[setting] = sampled_selected
            tv_values.append(
                total_variation_distance(
                    sampled_full, ideal_distributions[setting]
                )
            )
        if not valid:
            continue
        gamma = reconstruct_gamma(postselected_samples, manifest)
        evaluated = evaluate_gamma(gamma, target_gamma, reference)
        records.append(
            {
                "bootstrap_index": bootstrap_index,
                "mean_setting_tv_vs_ideal": float(np.mean(tv_values)),
                **evaluated,
            }
        )
    if not records:
        raise RuntimeError("All bootstrap samples failed postselection.")
    numeric_keys = [
        key for key in records[0] if key != "bootstrap_index"
    ]
    summary = {
        key: percentile_summary(
            [float(record[key]) for record in records]
        )
        for key in numeric_keys
    }
    return records, summary


def plot_gamma_comparison(
    path: Path,
    target: np.ndarray,
    raw: np.ndarray,
    projected: np.ndarray,
) -> None:
    matrices = [target, raw, projected, raw - target]
    titles = [
        "Classical RHF target",
        "Measured raw",
        "Rank-2 projected",
        "Raw minus target",
    ]
    vmins = [-0.55, -0.55, -0.55, -0.12]
    vmaxs = [1.05, 1.05, 1.05, 0.12]
    cmaps = ["coolwarm", "coolwarm", "coolwarm", "RdBu_r"]
    figure, axes = plt.subplots(
        2, 2, figsize=(8.2, 7.0), constrained_layout=True
    )
    for axis, matrix, title, vmin, vmax, cmap in zip(
        axes.flat, matrices, titles, vmins, vmaxs, cmaps
    ):
        image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
        for row in range(4):
            for column in range(4):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(4), [f"q{i}" for i in range(4)])
        axis.set_yticks(range(4), [f"q{i}" for i in range(4)])
        figure.colorbar(image, ax=axis, shrink=0.78)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_energy_comparison(
    path: Path,
    energies: dict[str, float],
    measured_bootstrap: dict[str, dict[str, float]] | None,
) -> None:
    labels = list(energies)
    reference = energies["PySCF RHF"]
    errors = np.asarray(
        [1000.0 * (energies[label] - reference) for label in labels]
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    colors = ["#404B69", "#4C78A8", "#F28E2B", "#59A14F"]
    axis.bar(labels, errors, color=colors[: len(labels)])
    axis.axhspan(
        -1000.0 * CHEMICAL_ACCURACY_HARTREE,
        1000.0 * CHEMICAL_ACCURACY_HARTREE,
        color="#59A14F",
        alpha=0.14,
        label="±1 kcal/mol",
    )
    if measured_bootstrap is not None:
        for label, key in [
            ("Measured raw", "raw_energy_error_millihartree"),
            ("Rank-2 projected", "projected_energy_error_millihartree"),
        ]:
            if label not in labels:
                continue
            index = labels.index(label)
            interval = measured_bootstrap[key]
            axis.vlines(
                index,
                interval["q025"],
                interval["q975"],
                color="black",
                linewidth=1.3,
            )
            axis.hlines(
                [interval["q025"], interval["q975"]],
                index - 0.06,
                index + 0.06,
                color="black",
                linewidth=1.3,
            )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_ylabel("Energy error relative to PySCF RHF (mHa)")
    axis.set_title("Linear H₄ RHF/STO-3G energy closure")
    axis.tick_params(axis="x", rotation=12)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/example_ideal_probabilities"),
    )
    parser.add_argument(
        "--circuit-dir", type=Path, default=Path("circuits")
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reference/h4_r1.3000_sto3g_reference.npz"),
    )
    parser.add_argument(
        "--gaussian-log", type=Path, default=None
    )
    parser.add_argument(
        "--bit-order",
        choices=("q3q2q1q0", "q0q1q2q3"),
        default="q3q2q1q0",
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5210)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results")
    )
    arguments = parser.parse_args()
    if arguments.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative.")
    if arguments.shots is not None and arguments.shots <= 0:
        raise ValueError("--shots must be positive.")

    manifest = json.loads(
        (arguments.circuit_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    reference = load_reference(arguments.reference)
    target_gamma = np.asarray(
        reference["gamma_one_spin_orth"], dtype=float
    )

    full_distributions: dict[str, dict[str, float]] = {}
    postselected_distributions: dict[str, dict[str, float]] = {}
    ideal_distributions: dict[str, dict[str, float]] = {}
    shots_by_setting: dict[str, int] = {}
    setting_rows: list[dict] = []
    for circuit in manifest["circuits"]:
        setting = circuit["setting"]
        input_path = arguments.input_dir / f"{setting}.json"
        loaded = load_distribution(input_path)
        postselected, pass_probability = postselect_particle_number(
            loaded.probabilities
        )
        resolved_shots = resolve_shots(
            arguments.shots, loaded.shots, input_path.name
        )
        if resolved_shots is not None:
            shots_by_setting[setting] = resolved_shots
        full_distributions[setting] = loaded.probabilities
        postselected_distributions[setting] = postselected

        gate_path = arguments.circuit_dir / circuit["gate_body_file"]
        ideal = slater_probabilities_from_circuit(
            parse_native_circuit(gate_path.read_text(encoding="utf-8"))
        )
        ideal_distributions[setting] = ideal
        setting_rows.append(
            {
                "setting": setting,
                "file": input_path.name,
                "value_type": loaded.value_type,
                "shots": resolved_shots,
                "particle_number_pass_probability": pass_probability,
                "tv_distance_full_vs_exact_circuit": (
                    total_variation_distance(loaded.probabilities, ideal)
                ),
            }
        )

    raw_gamma = reconstruct_gamma(
        postselected_distributions, manifest, arguments.bit_order
    )
    projected_gamma = rank_projector(
        raw_gamma, N_PARTICLES_PER_SPIN
    )
    raw_metrics = gamma_metrics(raw_gamma, target_gamma)
    projected_metrics = gamma_metrics(projected_gamma, target_gamma)
    evaluated = evaluate_gamma(raw_gamma, target_gamma, reference)
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

    measured_records: list[dict] = []
    measured_summary: dict[str, dict[str, float]] | None = None
    ideal_records: list[dict] = []
    ideal_summary: dict[str, dict[str, float]] | None = None
    if arguments.bootstrap > 0:
        missing = [
            circuit["setting"]
            for circuit in manifest["circuits"]
            if circuit["setting"] not in shots_by_setting
        ]
        if missing:
            raise ValueError(
                "Bootstrap needs shots for every setting; pass --shots. "
                f"Missing: {missing}."
            )
        measured_records, measured_summary = bootstrap_model(
            full_distributions,
            shots_by_setting,
            manifest,
            target_gamma,
            reference,
            ideal_distributions,
            arguments.bootstrap,
            arguments.seed,
        )
        ideal_records, ideal_summary = bootstrap_model(
            ideal_distributions,
            shots_by_setting,
            manifest,
            target_gamma,
            reference,
            ideal_distributions,
            arguments.bootstrap,
            arguments.seed + 1,
        )

    gaussian_energy: float | None = None
    gaussian_energies: list[float] = []
    if arguments.gaussian_log is not None:
        gaussian_energies = parse_gaussian_scf_energies(
            arguments.gaussian_log.read_text(
                encoding="utf-8", errors="replace"
            )
        )
        if not gaussian_energies:
            raise ValueError("No 'SCF Done' energy found in Gaussian log.")
        gaussian_energy = gaussian_energies[-1]

    mean_tv = float(
        np.mean(
            [
                row["tv_distance_full_vs_exact_circuit"]
                for row in setting_rows
            ]
        )
    )
    min_particle_pass = float(
        min(
            row["particle_number_pass_probability"]
            for row in setting_rows
        )
    )
    energy_error = abs(
        projected_energy["total_hartree"]
        - float(reference["rhf_energy"])
    )
    if ideal_summary is None:
        shot_model_pass = mean_tv <= 1e-9
        shot_tv_q975 = None
    else:
        shot_tv_q975 = ideal_summary["mean_setting_tv_vs_ideal"]["q975"]
        shot_model_pass = mean_tv <= shot_tv_q975 + 1e-12
    point_energy_pass = energy_error <= CHEMICAL_ACCURACY_HARTREE
    if measured_summary is None:
        confidence_energy_pass = point_energy_pass
        projected_energy_error_95 = None
    else:
        projected_energy_error_95 = measured_summary[
            "projected_energy_error_millihartree"
        ]
        confidence_energy_pass = max(
            abs(projected_energy_error_95["q025"]),
            abs(projected_energy_error_95["q975"]),
        ) <= 1000.0 * CHEMICAL_ACCURACY_HARTREE

    acceptance = {
        "particle_number_pass": min_particle_pass >= 0.95,
        "particle_number_threshold": 0.95,
        "shot_model_pass": shot_model_pass,
        "ideal_shot_mean_tv_q975": shot_tv_q975,
        "one_spin_slater_fidelity_pass": (
            evaluated["one_spin_slater_fidelity"] >= 0.99
        ),
        "one_spin_slater_fidelity_threshold": 0.99,
        "projected_energy_point_within_1_kcal_per_mol": point_energy_pass,
        "projected_energy_95_percent_interval_millihartree": (
            projected_energy_error_95
        ),
        "projected_energy_95_percent_within_1_kcal_per_mol": (
            confidence_energy_pass
        ),
    }
    acceptance["overall_point_estimate_pass"] = bool(
        acceptance["particle_number_pass"]
        and acceptance["shot_model_pass"]
        and acceptance["one_spin_slater_fidelity_pass"]
        and acceptance["projected_energy_point_within_1_kcal_per_mol"]
    )
    acceptance["overall_molecular_reproduction_pass"] = bool(
        acceptance["overall_point_estimate_pass"]
        and acceptance[
            "projected_energy_95_percent_within_1_kcal_per_mol"
        ]
    )

    summary = {
        "model": {
            "molecule": "linear H4",
            "method": "RHF",
            "basis": "STO-3G",
            "spacing_angstrom": float(reference["spacing_angstrom"]),
            "encoding": "4 qubits, 2 particles, one RHF spin sector",
            "bitstring_order": arguments.bit_order,
        },
        "input": {
            "directory": str(arguments.input_dir),
            "settings": setting_rows,
            "mean_setting_tv_vs_exact_circuit": mean_tv,
            "minimum_particle_number_pass_probability": min_particle_pass,
        },
        "gamma_reference": target_gamma.tolist(),
        "gamma_raw": raw_gamma.tolist(),
        "gamma_projected_rank2": projected_gamma.tolist(),
        "gamma_raw_metrics": raw_metrics,
        "gamma_projected_metrics": projected_metrics,
        "fidelities": {
            "one_spin_slater_fidelity": evaluated[
                "one_spin_slater_fidelity"
            ],
            "full_rhf_determinant_fidelity": evaluated[
                "full_rhf_determinant_fidelity"
            ],
        },
        "energy": {
            "pyscf_rhf_hartree": float(reference["rhf_energy"]),
            "functional_target": target_energy,
            "functional_closure_error_hartree": (
                target_energy["total_hartree"]
                - float(reference["rhf_energy"])
            ),
            "raw": raw_energy,
            "projected_rank2": projected_energy,
            "raw_error_millihartree": evaluated[
                "raw_energy_error_millihartree"
            ],
            "projected_error_millihartree": evaluated[
                "projected_energy_error_millihartree"
            ],
            "chemical_accuracy_hartree": CHEMICAL_ACCURACY_HARTREE,
        },
        "bootstrap_measured_95_percent": measured_summary,
        "bootstrap_ideal_shot_model_95_percent": ideal_summary,
        "gaussian": (
            None
            if gaussian_energy is None
            else {
                "log": str(arguments.gaussian_log),
                "all_scf_energies_hartree": gaussian_energies,
                "final_rhf_hartree": gaussian_energy,
                "difference_vs_pyscf_microhartree": 1e6
                * (gaussian_energy - float(reference["rhf_energy"])),
                "difference_vs_projected_quantum_millihartree": 1000.0
                * (
                    projected_energy["total_hartree"]
                    - gaussian_energy
                ),
            }
        ),
        "acceptance": acceptance,
    }

    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    save_matrix_csv(output / "gamma_reference.csv", target_gamma)
    save_matrix_csv(output / "gamma_raw.csv", raw_gamma)
    save_matrix_csv(output / "gamma_projected_rank2.csv", projected_gamma)
    write_records_csv(output / "setting_summary.csv", setting_rows)
    write_records_csv(
        output / "bootstrap_measured.csv", measured_records
    )
    write_records_csv(
        output / "bootstrap_ideal_shot_model.csv", ideal_records
    )

    energy_rows = [
        {
            "quantity": "PySCF RHF",
            "energy_hartree": float(reference["rhf_energy"]),
            "error_vs_pyscf_millihartree": 0.0,
        },
        {
            "quantity": "Measured raw",
            "energy_hartree": raw_energy["total_hartree"],
            "error_vs_pyscf_millihartree": evaluated[
                "raw_energy_error_millihartree"
            ],
        },
        {
            "quantity": "Rank-2 projected",
            "energy_hartree": projected_energy["total_hartree"],
            "error_vs_pyscf_millihartree": evaluated[
                "projected_energy_error_millihartree"
            ],
        },
    ]
    if gaussian_energy is not None:
        energy_rows.append(
            {
                "quantity": "Gaussian RHF",
                "energy_hartree": gaussian_energy,
                "error_vs_pyscf_millihartree": 1000.0
                * (gaussian_energy - float(reference["rhf_energy"])),
            }
        )
    write_records_csv(output / "energy_summary.csv", energy_rows)

    plot_gamma_comparison(
        output / "gamma_comparison.png",
        target_gamma,
        raw_gamma,
        projected_gamma,
    )
    energy_plot_data = {
        row["quantity"]: float(row["energy_hartree"])
        for row in energy_rows
    }
    plot_energy_comparison(
        output / "energy_comparison.png",
        energy_plot_data,
        measured_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
