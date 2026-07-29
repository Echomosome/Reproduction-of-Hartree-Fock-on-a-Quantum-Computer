#!/usr/bin/env python3
"""Reconstruct one diazene 1-RDM and RHF energy from ten JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diazene_core import (
    DEFAULT_BIT_ORDER,
    N_MODES,
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
PAPER_MECHANISM_SCALE_HARTREE = 40.0e-3


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",", fmt="%.12f")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def multinomial_probabilities(
    probabilities: dict[str, float],
    shots: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    keys = sorted(probabilities)
    vector = np.asarray([probabilities[key] for key in keys], dtype=float)
    counts = rng.multinomial(shots, vector)
    return {
        key: float(count / shots)
        for key, count in zip(keys, counts)
        if count > 0
    }


def percentile(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    q025, median, q975 = np.quantile(array, [0.025, 0.5, 0.975])
    return {
        "q025": float(q025),
        "median": float(median),
        "q975": float(q975),
    }


def evaluate_gamma(
    gamma: np.ndarray,
    target: np.ndarray,
    reference: dict[str, np.ndarray],
) -> dict[str, float]:
    projected = rank_projector(gamma, N_PARTICLES_PER_SPIN)
    constant = float(reference["constant_offset_hartree"])
    raw_energy = rhf_energy_components(
        gamma,
        reference["h1_active"],
        reference["eri_active"],
        constant,
    )["total_hartree"]
    projected_energy = rhf_energy_components(
        projected,
        reference["h1_active"],
        reference["eri_active"],
        constant,
    )["total_hartree"]
    active_reference = float(reference["active_rhf_energy_hartree"])
    full_reference = float(reference["full_rhf_energy_hartree"])
    fidelity = slater_subspace_fidelity(
        projected, target, N_PARTICLES_PER_SPIN
    )
    return {
        "raw_energy_hartree": raw_energy,
        "projected_energy_hartree": projected_energy,
        "raw_measurement_error_vs_active_millihartree": (
            1000.0 * (raw_energy - active_reference)
        ),
        "projected_measurement_error_vs_active_millihartree": (
            1000.0 * (projected_energy - active_reference)
        ),
        "raw_end_to_end_error_vs_full_millihartree": (
            1000.0 * (raw_energy - full_reference)
        ),
        "projected_end_to_end_error_vs_full_millihartree": (
            1000.0 * (projected_energy - full_reference)
        ),
        "raw_gamma_frobenius_error": float(
            np.linalg.norm(gamma - target)
        ),
        "projected_gamma_frobenius_error": float(
            np.linalg.norm(projected - target)
        ),
        "raw_idempotency_frobenius": float(
            np.linalg.norm(gamma @ gamma - gamma)
        ),
        "one_spin_slater_fidelity": fidelity,
        "closed_shell_determinant_fidelity": fidelity**2,
    }


def bootstrap(
    source: dict[str, dict[str, float]],
    shots: dict[str, int],
    manifest: dict,
    target: np.ndarray,
    reference: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for index in range(repetitions):
        selected: dict[str, dict[str, float]] = {}
        valid = True
        for circuit in manifest["circuits"]:
            setting = circuit["setting"]
            sampled = multinomial_probabilities(
                source[setting], shots[setting], rng
            )
            try:
                selected[setting], _ = postselect_particle_number(sampled)
            except ValueError:
                valid = False
                break
        if not valid:
            continue
        gamma = reconstruct_gamma(selected, manifest)
        records.append(
            {"bootstrap_index": index, **evaluate_gamma(
                gamma, target, reference
            )}
        )
    if not records:
        raise RuntimeError("All bootstrap replicates failed postselection.")
    keys = [key for key in records[0] if key != "bootstrap_index"]
    summary = {
        key: percentile([float(record[key]) for record in records])
        for key in keys
    }
    return records, summary


def plot_gamma(
    path: Path,
    target: np.ndarray,
    raw: np.ndarray,
    projected: np.ndarray,
) -> None:
    matrices = [target, raw, projected, raw - target]
    titles = [
        "Frozen-core RHF target",
        "Measured raw",
        "Rank-6 projected",
        "Raw minus target",
    ]
    figure, axes = plt.subplots(
        2, 2, figsize=(9.2, 8.0), constrained_layout=True
    )
    for axis, matrix, title in zip(axes.flat, matrices, titles):
        limit = 1.0 if title != "Raw minus target" else max(
            0.05, float(np.max(np.abs(matrix)))
        )
        image = axis.imshow(
            matrix,
            vmin=-limit,
            vmax=limit,
            cmap="RdBu_r",
        )
        axis.set_title(title)
        axis.set_xticks(range(N_MODES), [f"q{i}" for i in range(N_MODES)])
        axis.set_yticks(range(N_MODES), [f"q{i}" for i in range(N_MODES)])
        axis.tick_params(labelsize=7)
        figure.colorbar(image, ax=axis, shrink=0.78)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_energy(
    path: Path,
    full_reference: float,
    energies: dict[str, float],
) -> None:
    labels = list(energies)
    errors = [
        1000.0 * (energies[label] - full_reference) for label in labels
    ]
    figure, axis = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    colors = ["#404B69", "#4C78A8", "#F28E2B", "#59A14F", "#B279A2"]
    axis.bar(labels, errors, color=colors[: len(labels)])
    axis.axhspan(
        -1000.0 * CHEMICAL_ACCURACY_HARTREE,
        1000.0 * CHEMICAL_ACCURACY_HARTREE,
        color="#59A14F",
        alpha=0.13,
        label="±1 kcal/mol",
    )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_ylabel("Error relative to full PySCF RHF (mHa)")
    axis.tick_params(axis="x", rotation=14)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def resolve_input_directory(root: Path, geometry_id: str) -> Path:
    nested = root / geometry_id
    return nested if nested.is_dir() else root


def analyze(
    geometry_id: str,
    input_root: Path,
    circuit_root: Path,
    reference_root: Path,
    output_dir: Path,
    bit_order: str,
    explicit_shots: int | None,
    bootstrap_repetitions: int,
    seed: int,
    gaussian_log: Path | None,
) -> dict:
    geometry_circuit_dir = circuit_root / geometry_id
    manifest = json.loads(
        (geometry_circuit_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    reference = load_reference(reference_root / f"{geometry_id}.npz")
    target = np.asarray(
        reference["gamma_active_one_spin"], dtype=float
    )
    input_dir = resolve_input_directory(input_root, geometry_id)

    full_distributions: dict[str, dict[str, float]] = {}
    selected_distributions: dict[str, dict[str, float]] = {}
    ideal_distributions: dict[str, dict[str, float]] = {}
    shots_by_setting: dict[str, int] = {}
    setting_rows: list[dict] = []
    for circuit in manifest["circuits"]:
        setting = circuit["setting"]
        input_path = input_dir / f"{setting}.json"
        loaded = load_distribution(input_path)
        selected, pass_probability = postselect_particle_number(
            loaded.probabilities
        )
        shots = explicit_shots or loaded.shots
        if shots is not None:
            shots_by_setting[setting] = shots
        full_distributions[setting] = loaded.probabilities
        selected_distributions[setting] = selected

        gates = parse_native_circuit(
            (
                geometry_circuit_dir / circuit["gate_body_file"]
            ).read_text(encoding="utf-8")
        )
        ideal = slater_probabilities_from_circuit(gates)
        ideal_distributions[setting] = ideal
        setting_rows.append(
            {
                "setting": setting,
                "input_file": input_path.name,
                "value_type": loaded.value_type,
                "shots": shots,
                "particle_number_pass_probability": pass_probability,
                "tv_distance_vs_exact_circuit": (
                    total_variation_distance(
                        loaded.probabilities, ideal
                    )
                ),
            }
        )

    raw = reconstruct_gamma(
        selected_distributions, manifest, bit_order
    )
    projected = rank_projector(raw, N_PARTICLES_PER_SPIN)
    evaluated = evaluate_gamma(raw, target, reference)
    constant = float(reference["constant_offset_hartree"])
    raw_components = rhf_energy_components(
        raw, reference["h1_active"], reference["eri_active"], constant
    )
    projected_components = rhf_energy_components(
        projected,
        reference["h1_active"],
        reference["eri_active"],
        constant,
    )
    target_components = rhf_energy_components(
        target,
        reference["h1_active"],
        reference["eri_active"],
        constant,
    )

    measured_bootstrap_records: list[dict] = []
    measured_bootstrap_summary = None
    ideal_bootstrap_records: list[dict] = []
    ideal_bootstrap_summary = None
    if bootstrap_repetitions > 0:
        missing = [
            circuit["setting"]
            for circuit in manifest["circuits"]
            if circuit["setting"] not in shots_by_setting
        ]
        if missing:
            raise ValueError(
                "Bootstrap needs shots for every setting; "
                "provide --shots. Missing: " + ", ".join(missing)
            )
        (
            measured_bootstrap_records,
            measured_bootstrap_summary,
        ) = bootstrap(
            full_distributions,
            shots_by_setting,
            manifest,
            target,
            reference,
            bootstrap_repetitions,
            seed,
        )
        (
            ideal_bootstrap_records,
            ideal_bootstrap_summary,
        ) = bootstrap(
            ideal_distributions,
            shots_by_setting,
            manifest,
            target,
            reference,
            bootstrap_repetitions,
            seed + 1,
        )

    gaussian_energies: list[float] = []
    gaussian_energy = None
    if gaussian_log is not None:
        gaussian_energies = parse_gaussian_scf_energies(
            gaussian_log.read_text(encoding="utf-8", errors="replace")
        )
        if not gaussian_energies:
            raise ValueError("No 'SCF Done' record in Gaussian log.")
        gaussian_energy = gaussian_energies[-1]

    full_reference = float(reference["full_rhf_energy_hartree"])
    active_reference = float(reference["active_rhf_energy_hartree"])
    energy_decomposition = {
        "quantum_measurement_error_hartree": (
            projected_components["total_hartree"] - active_reference
        ),
        "frozen_core_bias_hartree": (
            active_reference - full_reference
        ),
        "classical_program_difference_hartree": (
            None
            if gaussian_energy is None
            else full_reference - gaussian_energy
        ),
        "end_to_end_quantum_minus_gaussian_hartree": (
            None
            if gaussian_energy is None
            else projected_components["total_hartree"] - gaussian_energy
        ),
    }

    summary = {
        "geometry_id": geometry_id,
        "pathway": str(reference["pathway"]),
        "reaction_coordinate_deg": float(
            reference["reaction_coordinate_deg"]
        ),
        "input_directory": str(input_dir),
        "model": {
            "method": "RHF",
            "basis": "STO-3G",
            "full_spatial_modes": 12,
            "full_electrons": 16,
            "frozen_spatial_orbitals": 2,
            "active_modes": N_MODES,
            "active_particles_per_spin": N_PARTICLES_PER_SPIN,
            "measurement_settings": len(manifest["circuits"]),
            "bit_order": bit_order,
        },
        "energies_hartree": {
            "full_pyscf_rhf": full_reference,
            "frozen_core_active_rhf": active_reference,
            "measured_raw": raw_components["total_hartree"],
            "measured_rank6_projected": projected_components[
                "total_hartree"
            ],
            "gaussian_rhf": gaussian_energy,
        },
        "energy_components": {
            "target_active": target_components,
            "measured_raw": raw_components,
            "measured_rank6_projected": projected_components,
        },
        "error_decomposition": energy_decomposition,
        "raw_gamma_metrics": gamma_metrics(raw, target),
        "projected_gamma_metrics": gamma_metrics(projected, target),
        "fidelity": {
            "one_spin_slater": evaluated["one_spin_slater_fidelity"],
            "closed_shell_determinant": evaluated[
                "closed_shell_determinant_fidelity"
            ],
        },
        "setting_summary": setting_rows,
        "mean_tv_distance_vs_exact_circuit": float(
            np.mean(
                [
                    row["tv_distance_vs_exact_circuit"]
                    for row in setting_rows
                ]
            )
        ),
        "minimum_particle_number_pass_probability": float(
            min(
                row["particle_number_pass_probability"]
                for row in setting_rows
            )
        ),
        "thresholds": {
            "chemical_accuracy_hartree_1_kcal_per_mol": (
                CHEMICAL_ACCURACY_HARTREE
            ),
            "paper_mechanism_energy_scale_hartree": (
                PAPER_MECHANISM_SCALE_HARTREE
            ),
            "projected_point_within_chemical_accuracy_vs_active": (
                abs(
                    projected_components["total_hartree"]
                    - active_reference
                )
                <= CHEMICAL_ACCURACY_HARTREE
            ),
            "projected_point_within_40_millihartree_vs_active": (
                abs(
                    projected_components["total_hartree"]
                    - active_reference
                )
                <= PAPER_MECHANISM_SCALE_HARTREE
            ),
        },
        "bootstrap": {
            "repetitions": bootstrap_repetitions,
            "measured_distribution": measured_bootstrap_summary,
            "ideal_shot_model": ideal_bootstrap_summary,
        },
        "gaussian_scf_records_hartree": gaussian_energies,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(output_dir / "gamma_target.csv", target)
    save_matrix(output_dir / "gamma_raw.csv", raw)
    save_matrix(output_dir / "gamma_rank6_projected.csv", projected)
    write_csv(output_dir / "setting_summary.csv", setting_rows)
    write_csv(
        output_dir / "energy_summary.csv",
        [
            {
                "geometry_id": geometry_id,
                "full_pyscf_rhf_hartree": full_reference,
                "frozen_core_active_rhf_hartree": active_reference,
                "frozen_core_bias_millihartree": 1000.0
                * (active_reference - full_reference),
                "measured_raw_hartree": raw_components[
                    "total_hartree"
                ],
                "measured_rank6_projected_hartree": (
                    projected_components["total_hartree"]
                ),
                "rank6_measurement_error_vs_active_millihartree": (
                    evaluated[
                        "projected_measurement_error_vs_active_millihartree"
                    ]
                ),
                "rank6_end_to_end_error_vs_full_millihartree": (
                    evaluated[
                        "projected_end_to_end_error_vs_full_millihartree"
                    ]
                ),
                "gaussian_rhf_hartree": gaussian_energy,
            }
        ],
    )
    write_csv(
        output_dir / "bootstrap_measured.csv",
        measured_bootstrap_records,
    )
    write_csv(
        output_dir / "bootstrap_ideal_shot_model.csv",
        ideal_bootstrap_records,
    )
    write_json(output_dir / "summary.json", summary)
    plot_gamma(output_dir / "gamma_comparison.png", target, raw, projected)
    energy_plot_values = {
        "Frozen-core\nreference": active_reference,
        "Measured\nraw": raw_components["total_hartree"],
        "Rank-6\nprojected": projected_components["total_hartree"],
    }
    if gaussian_energy is not None:
        energy_plot_values["Gaussian\nRHF"] = gaussian_energy
    plot_energy(
        output_dir / "energy_comparison.png",
        full_reference,
        energy_plot_values,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-id", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--circuit-dir", type=Path, default=Path("circuits")
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("reference")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/single")
    )
    parser.add_argument("--bit-order", default=DEFAULT_BIT_ORDER)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5210)
    parser.add_argument("--gaussian-log", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.shots is not None and arguments.shots <= 0:
        raise ValueError("--shots must be positive.")
    if arguments.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative.")

    summary = analyze(
        geometry_id=arguments.geometry_id,
        input_root=arguments.input_dir,
        circuit_root=arguments.circuit_dir,
        reference_root=arguments.reference_dir,
        output_dir=arguments.output_dir,
        bit_order=arguments.bit_order,
        explicit_shots=arguments.shots,
        bootstrap_repetitions=arguments.bootstrap,
        seed=arguments.seed,
        gaussian_log=arguments.gaussian_log,
    )
    print(
        json.dumps(
            {
                "geometry_id": summary["geometry_id"],
                "full_pyscf_rhf_hartree": summary[
                    "energies_hartree"
                ]["full_pyscf_rhf"],
                "active_rhf_hartree": summary[
                    "energies_hartree"
                ]["frozen_core_active_rhf"],
                "rank6_projected_hartree": summary[
                    "energies_hartree"
                ]["measured_rank6_projected"],
                "rank6_measurement_error_millihartree": 1000.0
                * summary["error_decomposition"][
                    "quantum_measurement_error_hartree"
                ],
                "one_spin_slater_fidelity": summary["fidelity"][
                    "one_spin_slater"
                ],
                "saved": str(arguments.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
