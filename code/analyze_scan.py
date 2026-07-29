#!/usr/bin/env python3
"""Analyze all diazene geometries and plot both isomerization pathways."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_geometry import analyze
from diazene_core import DEFAULT_BIT_ORDER, load_geometries, write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pathway_statistics(
    rows: list[dict], energy_key: str
) -> dict[str, dict | float | None]:
    result: dict[str, dict | float | None] = {}
    maxima: dict[str, float] = {}
    for pathway in ("out_of_plane", "in_plane"):
        selected = sorted(
            [row for row in rows if row["pathway"] == pathway],
            key=lambda row: row["path_index"],
        )
        valid = [
            row for row in selected if row.get(energy_key) is not None
        ]
        if not valid:
            result[pathway] = None
            continue
        maximum = max(valid, key=lambda row: row[energy_key])
        endpoint_minimum = min(
            valid[0][energy_key], valid[-1][energy_key]
        )
        maxima[pathway] = maximum[energy_key]
        result[pathway] = {
            "maximum_geometry_id": maximum["geometry_id"],
            "maximum_coordinate_deg": maximum["reaction_coordinate_deg"],
            "maximum_energy_hartree": maximum[energy_key],
            "barrier_from_lower_endpoint_millihartree": (
                1000.0 * (maximum[energy_key] - endpoint_minimum)
            ),
        }
    if set(maxima) == {"out_of_plane", "in_plane"}:
        result["in_plane_minus_out_of_plane_ts_gap_millihartree"] = (
            1000.0
            * (maxima["in_plane"] - maxima["out_of_plane"])
        )
    else:
        result["in_plane_minus_out_of_plane_ts_gap_millihartree"] = None
    return result


def plot_curves(path: Path, rows: list[dict]) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.5), constrained_layout=True
    )
    colors = {
        "full_pyscf_rhf_hartree": "#30343B",
        "active_rhf_hartree": "#4C78A8",
        "quantum_rank6_hartree": "#F28E2B",
        "gaussian_rhf_hartree": "#59A14F",
    }
    labels = {
        "full_pyscf_rhf_hartree": "Full 12-orbital PySCF RHF",
        "active_rhf_hartree": "10-mode frozen-core RHF",
        "quantum_rank6_hartree": "Quantum rank-6 energy",
        "gaussian_rhf_hartree": "Gaussian RHF",
    }
    for axis, pathway, title in zip(
        axes,
        ("out_of_plane", "in_plane"),
        ("Out-of-plane HNNH torsion", "In-plane H rotation"),
    ):
        selected = sorted(
            [row for row in rows if row["pathway"] == pathway],
            key=lambda row: row["path_index"],
        )
        coordinates = np.asarray(
            [row["reaction_coordinate_deg"] for row in selected],
            dtype=float,
        )
        full_values = np.asarray(
            [row["full_pyscf_rhf_hartree"] for row in selected]
        )
        zero = float(np.min(full_values))
        for key in (
            "full_pyscf_rhf_hartree",
            "active_rhf_hartree",
            "quantum_rank6_hartree",
            "gaussian_rhf_hartree",
        ):
            if any(row.get(key) is None for row in selected):
                continue
            values = 1000.0 * (
                np.asarray([row[key] for row in selected], dtype=float)
                - zero
            )
            axis.plot(
                coordinates,
                values,
                marker="o",
                linewidth=1.7,
                markersize=4,
                color=colors[key],
                label=labels[key],
            )
        axis.set_title(title)
        axis.set_xlabel("Published reaction coordinate (deg)")
        axis.set_ylabel("Relative energy (mHa)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_errors(path: Path, rows: list[dict]) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.2), constrained_layout=True
    )
    for axis, pathway, title in zip(
        axes,
        ("out_of_plane", "in_plane"),
        ("Out-of-plane", "In-plane"),
    ):
        selected = sorted(
            [row for row in rows if row["pathway"] == pathway],
            key=lambda row: row["path_index"],
        )
        x = [row["reaction_coordinate_deg"] for row in selected]
        freeze = [
            row["frozen_core_bias_millihartree"] for row in selected
        ]
        measurement = [
            row["measurement_error_vs_active_millihartree"]
            for row in selected
        ]
        end_to_end = [
            row["end_to_end_error_vs_full_millihartree"]
            for row in selected
        ]
        axis.plot(x, freeze, marker="o", label="Frozen-core bias")
        axis.plot(x, measurement, marker="o", label="Measurement error")
        axis.plot(x, end_to_end, marker="o", label="End-to-end error")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhspan(-1.593601, 1.593601, color="#59A14F", alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel("Published reaction coordinate (deg)")
        axis.set_ylabel("Energy error (mHa)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometries",
        type=Path,
        default=Path("data/geometries_paper.json"),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--circuit-dir", type=Path, default=Path("circuits")
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("reference")
    )
    parser.add_argument(
        "--gaussian-log-dir", type=Path, default=None
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/scan")
    )
    parser.add_argument("--bit-order", default=DEFAULT_BIT_ORDER)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5210)
    arguments = parser.parse_args()
    if arguments.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative.")

    geometries = sorted(
        load_geometries(arguments.geometries),
        key=lambda item: (item["pathway"], item["path_index"]),
    )
    rows: list[dict] = []
    for geometry in geometries:
        geometry_id = geometry["id"]
        gaussian_log = None
        if arguments.gaussian_log_dir is not None:
            candidate = arguments.gaussian_log_dir / f"{geometry_id}.log"
            if candidate.exists():
                gaussian_log = candidate
        summary = analyze(
            geometry_id=geometry_id,
            input_root=arguments.input_dir,
            circuit_root=arguments.circuit_dir,
            reference_root=arguments.reference_dir,
            output_dir=arguments.output_dir / "geometries" / geometry_id,
            bit_order=arguments.bit_order,
            explicit_shots=arguments.shots,
            bootstrap_repetitions=arguments.bootstrap,
            seed=arguments.seed + int(geometry["path_index"]),
            gaussian_log=gaussian_log,
        )
        energies = summary["energies_hartree"]
        errors = summary["error_decomposition"]
        row = {
            "geometry_id": geometry_id,
            "pathway": geometry["pathway"],
            "path_index": geometry["path_index"],
            "reaction_coordinate_deg": geometry[
                "reaction_coordinate_deg"
            ],
            "full_pyscf_rhf_hartree": energies["full_pyscf_rhf"],
            "active_rhf_hartree": energies[
                "frozen_core_active_rhf"
            ],
            "quantum_raw_hartree": energies["measured_raw"],
            "quantum_rank6_hartree": energies[
                "measured_rank6_projected"
            ],
            "gaussian_rhf_hartree": energies["gaussian_rhf"],
            "frozen_core_bias_millihartree": 1000.0
            * errors["frozen_core_bias_hartree"],
            "measurement_error_vs_active_millihartree": 1000.0
            * errors["quantum_measurement_error_hartree"],
            "end_to_end_error_vs_full_millihartree": 1000.0
            * (
                energies["measured_rank6_projected"]
                - energies["full_pyscf_rhf"]
            ),
            "one_spin_slater_fidelity": summary["fidelity"][
                "one_spin_slater"
            ],
            "minimum_particle_number_pass_probability": summary[
                "minimum_particle_number_pass_probability"
            ],
            "mean_tv_distance_vs_exact_circuit": summary[
                "mean_tv_distance_vs_exact_circuit"
            ],
        }
        rows.append(row)
        print(
            f"{geometry_id}: E_Q={row['quantum_rank6_hartree']:.9f} Ha, "
            f"measurement error="
            f"{row['measurement_error_vs_active_millihartree']:+.3f} mHa"
        )

    statistics = {
        "full_pyscf_rhf": pathway_statistics(
            rows, "full_pyscf_rhf_hartree"
        ),
        "frozen_core_active_rhf": pathway_statistics(
            rows, "active_rhf_hartree"
        ),
        "quantum_rank6": pathway_statistics(
            rows, "quantum_rank6_hartree"
        ),
        "gaussian_rhf": pathway_statistics(
            rows, "gaussian_rhf_hartree"
        ),
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output_dir / "energy_curve.csv", rows)
    write_json(
        arguments.output_dir / "scan_summary.json",
        {
            "input_directory": str(arguments.input_dir),
            "number_of_geometries": len(rows),
            "rows": rows,
            "pathway_statistics": statistics,
        },
    )
    plot_curves(arguments.output_dir / "energy_curves.png", rows)
    plot_errors(arguments.output_dir / "error_decomposition.png", rows)
    print(
        json.dumps(
            {
                "saved": str(arguments.output_dir),
                "full_rhf_ts_gap_millihartree": statistics[
                    "full_pyscf_rhf"
                ]["in_plane_minus_out_of_plane_ts_gap_millihartree"],
                "quantum_ts_gap_millihartree": statistics[
                    "quantum_rank6"
                ]["in_plane_minus_out_of_plane_ts_gap_millihartree"],
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
