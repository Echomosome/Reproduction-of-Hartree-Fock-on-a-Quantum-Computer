#!/usr/bin/env python3
"""Tabulate how geometry changes every downstream molecular/circuit parameter."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from diazene_core import geometry_metrics, load_geometries, load_reference


def z_angles(path: Path) -> dict[tuple[int, int, int, int], float]:
    result: dict[tuple[int, int, int, int], float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] != "z_diagonal":
                continue
            key = (
                int(row["parallel_layer"]),
                int(row["index_within_layer"]),
                int(row["first_mode"]),
                int(row["second_mode"]),
            )
            result[key] = float(row["givens_theta_rad"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometries",
        type=Path,
        default=Path("data/geometries_paper.json"),
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("reference")
    )
    parser.add_argument(
        "--circuit-dir", type=Path, default=Path("circuits")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/parameter_evolution.csv"),
    )
    arguments = parser.parse_args()

    geometries = sorted(
        load_geometries(arguments.geometries),
        key=lambda item: (item["pathway"], item["path_index"]),
    )
    previous: dict[str, dict] = {}
    rows: list[dict] = []
    for geometry in geometries:
        geometry_id = geometry["id"]
        reference = load_reference(
            arguments.reference_dir / f"{geometry_id}.npz"
        )
        metrics = geometry_metrics(geometry)
        angles = z_angles(
            arguments.circuit_dir
            / geometry_id
            / "givens_parameters.csv"
        )
        theta_values = np.asarray(list(angles.values()), dtype=float)
        old = previous.get(geometry["pathway"])
        if old is None:
            delta_h1 = delta_eri = delta_gamma = None
            theta_rms = theta_max = None
            theta_common = 0
            theta_unmatched = 0
        else:
            delta_h1 = float(
                np.linalg.norm(
                    reference["h1_active"] - old["reference"]["h1_active"]
                )
            )
            delta_eri = float(
                np.linalg.norm(
                    reference["eri_active"]
                    - old["reference"]["eri_active"]
                )
            )
            delta_gamma = float(
                np.linalg.norm(
                    reference["gamma_active_one_spin"]
                    - old["reference"]["gamma_active_one_spin"]
                )
            )
            common = sorted(set(angles) & set(old["angles"]))
            differences = np.asarray(
                [angles[key] - old["angles"][key] for key in common],
                dtype=float,
            )
            theta_common = len(common)
            theta_unmatched = len(set(angles) ^ set(old["angles"]))
            theta_rms = (
                None
                if not len(differences)
                else float(np.sqrt(np.mean(differences**2)))
            )
            theta_max = (
                None
                if not len(differences)
                else float(np.max(np.abs(differences)))
            )

        rows.append(
            {
                "geometry_id": geometry_id,
                "pathway": geometry["pathway"],
                "path_index": geometry["path_index"],
                "reaction_coordinate_deg": geometry[
                    "reaction_coordinate_deg"
                ],
                "r_h1_n1_angstrom": metrics["r_h1_n1_angstrom"],
                "r_n1_n2_angstrom": metrics["r_n1_n2_angstrom"],
                "r_n2_h2_angstrom": metrics["r_n2_h2_angstrom"],
                "angle_h1_n1_n2_deg": metrics[
                    "angle_h1_n1_n2_deg"
                ],
                "angle_n1_n2_h2_deg": metrics[
                    "angle_n1_n2_h2_deg"
                ],
                "dihedral_0_to_360_deg": metrics[
                    "dihedral_h1_n1_n2_h2_0_to_360_deg"
                ],
                "nuclear_repulsion_hartree": float(
                    reference["nuclear_repulsion_hartree"]
                ),
                "frozen_constant_offset_hartree": float(
                    reference["constant_offset_hartree"]
                ),
                "full_rhf_energy_hartree": float(
                    reference["full_rhf_energy_hartree"]
                ),
                "active_rhf_energy_hartree": float(
                    reference["active_rhf_energy_hartree"]
                ),
                "frozen_core_bias_millihartree": 1000.0
                * (
                    float(reference["active_rhf_energy_hartree"])
                    - float(reference["full_rhf_energy_hartree"])
                ),
                "z_setting_givens_count": len(theta_values),
                "z_setting_theta_min_rad": float(np.min(theta_values)),
                "z_setting_theta_max_rad": float(np.max(theta_values)),
                "z_setting_theta_l2_norm_rad": float(
                    np.linalg.norm(theta_values)
                ),
                "delta_h1_active_frobenius_from_previous": delta_h1,
                "delta_eri_active_frobenius_from_previous": delta_eri,
                "delta_gamma_frobenius_from_previous": delta_gamma,
                "delta_theta_common_count": theta_common,
                "delta_theta_unmatched_operation_count": theta_unmatched,
                "delta_theta_rms_common_rad": theta_rms,
                "delta_theta_max_abs_common_rad": theta_max,
            }
        )
        previous[geometry["pathway"]] = {
            "reference": reference,
            "angles": angles,
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {arguments.output}.")


if __name__ == "__main__":
    main()
