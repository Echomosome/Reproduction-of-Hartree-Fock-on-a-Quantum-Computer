#!/usr/bin/env python3
"""Generate 10-setting diazene 1-RDM circuits in SQISWAP/RZ syntax."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from diazene_core import (
    DEFAULT_BIT_ORDER,
    N_MODES,
    N_PARTICLES_PER_SPIN,
    analyzer_matrix,
    complete_graph_matchings,
    gamma_from_native_circuit,
    load_reference,
    native_givens_lines,
    parse_native_circuit,
    slater_givens_decomposition,
    write_json,
)


def full_originir(body: list[str]) -> str:
    lines = [f"QINIT {N_MODES}", f"CREG {N_MODES}", *body]
    lines.extend(
        f"MEASURE q[{qubit}],c[{qubit}]"
        for qubit in range(N_MODES)
    )
    return "\n".join(lines) + "\n"


def save_text(path: Path, lines: list[str] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def setting_specs() -> list[dict]:
    result = [
        {
            "setting": "z_diagonal",
            "index": 0,
            "kind": "diagonal",
            "pairs": None,
        }
    ]
    for index, matching in enumerate(
        complete_graph_matchings(N_MODES), start=1
    ):
        pair_label = "_".join(f"{i}{j}" for i, j in matching)
        result.append(
            {
                "setting": f"matching_{index:02d}_{pair_label}",
                "index": index,
                "kind": "offdiagonal_matching",
                "pairs": matching,
            }
        )
    return result


def circuit_for_occupied_orbitals(
    occupied_orbitals: np.ndarray,
) -> tuple[list[str], list[dict]]:
    layers = slater_givens_decomposition(occupied_orbitals)
    body = [
        f"X q[{mode}]" for mode in range(N_PARTICLES_PER_SPIN)
    ]
    parameter_rows: list[dict] = []
    global_index = 0
    for layer_index, layer in enumerate(layers):
        for operation_index, (first, second, theta) in enumerate(layer):
            body.extend(native_givens_lines(first, second, theta))
            parameter_rows.append(
                {
                    "givens_index": global_index,
                    "parallel_layer": layer_index,
                    "index_within_layer": operation_index,
                    "first_mode": first,
                    "second_mode": second,
                    "givens_theta_rad": theta,
                    "rz_first_pi_plus_theta_rad": math.pi + theta,
                    "rz_second_minus_theta_rad": -theta,
                    "rz_first_final_pi_rad": math.pi,
                }
            )
            global_index += 1
    return body, parameter_rows


def build_geometry_circuits(
    geometry_id: str,
    reference_path: Path,
    output_dir: Path,
) -> dict:
    reference = load_reference(reference_path)
    occupied = np.asarray(
        reference["occupied_orbitals_active"], dtype=float
    )
    target_gamma = np.asarray(
        reference["gamma_active_one_spin"], dtype=float
    )
    pathway = str(reference["pathway"])
    coordinate = float(reference["reaction_coordinate_deg"])

    gate_dir = output_dir / "gate_body_radians"
    originir_dir = output_dir / "originir_full"
    circuits: list[dict] = []
    all_parameter_rows: list[dict] = []
    measurement_rows: list[dict] = []

    for specification in setting_specs():
        pairs = specification["pairs"]
        if pairs is None:
            analyzer = np.eye(N_MODES)
            measurement_map: list[dict[str, int]] = []
        else:
            analyzer, measurement_map = analyzer_matrix(pairs, N_MODES)
        measured_occupied = analyzer @ occupied
        body, parameter_rows = circuit_for_occupied_orbitals(
            measured_occupied
        )
        stem = (
            f"Diazene-{geometry_id}-"
            f"{specification['index']:02d}-{specification['setting']}"
        )
        gate_path = gate_dir / f"{stem}.txt"
        originir_path = originir_dir / f"{stem}.originir"
        save_text(gate_path, body)
        save_text(originir_path, full_originir(body))

        parsed = parse_native_circuit(
            gate_path.read_text(encoding="utf-8")
        )
        circuit_gamma = gamma_from_native_circuit(parsed)
        expected_gamma = analyzer @ target_gamma @ analyzer.T
        closure_error = float(
            np.max(np.abs(circuit_gamma - expected_gamma))
        )
        if closure_error > 1e-8:
            raise RuntimeError(
                f"{geometry_id}/{specification['setting']}: "
                f"circuit closure error {closure_error:.3e}."
            )

        for row in parameter_rows:
            row["setting"] = specification["setting"]
            all_parameter_rows.append(row)
        for mapping in measurement_map:
            measurement_rows.append(
                {"setting": specification["setting"], **mapping}
            )
        circuits.append(
            {
                "setting": specification["setting"],
                "setting_index": specification["index"],
                "kind": specification["kind"],
                "logical_pairs": pairs,
                "measurement_map": measurement_map,
                "gate_body_file": str(
                    gate_path.relative_to(output_dir)
                ),
                "originir_file": str(
                    originir_path.relative_to(output_dir)
                ),
                "givens_rotation_count": len(parameter_rows),
                "sqiswap_count": 2 * len(parameter_rows),
                "rz_count": 3 * len(parameter_rows),
                "x_count": N_PARTICLES_PER_SPIN,
                "max_abs_gamma_closure_error": closure_error,
            }
        )

    manifest = {
        "schema_version": 1,
        "geometry_id": geometry_id,
        "pathway": pathway,
        "reaction_coordinate_deg": coordinate,
        "model": {
            "method": "RHF",
            "basis": "STO-3G",
            "preliminary_scf_cycles": 2,
            "frozen_spatial_orbitals": 2,
            "active_spatial_modes": N_MODES,
            "particles_per_spin": N_PARTICLES_PER_SPIN,
            "encoded_spin_sector": "one of two identical RHF sectors",
        },
        "measurement": {
            "real_one_rdm": True,
            "number_of_settings": len(circuits),
            "diagonal_settings": 1,
            "offdiagonal_matching_settings": N_MODES - 1,
            "independent_diagonal_elements": N_MODES,
            "independent_offdiagonal_elements": (
                N_MODES * (N_MODES - 1) // 2
            ),
            "reconstruction_formula": (
                "gamma_ij=(<n_left>-<n_right>)/2"
            ),
        },
        "gate_conventions": {
            "bitstring_order": DEFAULT_BIT_ORDER,
            "rz_unit": "radian",
            "sqiswap_one_particle_block": (
                "1/sqrt(2) * [[1, i], [i, 1]]"
            ),
            "givens_matrix": (
                "[[cos(theta),-sin(theta)],"
                "[sin(theta),cos(theta)]]"
            ),
            "givens_native_decomposition": [
                "SQISWAP q[second],q[first]",
                "RZ q[first],(pi+theta)",
                "RZ q[second],(-theta)",
                "SQISWAP q[second],q[first]",
                "RZ q[first],(pi)",
            ],
        },
        "circuits": circuits,
    }
    write_json(output_dir / "manifest.json", manifest)

    with (output_dir / "givens_parameters.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(all_parameter_rows[0])
        )
        writer.writeheader()
        writer.writerows(all_parameter_rows)
    with (output_dir / "measurement_map.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(measurement_rows[0])
        )
        writer.writeheader()
        writer.writerows(measurement_rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("reference")
    )
    parser.add_argument(
        "--geometry-id",
        default=None,
        help="Generate one geometry; default uses every reference NPZ.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("circuits")
    )
    arguments = parser.parse_args()

    if arguments.geometry_id is None:
        reference_paths = sorted(
            path
            for path in arguments.reference_dir.glob("*.npz")
            if path.stem not in {"reference_summary"}
        )
    else:
        reference_paths = [
            arguments.reference_dir / f"{arguments.geometry_id}.npz"
        ]
    if not reference_paths:
        raise FileNotFoundError("No reference NPZ files found.")

    top_level: list[dict] = []
    for reference_path in reference_paths:
        if not reference_path.exists():
            raise FileNotFoundError(reference_path)
        geometry_id = reference_path.stem
        manifest = build_geometry_circuits(
            geometry_id,
            reference_path,
            arguments.output_dir / geometry_id,
        )
        max_closure = max(
            circuit["max_abs_gamma_closure_error"]
            for circuit in manifest["circuits"]
        )
        rotations = [
            circuit["givens_rotation_count"]
            for circuit in manifest["circuits"]
        ]
        top_level.append(
            {
                "geometry_id": geometry_id,
                "pathway": manifest["pathway"],
                "reaction_coordinate_deg": manifest[
                    "reaction_coordinate_deg"
                ],
                "number_of_settings": len(manifest["circuits"]),
                "min_givens_rotations": min(rotations),
                "max_givens_rotations": max(rotations),
                "max_abs_gamma_closure_error": max_closure,
                "manifest": f"{geometry_id}/manifest.json",
            }
        )
        print(
            f"{geometry_id}: {len(manifest['circuits'])} settings, "
            f"{min(rotations)}-{max(rotations)} Givens/setting, "
            f"closure={max_closure:.3e}"
        )
    write_json(
        arguments.output_dir / "scan_manifest.json",
        {
            "schema_version": 1,
            "description": (
                "Diazene 10-mode/6-particle RHF measurement circuits"
            ),
            "geometries": top_level,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
