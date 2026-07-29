#!/usr/bin/env python3
"""Generate four H4 measurement circuits in native SQISWAP/RZ syntax."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from h4_core import (
    N_MODES,
    analyzer_matrix,
    gamma_from_native_circuit,
    native_givens_lines,
    nearest_neighbor_givens_decomposition,
    parse_native_circuit,
    write_json,
)


SETTINGS: list[dict] = [
    {
        "setting": "z_diagonal",
        "stem": "H4-00-z-diagonal",
        "kind": "diagonal",
        "pairs": None,
    },
    {
        "setting": "matching_01_23",
        "stem": "H4-01-matching-01-23",
        "kind": "offdiagonal_matching",
        "pairs": [(0, 1), (2, 3)],
    },
    {
        "setting": "matching_02_13",
        "stem": "H4-02-matching-02-13",
        "kind": "offdiagonal_matching",
        "pairs": [(0, 2), (1, 3)],
    },
    {
        "setting": "matching_03_12",
        "stem": "H4-03-matching-03-12",
        "kind": "offdiagonal_matching",
        "pairs": [(0, 3), (1, 2)],
    },
]


def full_originir(body: list[str], n_qubits: int = N_MODES) -> str:
    lines = [f"QINIT {n_qubits}", f"CREG {n_qubits}", *body]
    lines.extend(
        f"MEASURE q[{qubit}],c[{qubit}]" for qubit in range(n_qubits)
    )
    return "\n".join(lines) + "\n"


def save_text(path: Path, lines: list[str] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def circuit_body_for_orthogonal(orthogonal: np.ndarray) -> tuple[list[str], list]:
    rotations = nearest_neighbor_givens_decomposition(orthogonal)
    body = ["X q[0]", "X q[1]"]
    for first, second, angle in rotations:
        body.extend(native_givens_lines(first, second, angle))
    return body, rotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reference/h4_r1.3000_sto3g_reference.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("circuits")
    )
    arguments = parser.parse_args()

    with np.load(arguments.reference, allow_pickle=False) as reference:
        orbital_rotation = np.asarray(
            reference["mo_coeff_orth"], dtype=float
        )
        target_gamma = np.asarray(
            reference["gamma_one_spin_orth"], dtype=float
        )
        spacing = float(reference["spacing_angstrom"])

    gate_dir = arguments.output_dir / "gate_body_radians"
    originir_dir = arguments.output_dir / "originir_full"
    manifest_circuits: list[dict] = []
    map_rows: list[dict] = []

    for setting_specification in SETTINGS:
        pairs = setting_specification["pairs"]
        if pairs is None:
            analyzer = np.eye(N_MODES)
            measurement_map: list[dict[str, int]] = []
        else:
            analyzer, measurement_map = analyzer_matrix(pairs)
        total_rotation = analyzer @ orbital_rotation
        body, rotations = circuit_body_for_orthogonal(total_rotation)
        gate_path = gate_dir / f"{setting_specification['stem']}.txt"
        originir_path = (
            originir_dir / f"{setting_specification['stem']}.originir"
        )
        save_text(gate_path, body)
        save_text(originir_path, full_originir(body))

        simulated_gamma = gamma_from_native_circuit(
            parse_native_circuit("\n".join(body))
        )
        expected_gamma = analyzer @ target_gamma @ analyzer.T
        closure_error = float(
            np.max(np.abs(simulated_gamma - expected_gamma))
        )
        if closure_error > 2e-10:
            raise RuntimeError(
                f"{setting_specification['setting']} circuit closure failed: "
                f"{closure_error:.3e}."
            )

        n_sqiswap = sum(line.startswith("SQISWAP") for line in body)
        n_rz = sum(line.startswith("RZ") for line in body)
        entry = {
            "setting": setting_specification["setting"],
            "kind": setting_specification["kind"],
            "gate_body_file": str(
                gate_path.relative_to(arguments.output_dir)
            ),
            "originir_file": str(
                originir_path.relative_to(arguments.output_dir)
            ),
            "measurement_map": measurement_map,
            "givens_rotations": [
                {
                    "first": first,
                    "second": second,
                    "theta_radians": angle,
                    "theta_over_pi": angle / np.pi,
                }
                for first, second, angle in rotations
            ],
            "gate_counts": {
                "X": 2,
                "SQISWAP": n_sqiswap,
                "RZ": n_rz,
            },
            "max_exact_gamma_closure_error": closure_error,
        }
        manifest_circuits.append(entry)
        for mapping in measurement_map:
            map_rows.append(
                {
                    "setting": setting_specification["setting"],
                    **mapping,
                    "formula": "(<n_left>-<n_right>)/2",
                }
            )

    manifest = {
        "model": "linear H4, RHF/STO-3G, Lowdin orthogonalized AO basis",
        "spacing_angstrom": spacing,
        "encoding": {
            "qubits": 4,
            "particles": 2,
            "meaning": "one spin sector; closed-shell RHF duplicates alpha and beta",
            "initial_occupied_modes": [0, 1],
            "returned_bitstring_order": "q[3]q[2]q[1]q[0]",
        },
        "angle_units": "radians",
        "sqiswap_convention": "+i",
        "compilation": (
            "Each state-preparation-plus-analyzer matrix is decomposed into "
            "nearest-neighbour real Givens rotations, then each Givens is "
            "compiled to two SQISWAP gates and RZ phases."
        ),
        "circuits": manifest_circuits,
    }
    write_json(arguments.output_dir / "manifest.json", manifest)

    measurement_map_path = arguments.output_dir / "measurement_map.csv"
    measurement_map_path.parent.mkdir(parents=True, exist_ok=True)
    with measurement_map_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting",
                "physical_left",
                "physical_right",
                "logical_i",
                "logical_j",
                "formula",
            ],
        )
        writer.writeheader()
        writer.writerows(map_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
