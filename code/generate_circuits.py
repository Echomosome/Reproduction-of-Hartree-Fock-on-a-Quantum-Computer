#!/usr/bin/env python3
"""Generate H2 native gate-body and full OriginIR measurement circuits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from h2_core import (
    native_givens_lines,
    real_coherence_analyzer_lines,
    write_json,
)


def full_originir(body: list[str], n_qubits: int) -> str:
    lines = [f"QINIT {n_qubits}", f"CREG {n_qubits}", *body]
    lines.extend(
        f"MEASURE q[{qubit}],c[{qubit}]" for qubit in range(n_qubits)
    )
    return "\n".join(lines) + "\n"


def save_text(path: Path, lines: list[str] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reference/h2_r0.7414_sto3g_reference.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("circuits")
    )
    arguments = parser.parse_args()

    reference = np.load(arguments.reference)
    theta = float(reference["givens_theta_radians"])
    prep = ["X q[0]", *native_givens_lines(0, 1, theta)]
    diagonal = prep
    real_offdiagonal = [
        *prep,
        *real_coherence_analyzer_lines(0, 1),
    ]

    gate_body_dir = arguments.output_dir / "gate_body_radians"
    originir_dir = arguments.output_dir / "originir_full"
    save_text(gate_body_dir / "H2-00-z-diagonal.txt", diagonal)
    save_text(
        gate_body_dir / "H2-01-real-offdiagonal.txt",
        real_offdiagonal,
    )
    save_text(
        originir_dir / "H2-00-z-diagonal.originir",
        full_originir(diagonal, 2),
    )
    save_text(
        originir_dir / "H2-01-real-offdiagonal.originir",
        full_originir(real_offdiagonal, 2),
    )

    # Optional full four-qubit RHF determinant: q0/q1 are alpha; q2/q3 beta.
    full_prep = [
        "X q[0]",
        "X q[2]",
        *native_givens_lines(0, 1, theta),
        *native_givens_lines(2, 3, theta),
    ]
    full_real = [
        *full_prep,
        *real_coherence_analyzer_lines(0, 1),
        *real_coherence_analyzer_lines(2, 3),
    ]
    four_qubit_dir = arguments.output_dir / "four_qubit_optional"
    save_text(four_qubit_dir / "H2-full-00-z-diagonal.txt", full_prep)
    save_text(
        four_qubit_dir / "H2-full-01-real-offdiagonal.txt", full_real
    )

    manifest = {
        "model": "H2 RHF/STO-3G in the Lowdin orthogonalized AO basis",
        "primary_encoding": {
            "qubits": 2,
            "particles": 1,
            "meaning": "one spin sector; RHF duplicates it for alpha and beta",
            "bitstring_order": "q[1]q[0]",
        },
        "angle_units": "radians",
        "sqiswap_convention": "+i",
        "givens_theta_radians": theta,
        "givens_theta_over_pi": theta / math.pi,
        "circuits": [
            {
                "setting": "z_diagonal",
                "file": "gate_body_radians/H2-00-z-diagonal.txt",
                "extracts": ["D00", "D11"],
            },
            {
                "setting": "real_offdiagonal",
                "file": "gate_body_radians/H2-01-real-offdiagonal.txt",
                "extracts": ["Re(D01) = (n0 - n1) / 2"],
            },
        ],
        "optional_four_qubit_encoding": {
            "qubits": 4,
            "particles": 2,
            "pairs": {"alpha": [0, 1], "beta": [2, 3]},
        },
    }
    write_json(arguments.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
