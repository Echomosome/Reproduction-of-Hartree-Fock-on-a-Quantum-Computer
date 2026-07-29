#!/usr/bin/env python3
"""Generate Gaussian RHF/STO-3G input files for linear H4."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SCAN = [0.5, 0.9, 1.3, 1.7, 2.1, 2.5]


def filename_for_spacing(spacing: float) -> str:
    label = f"{spacing:.4f}".replace(".", "p")
    return f"h4_R{label}_RHF_STO3G.gjf"


def gaussian_input(spacing: float) -> str:
    coordinates = [
        (0.0, 0.0, (index - 1.5) * spacing) for index in range(4)
    ]
    checkpoint = filename_for_spacing(spacing).replace(".gjf", ".chk")
    lines = [
        f"%chk={checkpoint}",
        "%nprocshared=4",
        "%mem=2GB",
        "#p RHF/STO-3G SCF=(Tight,XQC,MaxCycle=512) NoSymm Pop=Full",
        "",
        f"Linear H4, nearest-neighbour spacing R={spacing:.4f} Angstrom",
        "",
        "0 1",
    ]
    lines.extend(
        f"H  {x: .10f}  {y: .10f}  {z: .10f}"
        for x, y, z in coordinates
    )
    return "\n".join(lines) + "\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=1.3)
    parser.add_argument(
        "--main-output-dir",
        type=Path,
        default=Path("gaussian/inputs"),
    )
    parser.add_argument(
        "--scan-output-dir",
        type=Path,
        default=Path("gaussian/scan_inputs"),
    )
    parser.add_argument(
        "--scan",
        type=float,
        nargs="*",
        default=DEFAULT_SCAN,
    )
    arguments = parser.parse_args()
    if arguments.spacing <= 0 or any(value <= 0 for value in arguments.scan):
        raise ValueError("All spacings must be positive.")

    arguments.main_output_dir.mkdir(parents=True, exist_ok=True)
    main_path = (
        arguments.main_output_dir
        / filename_for_spacing(arguments.spacing)
    )
    main_path.write_text(
        gaussian_input(arguments.spacing), encoding="utf-8"
    )

    arguments.scan_output_dir.mkdir(parents=True, exist_ok=True)
    for spacing in arguments.scan:
        path = arguments.scan_output_dir / filename_for_spacing(spacing)
        path.write_text(gaussian_input(spacing), encoding="utf-8")
    print(f"Main input: {main_path}")
    print(f"Scan inputs: {arguments.scan_output_dir}")


if __name__ == "__main__":
    main()
