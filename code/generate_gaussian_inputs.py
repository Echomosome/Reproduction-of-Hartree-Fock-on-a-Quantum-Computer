#!/usr/bin/env python3
"""Generate Gaussian RHF/STO-3G inputs for all published geometries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from diazene_core import load_geometries


ROUTE_SINGLE_POINT = (
    "#p RHF/STO-3G SCF=(Tight,XQC,MaxCycle=512) "
    "NoSymm Pop=Full"
)
ROUTE_RELAXED_SCAN = (
    "#p RHF/STO-3G Opt=(ModRedundant,CalcFC,MaxCycles=200) "
    "SCF=(Tight,XQC,MaxCycle=512) NoSymm"
)


def gaussian_input(
    geometry: dict,
    memory: str,
    nproc: int,
) -> str:
    lines = [
        f"%chk={geometry['id']}.chk",
        f"%mem={memory}",
        f"%nprocshared={nproc}",
        ROUTE_SINGLE_POINT,
        "",
        (
            f"Diazene {geometry['id']} | "
            f"{geometry['pathway']} "
            f"{geometry['reaction_coordinate_deg']:.3f} deg"
        ),
        "",
        "0 1",
    ]
    for atom in geometry["atoms"]:
        x, y, z = atom["xyz_angstrom"]
        lines.append(
            f"{atom['element']:<2s} {x: .8f} {y: .8f} {z: .8f}"
        )
    lines.extend(["", ""])
    return "\n".join(lines)


def relaxed_out_of_plane_scan(
    geometry: dict,
    memory: str,
    nproc: int,
) -> str:
    lines = [
        "%chk=diazene_out_of_plane_relaxed_scan.chk",
        f"%mem={memory}",
        f"%nprocshared={nproc}",
        ROUTE_RELAXED_SCAN,
        "",
        (
            "Diazene independent Gaussian relaxed HNNH scan; "
            "not the paper's tabulated nine-point path"
        ),
        "",
        "0 1",
    ]
    for atom in geometry["atoms"]:
        x, y, z = atom["xyz_angstrom"]
        lines.append(
            f"{atom['element']:<2s} {x: .8f} {y: .8f} {z: .8f}"
        )
    lines.extend(
        [
            "",
            "D 1 2 3 4 S 12 15.0",
            "",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometries",
        type=Path,
        default=Path("data/geometries_paper.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("gaussian")
    )
    parser.add_argument("--memory", default="4GB")
    parser.add_argument("--nproc", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.nproc <= 0:
        raise ValueError("--nproc must be positive.")

    geometries = load_geometries(arguments.geometries)
    input_dir = arguments.output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for geometry in geometries:
        path = input_dir / f"{geometry['id']}.gjf"
        path.write_text(
            gaussian_input(
                geometry, arguments.memory, arguments.nproc
            ),
            encoding="utf-8",
        )
    first_out_of_plane = next(
        geometry
        for geometry in geometries
        if geometry["id"] == "oop_003p157"
    )
    scan_dir = arguments.output_dir / "optional_relaxed_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "diazene_out_of_plane_relaxed_scan.gjf").write_text(
        relaxed_out_of_plane_scan(
            first_out_of_plane, arguments.memory, arguments.nproc
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(geometries)} single-point inputs to {input_dir} "
        "and one optional relaxed scan."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
