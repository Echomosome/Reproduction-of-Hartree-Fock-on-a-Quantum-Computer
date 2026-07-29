#!/usr/bin/env python3
"""Extract Gaussian RHF energies and compare them with a saved reference."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from h2_core import load_reference, parse_gaussian_scf_energies


def bond_length_from_name(path: Path) -> float | None:
    match = re.search(r"R(\d+)p(\d+)", path.stem, re.IGNORECASE)
    if not match:
        return None
    return float(f"{int(match.group(1))}.{match.group(2)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("gaussian_energy_comparison.csv")
    )
    arguments = parser.parse_args()

    reference_energy: float | None = None
    if arguments.reference is not None:
        reference_energy = float(
            load_reference(arguments.reference)["rhf_energy"]
        )

    rows: list[dict] = []
    for path in arguments.logs:
        energies = parse_gaussian_scf_energies(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if not energies:
            raise ValueError(f"No SCF Done energy found in {path}.")
        energy = energies[-1]
        rows.append(
            {
                "log_file": str(path),
                "bond_length_angstrom_from_filename": bond_length_from_name(
                    path
                ),
                "gaussian_final_scf_energy_hartree": energy,
                "reference_energy_hartree": reference_energy,
                "gaussian_minus_reference_hartree": (
                    None
                    if reference_energy is None
                    else energy - reference_energy
                ),
                "classical_crosscheck_pass_1_microhartree": (
                    None
                    if reference_energy is None
                    else abs(energy - reference_energy) <= 1e-6
                ),
            }
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
