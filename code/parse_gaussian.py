#!/usr/bin/env python3
"""Parse Gaussian SCF energies and compare them with PySCF references."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from diazene_core import parse_gaussian_scf_energies, write_json


def reference_lookup(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["geometry_id"]: row for row in payload.get("geometries", [])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs",
        type=Path,
        nargs="+",
        help="Gaussian .log files or directories containing .log files.",
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path("reference/reference_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("gaussian/gaussian_energy_comparison.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("gaussian/gaussian_energy_comparison.json"),
    )
    arguments = parser.parse_args()

    paths: list[Path] = []
    for item in arguments.logs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.log")))
        else:
            paths.append(item)
    if not paths:
        raise FileNotFoundError("No Gaussian log files found.")
    lookup = reference_lookup(
        arguments.reference_summary
        if arguments.reference_summary.exists()
        else None
    )

    rows: list[dict] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        energies = parse_gaussian_scf_energies(text)
        if not energies:
            raise ValueError(f"{path}: no 'SCF Done' energy found.")
        geometry_id = path.stem
        reference = lookup.get(geometry_id)
        gaussian_energy = energies[-1]
        full_pyscf = (
            None
            if reference is None
            else reference["full_rhf_energy_hartree"]
        )
        rows.append(
            {
                "geometry_id": geometry_id,
                "log_file": str(path),
                "normal_termination": (
                    "Normal termination of Gaussian" in text
                ),
                "scf_records": len(energies),
                "gaussian_final_rhf_hartree": gaussian_energy,
                "full_pyscf_rhf_hartree": full_pyscf,
                "gaussian_minus_pyscf_microhartree": (
                    None
                    if full_pyscf is None
                    else 1.0e6 * (gaussian_energy - full_pyscf)
                ),
                "pyscf_minus_gaussian_microhartree": (
                    None
                    if full_pyscf is None
                    else 1.0e6 * (full_pyscf - gaussian_energy)
                ),
            }
        )
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(arguments.output_json, rows)
    print(
        json.dumps(
            {
                "parsed_logs": len(rows),
                "all_normal_termination": all(
                    row["normal_termination"] for row in rows
                ),
                "saved_csv": str(arguments.output_csv),
                "saved_json": str(arguments.output_json),
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
