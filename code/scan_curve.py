#!/usr/bin/env python3
"""Compute an H2 RHF/STO-3G bond scan and matching Gaussian inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_reference import build_reference
from generate_gaussian_inputs import gaussian_input, parse_lengths, tag_for_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        default="0.30:3.00:0.10",
        help="Comma list or start:stop:step in Angstrom.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_scan")
    )
    parser.add_argument(
        "--gaussian-input-dir",
        type=Path,
        default=Path("gaussian/scan_inputs"),
    )
    parser.add_argument("--memory", default="2GB")
    parser.add_argument("--nproc", type=int, default=4)
    arguments = parser.parse_args()

    lengths = parse_lengths(arguments.lengths)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    arguments.gaussian_input_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    for length in lengths:
        reference = build_reference(length)
        rows.append(
            {
                "bond_length_angstrom": float(length),
                "pyscf_rhf_energy_hartree": float(
                    reference["rhf_energy"]
                ),
                "nuclear_repulsion_hartree": float(
                    reference["nuclear_repulsion"]
                ),
                "givens_theta_radians": float(
                    reference["givens_theta_radians"]
                ),
                "givens_theta_over_pi": float(
                    reference["givens_theta_over_pi"]
                ),
            }
        )
        tag = tag_for_length(length)
        (arguments.gaussian_input_dir / f"h2_R{tag}_RHF_STO3G.gjf").write_text(
            gaussian_input(length, arguments.memory, arguments.nproc),
            encoding="utf-8",
        )

    csv_path = arguments.output_dir / "h2_rhf_sto3g_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    minimum = min(rows, key=lambda row: row["pyscf_rhf_energy_hartree"])
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.plot(
        [row["bond_length_angstrom"] for row in rows],
        [row["pyscf_rhf_energy_hartree"] for row in rows],
        marker="o",
        markersize=3.2,
        linewidth=1.5,
        label="PySCF RHF/STO-3G",
    )
    axis.scatter(
        [minimum["bond_length_angstrom"]],
        [minimum["pyscf_rhf_energy_hartree"]],
        color="#E15759",
        zorder=4,
        label=(
            f"scan minimum: R={minimum['bond_length_angstrom']:.2f} Å"
        ),
    )
    axis.set_xlabel("H–H bond length R (Å)")
    axis.set_ylabel("RHF total energy (Ha)")
    axis.set_title("H₂ RHF/STO-3G potential-energy curve")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(
        arguments.output_dir / "h2_rhf_sto3g_curve.png", dpi=180
    )
    plt.close(figure)

    summary = {
        "points": len(rows),
        "scan_minimum": minimum,
        "important_symmetry_result": (
            "In the centered homonuclear H2/STO-3G Lowdin basis, theta stays "
            "pi/4; the energy curve changes because the molecular integrals "
            "and nuclear repulsion change."
        ),
        "csv": str(csv_path),
        "gaussian_input_dir": str(arguments.gaussian_input_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
