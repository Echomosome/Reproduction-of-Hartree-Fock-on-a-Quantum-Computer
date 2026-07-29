#!/usr/bin/env python3
"""Compute and plot the linear-H4 RHF/STO-3G binding curve with PySCF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_reference import build_reference
from h4_core import parse_gaussian_scf_energies


DEFAULT_SCAN = [0.5, 0.9, 1.3, 1.7, 2.1, 2.5]


def gaussian_label(spacing: float) -> str:
    return f"{spacing:.4f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spacings", type=float, nargs="*", default=DEFAULT_SCAN
    )
    parser.add_argument(
        "--gaussian-log-dir", type=Path, default=None
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/scan")
    )
    arguments = parser.parse_args()
    if any(spacing <= 0 for spacing in arguments.spacings):
        raise ValueError("All spacings must be positive.")

    rows: list[dict] = []
    for spacing in arguments.spacings:
        reference = build_reference(spacing)
        gaussian_energy = None
        if arguments.gaussian_log_dir is not None:
            label = gaussian_label(spacing)
            candidates = [
                arguments.gaussian_log_dir
                / f"h4_R{label}_RHF_STO3G.log",
                arguments.gaussian_log_dir
                / f"h4_R{label}_RHF_STO3G.out",
            ]
            existing = next((path for path in candidates if path.exists()), None)
            if existing is not None:
                energies = parse_gaussian_scf_energies(
                    existing.read_text(encoding="utf-8", errors="replace")
                )
                if energies:
                    gaussian_energy = energies[-1]
        rows.append(
            {
                "spacing_angstrom": spacing,
                "pyscf_rhf_hartree": float(reference["rhf_energy"]),
                "gaussian_rhf_hartree": gaussian_energy,
                "gaussian_minus_pyscf_microhartree": (
                    None
                    if gaussian_energy is None
                    else 1e6
                    * (
                        gaussian_energy
                        - float(reference["rhf_energy"])
                    )
                ),
            }
        )

    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "h4_rhf_sto3g_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    axis.plot(
        [row["spacing_angstrom"] for row in rows],
        [row["pyscf_rhf_hartree"] for row in rows],
        marker="o",
        label="PySCF RHF/STO-3G",
    )
    gaussian_rows = [
        row for row in rows if row["gaussian_rhf_hartree"] is not None
    ]
    if gaussian_rows:
        axis.scatter(
            [row["spacing_angstrom"] for row in gaussian_rows],
            [row["gaussian_rhf_hartree"] for row in gaussian_rows],
            marker="x",
            s=55,
            label="Gaussian RHF/STO-3G",
        )
    axis.set_xlabel("Nearest-neighbour H-H spacing R (Å)")
    axis.set_ylabel("Total RHF energy (Ha)")
    axis.set_title("Linear H₄ binding curve")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output / "h4_rhf_sto3g_curve.png", dpi=180)
    plt.close(figure)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
