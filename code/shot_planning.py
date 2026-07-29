#!/usr/bin/env python3
"""Estimate finite-shot H4 energy precision under the ideal circuit model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_results import bootstrap_model
from h4_core import (
    load_reference,
    parse_native_circuit,
    slater_probabilities_from_circuit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shots", type=int, nargs="*", default=[1000, 2000, 5000, 10000]
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5210)
    parser.add_argument(
        "--circuit-dir", type=Path, default=Path("circuits")
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reference/h4_r1.3000_sto3g_reference.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/shot_planning")
    )
    arguments = parser.parse_args()
    if any(shots <= 0 for shots in arguments.shots):
        raise ValueError("All shot counts must be positive.")

    manifest = json.loads(
        (arguments.circuit_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    reference = load_reference(arguments.reference)
    target = np.asarray(reference["gamma_one_spin_orth"], dtype=float)
    ideal: dict[str, dict[str, float]] = {}
    for circuit in manifest["circuits"]:
        gate_path = arguments.circuit_dir / circuit["gate_body_file"]
        ideal[circuit["setting"]] = slater_probabilities_from_circuit(
            parse_native_circuit(gate_path.read_text(encoding="utf-8"))
        )

    rows: list[dict] = []
    for index, shots in enumerate(arguments.shots):
        _, summary = bootstrap_model(
            ideal,
            {
                circuit["setting"]: shots
                for circuit in manifest["circuits"]
            },
            manifest,
            target,
            reference,
            ideal,
            arguments.bootstrap,
            arguments.seed + index,
        )
        interval = summary["projected_energy_error_millihartree"]
        rows.append(
            {
                "shots_per_setting": shots,
                "total_shots_four_settings": 4 * shots,
                "projected_energy_error_q025_millihartree": interval["q025"],
                "projected_energy_error_median_millihartree": interval[
                    "median"
                ],
                "projected_energy_error_q975_millihartree": interval["q975"],
                "q975_within_1_kcal_per_mol": (
                    interval["q975"] <= 1.593601
                ),
            }
        )

    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "shot_planning.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)
    axis.plot(
        [row["shots_per_setting"] for row in rows],
        [
            row["projected_energy_error_q975_millihartree"]
            for row in rows
        ],
        marker="o",
        label="97.5th percentile | projected error",
    )
    axis.axhline(
        1.593601,
        color="#D62728",
        linestyle="--",
        label="1 kcal/mol",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Shots per setting")
    axis.set_ylabel("Projected energy error (mHa)")
    axis.set_title("H₄ ideal finite-shot planning")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output / "shot_planning.png", dpi=180)
    plt.close(figure)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
