#!/usr/bin/env python3
"""Draw an exact overview of the generated Givens-rotation circuits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("circuits/manifest.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/h4_measurement_circuits.png"),
    )
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))

    figure, axes = plt.subplots(
        len(manifest["circuits"]),
        1,
        figsize=(11.0, 8.0),
        constrained_layout=True,
    )
    for axis, circuit in zip(axes, manifest["circuits"]):
        rotations = circuit["givens_rotations"]
        for mode in range(4):
            axis.hlines(mode, 0, len(rotations) + 1, color="#566573")
            axis.text(-0.35, mode, f"q{mode}", va="center", ha="right")
        axis.text(
            0.15,
            0,
            "X",
            ha="center",
            va="center",
            color="white",
            bbox={"boxstyle": "round", "facecolor": "#4C78A8"},
        )
        axis.text(
            0.15,
            1,
            "X",
            ha="center",
            va="center",
            color="white",
            bbox={"boxstyle": "round", "facecolor": "#4C78A8"},
        )
        for index, rotation in enumerate(rotations, start=1):
            first = rotation["first"]
            second = rotation["second"]
            theta = rotation["theta_radians"]
            axis.vlines(
                index,
                first,
                second,
                color="#F28E2B",
                linewidth=3,
            )
            axis.scatter(
                [index, index],
                [first, second],
                color="#F28E2B",
                s=28,
                zorder=3,
            )
            axis.text(
                index,
                (first + second) / 2,
                f"{theta:+.3f}",
                fontsize=7,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                },
            )
        axis.set_title(
            f"{circuit['setting']}  "
            f"({circuit['gate_counts']['SQISWAP']} SQISWAP after compilation)",
            fontsize=10,
            loc="left",
        )
        axis.set_xlim(-0.6, len(rotations) + 0.6)
        axis.set_ylim(3.5, -0.5)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
    figure.suptitle(
        "H₄ RHF: adjacent-mode Givens rotations (labels are radians)",
        fontsize=13,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    plt.close(figure)
    print(f"Saved {arguments.output}")


if __name__ == "__main__":
    main()
