#!/usr/bin/env python3
"""Draw the two-setting H2 measurement logic as a presentation-ready PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(axis, x, y, width, label, color):
    patch = FancyBboxPatch(
        (x - width / 2, y - 0.19),
        width,
        0.38,
        boxstyle="round,pad=0.04,rounding_size=0.05",
        facecolor=color,
        edgecolor="#2F3440",
        linewidth=1.0,
    )
    axis.add_patch(patch)
    axis.text(x, y, label, ha="center", va="center", fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/h2_measurement_circuits.png"),
    )
    arguments = parser.parse_args()

    figure, axis = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    axis.set_xlim(-0.2, 10.2)
    axis.set_ylim(-0.6, 4.2)
    axis.axis("off")

    sections = [
        (3.05, "Setting 0: Z diagonal", False),
        (0.95, "Setting 1: real off-diagonal", True),
    ]
    for center, title, has_analyzer in sections:
        q0_y, q1_y = center + 0.30, center - 0.30
        axis.text(
            -0.05,
            center + 0.72,
            title,
            fontsize=12,
            fontweight="bold",
            ha="left",
        )
        axis.text(0.15, q0_y, "q0", ha="right", va="center")
        axis.text(0.15, q1_y, "q1", ha="right", va="center")
        axis.hlines([q0_y, q1_y], 0.35, 9.85, color="#333333", linewidth=1.2)
        box(axis, 1.05, q0_y, 0.55, "X", "#F28E2B")
        box(
            axis,
            3.05,
            center,
            2.25,
            "Givens  G(θ=π/4)\n2×SQISWAP + RZ",
            "#8FB9DF",
        )
        if has_analyzer:
            box(
                axis,
                6.35,
                center,
                2.25,
                "Analyzer  A₀₁\nRZ(±π/4) + SQISWAP",
                "#B7D7A8",
            )
            measure_x = 9.15
        else:
            measure_x = 6.35
        box(axis, measure_x, q0_y, 0.85, "MZ", "#D9D9D9")
        box(axis, measure_x, q1_y, 0.85, "MZ", "#D9D9D9")
        if has_analyzer:
            axis.text(
                9.80,
                center,
                "γ₀₁ = (⟨n₀⟩−⟨n₁⟩)/2",
                ha="right",
                va="center",
                fontsize=10,
                color="#2E6A3E",
            )
        else:
            axis.text(
                9.80,
                center,
                "γ₀₀=⟨n₀⟩,  γ₁₁=⟨n₁⟩",
                ha="right",
                va="center",
                fontsize=10,
                color="#355C8A",
            )

    figure.suptitle(
        "H₂ RHF/STO-3G: two settings reconstruct the real 2×2 one-spin 1-RDM",
        fontsize=13,
        fontweight="bold",
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=200, facecolor="white")
    plt.close(figure)
    print(arguments.output)


if __name__ == "__main__":
    main()
