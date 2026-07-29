#!/usr/bin/env python3
"""Run the precomputed-reference H2 reproduction from a clean package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use a random finite-shot sample instead of exact proportions.",
    )
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    reference = "reference/h2_r0.7414_sto3g_reference.npz"
    run(
        [
            python,
            "code/generate_circuits.py",
            "--reference",
            reference,
            "--output-dir",
            "circuits",
        ],
        root,
    )
    run(
        [
            python,
            "code/simulate_counts.py",
            "--circuit-dir",
            "circuits/gate_body_radians",
            "--output-dir",
            "data/example_ideal_counts",
            "--shots",
            str(arguments.shots),
            "--mode",
            "sample" if arguments.sample else "exact",
            "--seed",
            "5210",
        ],
        root,
    )
    run(
        [
            python,
            "code/analyze_results.py",
            "--input-dir",
            "data/example_ideal_counts",
            "--reference",
            reference,
            "--output-dir",
            "results_demo",
            "--bootstrap",
            str(arguments.bootstrap),
            "--seed",
            "5210",
        ],
        root,
    )
    print("\nDemo complete: results_demo/summary.json")


if __name__ == "__main__":
    main()
