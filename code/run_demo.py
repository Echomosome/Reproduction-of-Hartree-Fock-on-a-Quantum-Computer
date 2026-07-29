#!/usr/bin/env python3
"""Regenerate circuits, synthetic data, figures, and both demo analyses."""

from __future__ import annotations

import subprocess
import sys


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    run("code/generate_circuits.py")
    run("code/generate_gaussian_inputs.py")
    run("code/make_circuit_figure.py")
    run(
        "code/simulate_data.py",
        "--output-dir",
        "data/example_ideal_probabilities",
        "--shots",
        "0",
    )
    run(
        "code/simulate_data.py",
        "--output-dir",
        "data/example_1000_shots",
        "--shots",
        "1000",
        "--seed",
        "5210",
    )
    run(
        "code/analyze_results.py",
        "--input-dir",
        "data/example_ideal_probabilities",
        "--output-dir",
        "results/ideal",
        "--bootstrap",
        "0",
    )
    run(
        "code/analyze_results.py",
        "--input-dir",
        "data/example_1000_shots",
        "--output-dir",
        "results/sample_1000_shots",
        "--bootstrap",
        "2000",
        "--seed",
        "5210",
    )
    run(
        "code/shot_planning.py",
        "--bootstrap",
        "2000",
        "--seed",
        "5210",
    )


if __name__ == "__main__":
    main()
