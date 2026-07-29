#!/usr/bin/env python3
"""Run the packaged diazene workflow from the project root.

``quick`` reuses the bundled molecular integrals and circuits, regenerates
the deterministic data analysis, and runs the test suite.  ``full`` also
rebuilds every PySCF reference, every circuit, and every simulated JSON file.
Gaussian is intentionally not invoked because it requires the user's local
licensed executable; its input generation is included in both modes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], project_root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help=(
            "quick: use bundled references/circuits/data; "
            "full: rebuild them with PySCF before analysis"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help=(
            "Bootstrap repetitions for the transition-region gap. "
            "Use 2000 to reproduce the bundled uncertainty result."
        ),
    )
    arguments = parser.parse_args()
    if arguments.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative.")

    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    if arguments.mode == "full":
        run([python, "code/build_references.py"], project_root)
        run([python, "code/generate_circuits.py"], project_root)
        run([python, "code/simulate_data.py"], project_root)
        run(
            [
                python,
                "code/simulate_data.py",
                "--shots",
                "1000",
                "--seed",
                "5210",
            ],
            project_root,
        )

    run([python, "code/generate_gaussian_inputs.py"], project_root)
    run([python, "code/summarize_parameters.py"], project_root)
    run(
        [
            python,
            "code/analyze_scan.py",
            "--input-dir",
            "data/ideal",
            "--output-dir",
            "results/ideal_scan",
        ],
        project_root,
    )
    run(
        [
            python,
            "code/analyze_scan.py",
            "--input-dir",
            "data/sample_1000_shots",
            "--output-dir",
            "results/sample_1000_shots_scan",
        ],
        project_root,
    )
    if arguments.bootstrap:
        run(
            [
                python,
                "code/bootstrap_mechanism_gap.py",
                "--input-dir",
                "data/ideal",
                "--shots",
                "1000",
                "--repetitions",
                str(arguments.bootstrap),
                "--seed",
                "5210",
                "--output-dir",
                "results/ideal_1000_shot_gap_bootstrap",
            ],
            project_root,
        )
    run([python, "tests/run_tests.py"], project_root)
    run([python, "code/generate_checksums.py"], project_root)


if __name__ == "__main__":
    main()
