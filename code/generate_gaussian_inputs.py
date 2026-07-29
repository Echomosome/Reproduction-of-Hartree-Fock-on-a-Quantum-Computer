#!/usr/bin/env python3
"""Generate Gaussian RHF/STO-3G inputs for one H2 point or a bond scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tag_for_length(length: float) -> str:
    return f"{length:.4f}".replace(".", "p")


def gaussian_input(length: float, memory: str, nproc: int) -> str:
    half = length / 2.0
    tag = tag_for_length(length)
    return f"""%chk=h2_R{tag}_RHF_STO3G.chk
%mem={memory}
%nprocshared={nproc}
#p RHF/STO-3G SP SCF=(Tight,Conver=10) Pop=Full NoSymm

H2 RHF/STO-3G R={length:.4f} Angstrom

0 1
H  0.0000000000  0.0000000000  {-half:.10f}
H  0.0000000000  0.0000000000   {half:.10f}

"""


def parse_lengths(specification: str) -> list[float]:
    if ":" in specification:
        start_text, stop_text, step_text = specification.split(":")
        start, stop, step = map(
            float, (start_text, stop_text, step_text)
        )
        if step <= 0 or stop < start:
            raise ValueError("Invalid start:stop:step specification.")
        values: list[float] = []
        current = start
        while current <= stop + step * 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(value) for value in specification.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        default="0.7414",
        help="Comma list or start:stop:step in Angstrom.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("gaussian/inputs")
    )
    parser.add_argument("--memory", default="2GB")
    parser.add_argument("--nproc", type=int, default=4)
    arguments = parser.parse_args()

    lengths = parse_lengths(arguments.lengths)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for length in lengths:
        tag = tag_for_length(length)
        path = arguments.output_dir / f"h2_R{tag}_RHF_STO3G.gjf"
        path.write_text(
            gaussian_input(length, arguments.memory, arguments.nproc),
            encoding="utf-8",
        )
        files.append(str(path))
    print(json.dumps({"bond_lengths_angstrom": lengths, "files": files}, indent=2))


if __name__ == "__main__":
    main()
