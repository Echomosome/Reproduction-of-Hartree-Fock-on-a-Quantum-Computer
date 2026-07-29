"""Gaussian input rendering and SCF Done parser."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np


SCF_DONE = re.compile(
    r"SCF Done:\s+E\([^)]+\)\s*=\s*([+-]?\d+\.\d+(?:[DEde][+-]?\d+)?)"
)


def render_gaussian_input(
    coordinates_angstrom: np.ndarray,
    *,
    checkpoint: str = "H8_R1p3000_RHF_STO3G.chk",
    memory: str = "4GB",
    processors: int = 4,
) -> str:
    coordinate_lines = "\n".join(
        f"H  {x: .10f}  {y: .10f}  {z: .10f}"
        for x, y, z in np.asarray(coordinates_angstrom)
    )
    return (
        f"%chk={checkpoint}\n"
        f"%mem={memory}\n"
        f"%nprocshared={processors}\n"
        "#p RHF/STO-3G SCF=(Tight,XQC,MaxCycle=512) NoSymm Pop=Full\n\n"
        "Linear H8, R=1.3000 Angstrom, RHF/STO-3G\n\n"
        "0 1\n"
        f"{coordinate_lines}\n\n"
    )


def parse_scf_energy(text: str) -> float:
    matches = SCF_DONE.findall(text)
    if not matches:
        raise ValueError("No Gaussian 'SCF Done' energy was found.")
    return float(matches[-1].replace("D", "E").replace("d", "e"))


def parse_scf_energy_file(path: str | Path) -> float:
    return parse_scf_energy(
        Path(path).read_text(encoding="utf-8", errors="replace")
    )
