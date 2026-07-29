#!/usr/bin/env python3
"""Extract Gaussian SCF Done energies and optionally compare to PySCF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h4_core import load_reference, parse_gaussian_scf_energies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reference/h4_r1.3000_sto3g_reference.npz"),
    )
    arguments = parser.parse_args()
    energies = parse_gaussian_scf_energies(
        arguments.log.read_text(encoding="utf-8", errors="replace")
    )
    if not energies:
        raise SystemExit("No 'SCF Done' energy was found.")
    reference = load_reference(arguments.reference)
    final = energies[-1]
    pyscf = float(reference["rhf_energy"])
    print(
        json.dumps(
            {
                "log": str(arguments.log),
                "all_scf_energies_hartree": energies,
                "final_gaussian_rhf_hartree": final,
                "pyscf_rhf_hartree": pyscf,
                "difference_microhartree": 1e6 * (final - pyscf),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
