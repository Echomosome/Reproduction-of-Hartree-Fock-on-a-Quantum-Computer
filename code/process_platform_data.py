#!/usr/bin/env python3
"""Process nine user/platform JSON files into an H8 1-RDM and RHF energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from h8hf.data_processing import analyze_gamma, reconstruct_gamma  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing setting_00_diagonal.json through setting_08_offdiag.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "results" / "platform_data" / "summary.json",
    )
    args = parser.parse_args()

    manifest = json.loads(
        (PACKAGE_ROOT / "circuits" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    setting_files = {}
    for setting in manifest["settings"]:
        candidate = args.input_dir / f"{setting['id']}.json"
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        setting_files[setting["id"]] = candidate
    gamma, diagnostics = reconstruct_gamma(setting_files, manifest)

    arrays = np.load(
        PACKAGE_ROOT / "reference" / "integrals_and_orbitals.npz"
    )
    analysis = analyze_gamma(
        gamma,
        arrays["gamma_reference"],
        arrays["h_core_basis"],
        arrays["eri_core_basis"],
        float(arrays["nuclear_repulsion"]),
        int(manifest["system"]["n_particles_one_spin"]),
    )
    analysis["settings"] = diagnostics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
