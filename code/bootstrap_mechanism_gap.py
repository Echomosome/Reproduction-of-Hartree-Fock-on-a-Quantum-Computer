#!/usr/bin/env python3
"""Bootstrap the discrete nine-point transition-region energy gap."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_geometry import bootstrap
from diazene_core import (
    load_distribution,
    load_reference,
    postselect_particle_number,
    write_json,
)


TRANSITION_REGION_IDS = {
    "out_of_plane": "oop_095p641",
    "in_plane": "ip_182p000",
}


def load_bootstrap_inputs(
    geometry_id: str,
    input_root: Path,
    circuit_root: Path,
    reference_root: Path,
    explicit_shots: int | None,
) -> tuple[
    dict,
    dict,
    np.ndarray,
    dict[str, dict[str, float]],
    dict[str, int],
]:
    manifest = json.loads(
        (circuit_root / geometry_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    reference = load_reference(reference_root / f"{geometry_id}.npz")
    target = np.asarray(reference["gamma_active_one_spin"], dtype=float)
    input_dir = (
        input_root / geometry_id
        if (input_root / geometry_id).is_dir()
        else input_root
    )
    distributions: dict[str, dict[str, float]] = {}
    shots: dict[str, int] = {}
    for circuit in manifest["circuits"]:
        setting = circuit["setting"]
        loaded = load_distribution(input_dir / f"{setting}.json")
        # Check that every source distribution has at least one valid sector.
        postselect_particle_number(loaded.probabilities)
        distributions[setting] = loaded.probabilities
        resolved = explicit_shots or loaded.shots
        if resolved is None:
            raise ValueError(
                f"{geometry_id}/{setting}: shots are unknown; pass --shots."
            )
        shots[setting] = resolved
    return manifest, reference, target, distributions, shots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--circuit-dir", type=Path, default=Path("circuits")
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("reference")
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5210)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mechanism_gap_bootstrap"),
    )
    arguments = parser.parse_args()
    if arguments.repetitions <= 0:
        raise ValueError("--repetitions must be positive.")

    energy_samples: dict[str, np.ndarray] = {}
    reference_energies: dict[str, float] = {}
    for pathway_index, (pathway, geometry_id) in enumerate(
        TRANSITION_REGION_IDS.items()
    ):
        (
            manifest,
            reference,
            target,
            distributions,
            shots,
        ) = load_bootstrap_inputs(
            geometry_id,
            arguments.input_dir,
            arguments.circuit_dir,
            arguments.reference_dir,
            arguments.shots,
        )
        records, _ = bootstrap(
            distributions,
            shots,
            manifest,
            target,
            reference,
            arguments.repetitions,
            arguments.seed + pathway_index,
        )
        energy_samples[pathway] = np.asarray(
            [record["projected_energy_hartree"] for record in records],
            dtype=float,
        )
        reference_energies[pathway] = float(
            reference["full_rhf_energy_hartree"]
        )

    sample_count = min(len(values) for values in energy_samples.values())
    gap_samples = 1000.0 * (
        energy_samples["in_plane"][:sample_count]
        - energy_samples["out_of_plane"][:sample_count]
    )
    q025, median, q975 = np.quantile(
        gap_samples, [0.025, 0.5, 0.975]
    )
    reference_gap = 1000.0 * (
        reference_energies["in_plane"]
        - reference_energies["out_of_plane"]
    )
    rows = [
        {
            "bootstrap_index": index,
            "in_plane_minus_out_of_plane_gap_millihartree": value,
        }
        for index, value in enumerate(gap_samples)
    ]
    summary = {
        "scope": (
            "Discrete transition-region points printed in Appendix J: "
            "ip_182p000 minus oop_095p641. This is not the continuous "
            "20-point-path 40.2 mHa value quoted in the paper."
        ),
        "repetitions_requested": arguments.repetitions,
        "repetitions_used": sample_count,
        "full_pyscf_discrete_gap_millihartree": reference_gap,
        "bootstrap_gap_millihartree": {
            "q025": float(q025),
            "median": float(median),
            "q975": float(q975),
        },
        "probability_gap_has_same_sign_as_reference": float(
            np.mean(np.sign(gap_samples) == np.sign(reference_gap))
        ),
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "gap_bootstrap.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(arguments.output_dir / "gap_summary.json", summary)

    figure, axis = plt.subplots(
        figsize=(7.3, 4.3), constrained_layout=True
    )
    axis.hist(gap_samples, bins=45, color="#4C78A8", alpha=0.8)
    axis.axvline(reference_gap, color="black", label="Full RHF reference")
    axis.axvline(median, color="#F28E2B", label="Bootstrap median")
    axis.axvspan(q025, q975, color="#F28E2B", alpha=0.15, label="95% interval")
    axis.set_xlabel("In-plane minus out-of-plane gap (mHa)")
    axis.set_ylabel("Bootstrap count")
    axis.legend(frameon=False)
    figure.savefig(
        arguments.output_dir / "gap_bootstrap.png", dpi=180
    )
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
