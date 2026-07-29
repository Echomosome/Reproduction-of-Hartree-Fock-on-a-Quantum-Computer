#!/usr/bin/env python3
"""Create exact or finite-shot example JSON results from native H2 circuits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from h2_core import (
    one_particle_probabilities_from_circuit,
    parse_native_circuit,
    write_json,
)


FILES = {
    "z_diagonal": "H2-00-z-diagonal.txt",
    "real_offdiagonal": "H2-01-real-offdiagonal.txt",
}


def largest_remainder_counts(
    probabilities: dict[str, float], shots: int
) -> dict[str, int]:
    keys = sorted(probabilities)
    expected = np.asarray([probabilities[key] * shots for key in keys])
    counts = np.floor(expected).astype(int)
    remaining = shots - int(counts.sum())
    if remaining:
        order = np.argsort(-(expected - counts))
        counts[order[:remaining]] += 1
    return {key: int(value) for key, value in zip(keys, counts)}


def sample_counts(
    probabilities: dict[str, float],
    shots: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    keys = sorted(probabilities)
    values = np.asarray([probabilities[key] for key in keys], dtype=float)
    counts = rng.multinomial(shots, values)
    return {key: int(value) for key, value in zip(keys, counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuit-dir",
        type=Path,
        default=Path("circuits/gate_body_radians"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/example_ideal_counts"),
    )
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument(
        "--mode", choices=("exact", "sample"), default="exact"
    )
    parser.add_argument("--seed", type=int, default=5210)
    arguments = parser.parse_args()
    if arguments.shots <= 0:
        raise ValueError("--shots must be positive.")

    rng = np.random.default_rng(arguments.seed)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for setting, filename in FILES.items():
        path = arguments.circuit_dir / filename
        gates = parse_native_circuit(path.read_text(encoding="utf-8"))
        probabilities = one_particle_probabilities_from_circuit(gates)
        if arguments.mode == "sample":
            counts = sample_counts(probabilities, arguments.shots, rng)
        else:
            counts = largest_remainder_counts(
                probabilities, arguments.shots
            )
        output = arguments.output_dir / f"{setting}.json"
        payload = {
            "status": "Completed",
            "setting": setting,
            "shots": arguments.shots,
            "source": (
                "synthetic exact native-circuit distribution"
                if arguments.mode == "exact"
                else "synthetic multinomial sample"
            ),
            "bitstring_order": "q[1]q[0]",
            "result": {
                "key": sorted(counts),
                "value": [counts[key] for key in sorted(counts)],
                "value_type": "counts",
            },
        }
        write_json(output, payload)
        summary[setting] = {
            "file": str(output),
            "exact_probabilities": probabilities,
            "counts": counts,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
