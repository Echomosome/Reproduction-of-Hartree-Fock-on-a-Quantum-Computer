#!/usr/bin/env python3
"""Generate exact or finite-shot diazene platform-style JSON data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from diazene_core import (
    DEFAULT_BIT_ORDER,
    parse_native_circuit,
    slater_probabilities_from_circuit,
    write_json,
)


def setting_seed(base_seed: int, geometry_id: str, setting: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{geometry_id}|{setting}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def sample_counts(
    probabilities: dict[str, float],
    shots: int,
    seed: int,
) -> dict[str, int]:
    keys = sorted(probabilities)
    vector = np.asarray([probabilities[key] for key in keys], dtype=float)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, vector)
    return {
        key: int(count)
        for key, count in zip(keys, counts)
        if count > 0
    }


def simulate_geometry(
    circuit_root: Path,
    geometry_id: str,
    output_root: Path,
    shots: int | None,
    base_seed: int,
) -> list[dict]:
    geometry_dir = circuit_root / geometry_id
    manifest = json.loads(
        (geometry_dir / "manifest.json").read_text(encoding="utf-8")
    )
    records: list[dict] = []
    for circuit in manifest["circuits"]:
        setting = circuit["setting"]
        gate_path = geometry_dir / circuit["gate_body_file"]
        gates = parse_native_circuit(
            gate_path.read_text(encoding="utf-8")
        )
        probabilities = slater_probabilities_from_circuit(gates)
        if shots is None:
            payload = {
                "status": "Completed",
                "geometry_id": geometry_id,
                "setting": setting,
                "value_type": "probabilities",
                "bit_order": DEFAULT_BIT_ORDER,
                "probabilities": probabilities,
                "source": "exact Slater-determinant circuit simulation",
            }
            output_subdir = output_root / "ideal" / geometry_id
            used_seed = None
        else:
            used_seed = setting_seed(
                base_seed, geometry_id, setting
            )
            payload = {
                "status": "Completed",
                "geometry_id": geometry_id,
                "setting": setting,
                "value_type": "counts",
                "shots": shots,
                "seed": used_seed,
                "bit_order": DEFAULT_BIT_ORDER,
                "counts": sample_counts(
                    probabilities, shots, used_seed
                ),
                "source": (
                    "multinomial sampling of exact Slater probabilities"
                ),
            }
            output_subdir = (
                output_root / f"sample_{shots}_shots" / geometry_id
            )
        output_path = output_subdir / f"{setting}.json"
        write_json(output_path, payload)
        records.append(
            {
                "geometry_id": geometry_id,
                "setting": setting,
                "shots": shots,
                "seed": used_seed,
                "output": str(output_path),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuit-dir", type=Path, default=Path("circuits")
    )
    parser.add_argument(
        "--geometry-id",
        default=None,
        help="Simulate one geometry; default simulates all manifests.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Omit for exact probabilities; otherwise sample counts.",
    )
    parser.add_argument("--seed", type=int, default=5210)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data")
    )
    arguments = parser.parse_args()
    if arguments.shots is not None and arguments.shots <= 0:
        raise ValueError("--shots must be positive.")

    if arguments.geometry_id is None:
        manifest_paths = sorted(
            arguments.circuit_dir.glob("*/manifest.json")
        )
        geometry_ids = [path.parent.name for path in manifest_paths]
    else:
        geometry_ids = [arguments.geometry_id]
    if not geometry_ids:
        raise FileNotFoundError("No geometry circuit manifests found.")

    records: list[dict] = []
    for geometry_id in geometry_ids:
        records.extend(
            simulate_geometry(
                arguments.circuit_dir,
                geometry_id,
                arguments.output_dir,
                arguments.shots,
                arguments.seed,
            )
        )
        mode = (
            "exact"
            if arguments.shots is None
            else f"{arguments.shots} shots"
        )
        print(f"{geometry_id}: wrote 10 {mode} JSON files")
    write_json(
        arguments.output_dir
        / (
            "ideal_generation_manifest.json"
            if arguments.shots is None
            else f"sample_{arguments.shots}_shots_manifest.json"
        ),
        records,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
