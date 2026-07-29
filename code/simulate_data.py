#!/usr/bin/env python3
"""Create exact-probability or finite-shot JSON data from H4 gate files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from h4_core import (
    parse_native_circuit,
    slater_probabilities_from_circuit,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuit-dir", type=Path, default=Path("circuits")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/example_ideal_probabilities"),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="0 writes exact probabilities; positive values sample counts.",
    )
    parser.add_argument("--seed", type=int, default=5210)
    arguments = parser.parse_args()
    if arguments.shots < 0:
        raise ValueError("--shots cannot be negative.")

    manifest = json.loads(
        (arguments.circuit_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rng = np.random.default_rng(arguments.seed)
    summaries: list[dict] = []
    for circuit in manifest["circuits"]:
        gate_path = arguments.circuit_dir / circuit["gate_body_file"]
        gates = parse_native_circuit(gate_path.read_text(encoding="utf-8"))
        exact = slater_probabilities_from_circuit(gates)
        keys = sorted(exact)
        vector = np.asarray([exact[key] for key in keys], dtype=float)
        if arguments.shots > 0:
            values = rng.multinomial(arguments.shots, vector).tolist()
            value_type = "counts"
            shots: int | None = arguments.shots
            source = (
                "synthetic multinomial sample from the exact native circuit"
            )
        else:
            values = [float(value) for value in vector]
            value_type = "probabilities"
            shots = None
            source = "synthetic exact native-circuit probabilities"

        payload = {
            "status": "Completed",
            "setting": circuit["setting"],
            "shots": shots,
            "source": source,
            "bitstring_order": "q[3]q[2]q[1]q[0]",
            "result": {
                "key": keys,
                "value": values,
                "value_type": value_type,
            },
        }
        output_path = arguments.output_dir / f"{circuit['setting']}.json"
        write_json(output_path, payload)
        summaries.append(
            {
                "setting": circuit["setting"],
                "file": str(output_path),
                "value_type": value_type,
                "shots": shots,
            }
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
