#!/usr/bin/env python3
"""Verify every file recorded in SHA256SUMS.csv."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    with (ROOT / "SHA256SUMS.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row["relative_path"]
            if not path.is_file():
                failures.append(f"missing: {row['relative_path']}")
            elif path.stat().st_size != int(row["bytes"]):
                failures.append(f"size: {row['relative_path']}")
            elif sha256(path) != row["sha256"]:
                failures.append(f"sha256: {row['relative_path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("All recorded files passed SHA-256 verification.")


if __name__ == "__main__":
    main()
