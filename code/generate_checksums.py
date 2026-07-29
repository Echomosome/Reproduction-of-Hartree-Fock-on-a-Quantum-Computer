#!/usr/bin/env python3
"""Write SHA-256 checksums for release files, excluding transient artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "CHECKSUMS.sha256"
    excluded_parts = {"__pycache__", ".git"}
    excluded_names = {"CHECKSUMS.sha256", "tmpzov7draq"}
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            path.name in excluded_names
            or any(part in excluded_parts for part in relative.parts)
            or path.suffix == ".pyc"
        ):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative.as_posix()}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} checksums to {output.name}.")


if __name__ == "__main__":
    main()
