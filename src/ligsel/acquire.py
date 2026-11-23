"""Register and verify the raw input files.

`python -m ligsel.acquire register` records each file's size and SHA-256 in
data/raw_manifest.json (commit that file: it holds no data). `verify` checks the
files on disk against it, so every result is tied to a stated snapshot.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys

from ligsel.paths import MANIFEST_PATH, load_config, raw_path


def sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def register() -> dict:
    config = load_config()
    entries = {}
    for key, spec in config["files"].items():
        p = raw_path(key, config)
        if not p.exists():
            sys.exit(f"missing raw file '{key}': {p}\n  source: {spec['source']}")
        entries[key] = {"path": spec["path"], "bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest = {"registered": dt.date.today().isoformat(), "files": entries}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"registered {len(entries)} raw files in {MANIFEST_PATH.name}")
    return manifest


def verify() -> list[str]:
    """Return a list of problems; empty means every raw file matches the manifest."""
    if not MANIFEST_PATH.exists():
        return ["no data/raw_manifest.json; run `make register` first"]
    manifest = json.loads(MANIFEST_PATH.read_text())
    config = load_config()
    problems = []
    for key, rec in manifest["files"].items():
        p = raw_path(key, config)
        if not p.exists():
            problems.append(f"{key}: missing at {p}")
        elif sha256(p) != rec["sha256"]:
            problems.append(f"{key}: checksum differs from the registered snapshot")
    return problems


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "verify"
    if cmd == "register":
        register()
    elif cmd == "verify":
        problems = verify()
        if problems:
            sys.exit("raw files do not match the manifest:\n  " + "\n  ".join(problems))
        print("all raw files match data/raw_manifest.json")
    else:
        sys.exit(f"unknown command {cmd!r}; use register or verify")


if __name__ == "__main__":
    main(sys.argv)
