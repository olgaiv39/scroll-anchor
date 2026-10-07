"""Fetch the fixed benchmark set and verify its pinned hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for entry in json.loads(Path(args.manifest).read_text())["files"]:
        name = entry["cube"] + "_" + entry["kind"] + ".nrrd"
        path = Path(args.output) / entry["cube"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".partial")
            try:
                total = 0
                with urllib.request.urlopen(entry["url"], timeout=30) as response, temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > 80 * 1024 * 1024:
                            raise RuntimeError("download exceeds 80 MiB cap")
                        output.write(chunk)
                if total != entry["bytes"] or hashlib.sha256(temporary.read_bytes()).hexdigest() != entry["sha256"]:
                    raise RuntimeError("download hash mismatch: " + str(path))
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if path.stat().st_size != entry["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("dataset hash mismatch: " + str(path))
        print("verified", path)


if __name__ == "__main__":
    main()
