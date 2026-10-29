#!/usr/bin/env python3
"""Diff two verify.py result files and print what changed between versions.

    python3 compare.py results/torch-2.14.1.json results/torch-2.15.0.json
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    old_path, new_path = sys.argv[1], sys.argv[2]
    old = json.load(open(old_path))
    new = json.load(open(new_path))
    old_v = old.get("env", {}).get("torch", old_path)
    new_v = new.get("env", {}).get("torch", new_path)

    changed = [k for k in old if k != "env" and old[k] != new.get(k)]
    same = [k for k in old if k != "env" and old[k] == new.get(k)]

    print(f"# {old_v} -> {new_v}\n")
    print(f"## Changed ({len(changed)})\n")
    for k in changed:
        print(f"### {k}")
        print(f"- {old_v}: {json.dumps(old[k], ensure_ascii=False)}")
        print(f"- {new_v}: {json.dumps(new.get(k), ensure_ascii=False)}\n")
    print(f"## Unchanged ({len(same)})\n")
    for k in same:
        print(f"- {k}: {json.dumps(old[k], ensure_ascii=False)[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
