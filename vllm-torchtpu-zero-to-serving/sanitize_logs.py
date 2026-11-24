#!/usr/bin/env python3
"""Strip host-specific detail from captured logs before publication.

Removes: the VM's unix username and home path, TPU VM internal hostnames,
private and ephemeral IP addresses, and GCP project ids passed in via
--project-id. Writes to results/sanitized_logs/.

Usage:
  ./sanitize_logs.py results/*.log --project-id my-project
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

OUT = Path("results/sanitized_logs")

# TPU VM hostnames look like t1v-n-XXXXXXXX-w-0; the middle field is unique
# to the node and ties a log to a specific machine.
RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/home/[A-Za-z0-9._-]+"), "/home/user"),
    (re.compile(r"\bt1v-n-[0-9a-f]+-w-\d+\b"), "t1v-n-REDACTED-w-0"),
    (re.compile(r"\btpu\.\d{10,}-\d+-\S+"), "tpu.REDACTED"),
    # Loopback and wildcard addresses are deliberately NOT redacted: they are
    # not host-identifying, and 127.0.0.1 is the evidence that the endpoint was
    # bound to localhost rather than exposed. Only routable addresses go.
    (re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)"
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "x.x.x.x"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "hf_REDACTED"),
    (re.compile(r"ya29\.[A-Za-z0-9._-]+"), "TOKEN_REDACTED"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--project-id", action="append", default=[],
                    help="GCP project id(s) to redact; repeatable")
    args = ap.parse_args()

    rules = list(RULES)
    for pid in args.project_id:
        rules.append((re.compile(re.escape(pid)), "GCP_PROJECT"))

    OUT.mkdir(parents=True, exist_ok=True)
    for raw in args.paths:
        src = Path(raw)
        if not src.is_file():
            continue
        text = src.read_text(errors="replace")
        counts = {}
        for pat, repl in rules:
            text, n = pat.subn(repl, text)
            if n:
                counts[pat.pattern[:34]] = n
        dst = OUT / src.name
        dst.write_text(text)
        detail = ", ".join(f"{k}={v}" for k, v in counts.items()) or "nothing to redact"
        print(f"{src.name} -> {dst}  ({detail})")


if __name__ == "__main__":
    main()
