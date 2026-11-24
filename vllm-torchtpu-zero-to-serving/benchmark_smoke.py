#!/usr/bin/env python3
"""Fallback single-request smoke benchmark (used if `vllm bench serve` is absent).

Reports TTFT, TPOT, end-to-end latency and output tok/s per shape, measured
from a streamed response. Warmup is separated from the measured run, and the
ACTUAL token counts are recorded -- a response can stop short of max_tokens.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

SHAPES = [
    ("short", 128, 128),
    ("prefill_heavy", 4096, 128),
    ("decode_heavy", 128, 1024),
]


def make_prompt(approx_tokens: int) -> str:
    # "word " is roughly one token; the exact count is read back from usage.
    return "The quick brown fox jumps over the lazy dog. " * max(1, approx_tokens // 10)


def one(client: OpenAI, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.monotonic()
    ttft = None
    n_chunks = 0
    usage = None
    finish = None
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage:
            usage = {"prompt_tokens": chunk.usage.prompt_tokens,
                     "completion_tokens": chunk.usage.completion_tokens}
        if not chunk.choices:
            continue
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
        d = chunk.choices[0].delta
        if d and d.content:
            if ttft is None:
                ttft = time.monotonic() - t0
            n_chunks += 1
    total = time.monotonic() - t0

    completion = (usage or {}).get("completion_tokens") or n_chunks
    tpot = ((total - ttft) / (completion - 1)) if (ttft and completion and completion > 1) else None
    return {
        "ttft_s": round(ttft, 4) if ttft else None,
        "tpot_ms": round(tpot * 1000, 3) if tpot else None,
        "e2e_latency_s": round(total, 4),
        "output_tok_s": round(completion / total, 2) if completion else None,
        "actual_prompt_tokens": (usage or {}).get("prompt_tokens"),
        "actual_completion_tokens": completion,
        "requested_max_tokens": max_tokens,
        "stopped_early": bool(completion and completion < max_tokens),
        "finish_reason": finish,
        "stream_chunks": n_chunks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="results/benchmark_smoke.json")
    args = ap.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")
    results: dict[str, object] = {"model": args.model, "note":
        "single request per shape; functional check only, not a performance study"}

    print("=== warmup (reported separately, excluded from measurements) ===")
    results["warmup"] = one(client, args.model, make_prompt(128), 128)
    print(json.dumps(results["warmup"], indent=2))

    measured = {}
    for name, inp, outp in SHAPES:
        print(f"=== {name}: {inp} in / {outp} out ===")
        measured[name] = one(client, args.model, make_prompt(inp), outp)
        print(json.dumps(measured[name], indent=2))
    results["measured"] = measured

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
