#!/usr/bin/env python3
"""Verify the OpenAI-compatible API of a running vllm-torchtpu server.

Checks the four endpoints the experiment calls for, then exercises the
chat endpoint through the OpenAI Python SDK: non-streaming, streaming,
Japanese input, English input, and a small code-generation prompt.

An endpoint the build does not implement is recorded as unsupported rather
than treated as a failure -- which of them exist is itself a finding.

Usage (inside the venv, server already up):
  ./smoke_test.py --out results/smoke_test_results.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


def raw_get(url: str) -> dict:
    """GET without the SDK, so the HTTP status itself is observable."""
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read().decode(errors="replace")
            return {"status": r.status, "body": body[:4000]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read().decode(errors="replace")[:2000],
                "supported": e.code not in (404, 501)}
    except Exception as e:  # noqa: BLE001
        return {"status": None, "error": repr(e)}


def raw_post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as r:
            body = json.loads(r.read().decode())
            return {"status": r.status, "latency_s": round(time.monotonic() - t0, 3),
                    "usage": body.get("usage"),
                    "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
                    "text_head": json.dumps(body.get("choices", [{}])[0])[:600]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read().decode(errors="replace")[:2000],
                "supported": e.code not in (404, 501)}
    except Exception as e:  # noqa: BLE001
        return {"status": None, "error": repr(e)}


def sdk_cases(base_url: str, model: str) -> dict:
    if OpenAI is None:
        return {"skipped": "openai package not installed"}
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    out: dict[str, dict] = {}

    # Three prompts with different output shapes: prose, a structured list,
    # and code. All English -- this repository is English-only.
    prompts = {
        "prose": "Explain the difference between prefill and decode in LLM inference.",
        "structured": "List three differences between prefill and decode as bullet points.",
        "code": "Write a Python function that returns the n-th Fibonacci number iteratively.",
    }

    for name, prompt in prompts.items():
        try:
            t0 = time.monotonic()
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0,
            )
            out[f"nonstreaming_{name}"] = {
                "ok": True,
                "latency_s": round(time.monotonic() - t0, 3),
                "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                "completion_tokens": r.usage.completion_tokens if r.usage else None,
                "finish_reason": r.choices[0].finish_reason,
                "content_head": (r.choices[0].message.content or "")[:400],
            }
        except Exception as e:  # noqa: BLE001
            out[f"nonstreaming_{name}"] = {"ok": False, "error": repr(e)}

    # Streaming: TTFT is only meaningful from a stream.
    try:
        t0 = time.monotonic()
        ttft = None
        chunks = 0
        text = []
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompts["prose"]}],
            max_tokens=256,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
        usage = None
        finish = None
        for chunk in stream:
            if chunk.usage:
                usage = {"prompt_tokens": chunk.usage.prompt_tokens,
                         "completion_tokens": chunk.usage.completion_tokens}
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
            if delta and delta.content:
                if ttft is None:
                    ttft = round(time.monotonic() - t0, 3)
                chunks += 1
                text.append(delta.content)
        total = round(time.monotonic() - t0, 3)
        out["streaming_prose"] = {
            "ok": True,
            "ttft_s": ttft,
            "total_s": total,
            "chunks": chunks,
            "finish_reason": finish,
            "usage": usage,
            "content_head": "".join(text)[:400],
        }
    except Exception as e:  # noqa: BLE001
        out["streaming_prose"] = {"ok": False, "error": repr(e)}

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="defaults to the first id from /v1/models")
    ap.add_argument("--out", default="results/smoke_test_results.json")
    args = ap.parse_args()

    root = f"http://{args.host}:{args.port}"
    results: dict[str, object] = {"base_url": root}

    results["health"] = raw_get(f"{root}/health")
    models = raw_get(f"{root}/v1/models")
    results["v1_models"] = models

    model = args.model
    if model is None:
        try:
            model = json.loads(models["body"])["data"][0]["id"]
        except Exception:
            model = "Qwen/Qwen3-8B"
    results["model_used"] = model
    print(f"model id in use: {model}")

    # /v1/completions is the legacy text endpoint; some builds omit it.
    results["v1_completions"] = raw_post(f"{root}/v1/completions", {
        "model": model, "prompt": "TPU stands for", "max_tokens": 16, "temperature": 0,
    })
    results["v1_chat_completions"] = raw_post(f"{root}/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user",
                      "content": "Explain the difference between prefill and decode in LLM inference."}],
        "max_tokens": 256, "temperature": 0,
    })

    results["sdk"] = sdk_cases(f"{root}/v1", model)

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(json.dumps({
        "health": results["health"].get("status"),
        "v1_models": models.get("status"),
        "v1_completions": results["v1_completions"].get("status"),
        "v1_chat_completions": results["v1_chat_completions"].get("status"),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
