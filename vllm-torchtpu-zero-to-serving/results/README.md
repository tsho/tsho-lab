# Raw results

From the 2026-11-24 run. Three TPU v6e-1 nodes were used: two Spot nodes were
preempted, and `europe-west4-a` (on-demand) completed the API and benchmark
phases.

| File | Contents |
|---|---|
| `versions.json` | resolved versions of Python, vllm-torchtpu, torch, torch_tpu, libtpu, jax |
| `pip_versions.txt` | full `uv pip freeze`, 210 packages |
| `setup_timings.txt` | unix timestamp before each install step |
| `startup_A.json` | fresh start: no model cache, no compile cache (1 run) |
| `startup_B.json` | model cached, compile cache cleared (3 runs) |
| `startup_C.json` | both cached, process restart (3 runs) |
| `smoke_test_results.json` | the four endpoints plus OpenAI SDK cases |
| `bench_short.json`, `bench_prefill_heavy.json`, `bench_decode_heavy.json` | `vllm bench serve` output, one request each |
| `sanitized_logs/` | server and install logs with host detail removed |

## Two fields not to quote

**`hf_model` byte counts in `startup_*.json` are wrong.** They read
32,794.9 MB. The real size is **16 GB**, confirmed with `du`. The walker that
produced them followed symlinks and counted the Hugging Face hub blobs twice.
`measure_startup.py` has been fixed; these files preserve the original output
rather than being retroactively edited.

**`mean_ttft_ms` in `bench_short.json` is not a valid TTFT.** It reads
3362 ms. `vllm bench serve` runs no warmup and this shape executed first, so it
paid first-touch compilation for its bucket. Use `bench_prefill_heavy.json`
(206 ms) and `bench_decode_heavy.json` (23 ms) for TTFT, and all three for
TPOT, which is consistent at 14.13–14.36 ms.
