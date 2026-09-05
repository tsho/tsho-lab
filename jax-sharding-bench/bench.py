#!/usr/bin/env python3
"""JAX sharding benchmark on TPU — the ICI chapter of the interconnect series.

Same questions as distributed-training-bench (GPU): what does sharding cost in
throughput, and what does it buy in memory — measured, not assumed. Pure JAX
(no flax), synthetic data, random-init GPT-like model so systems behavior is
isolated from convergence.

Sharding modes (GSPMD via NamedSharding on a 1-D "data" mesh):
  dp    replicated params + sharded batch            ~ ZeRO-0 / DDP
  fsdp  params sharded on axis 0 + sharded batch     ~ ZeRO-3 / FSDP2(reshard)
        (XLA inserts all-gathers per use; reduce-scatters grads)

Run on a TPU VM (v5e-8 / v6e-8):
  python3 bench.py --mode dp   --steps 15
  python3 bench.py --mode fsdp --steps 15
  python3 bench.py --mode fsdp --dim 2048 --layers 24   # bigger model

Writes results/*.jsonl with the same schema spirit as the GPU bench.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# --------------------------------------------------------------------------- #
# model: minimal pre-LN transformer LM in pure JAX (random init, bf16 compute)
# --------------------------------------------------------------------------- #
def init_params(key, vocab, dim, layers, heads, ff_mult=4):
    ks = jax.random.split(key, 2 + layers)
    scale = 1.0 / math.sqrt(dim)
    params = {
        "embed": jax.random.normal(ks[0], (vocab, dim), jnp.float32) * scale,
        "blocks": [],
        "ln_f": jnp.ones((dim,), jnp.float32),
    }
    for i in range(layers):
        k1, k2, k3, k4 = jax.random.split(ks[2 + i], 4)
        params["blocks"].append({
            "ln1": jnp.ones((dim,), jnp.float32),
            "qkv": jax.random.normal(k1, (dim, 3 * dim), jnp.float32) * scale,
            "proj": jax.random.normal(k2, (dim, dim), jnp.float32) * scale,
            "ln2": jnp.ones((dim,), jnp.float32),
            "up": jax.random.normal(k3, (dim, ff_mult * dim), jnp.float32) * scale,
            "down": jax.random.normal(k4, (ff_mult * dim, dim), jnp.float32) * scale,
        })
    return params


def forward(params, tokens, heads):
    x = params["embed"][tokens].astype(jnp.bfloat16)          # [B, T, D]
    B, T, D = x.shape
    causal = jnp.tril(jnp.ones((T, T), jnp.bool_))
    for blk in params["blocks"]:
        h = _rmsnorm(x, blk["ln1"])
        qkv = h @ blk["qkv"].astype(jnp.bfloat16)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, heads, D // heads).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, heads, D // heads).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, heads, D // heads).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D // heads)
        att = jnp.where(causal, att, -1e9)
        att = jax.nn.softmax(att.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        x = x + o @ blk["proj"].astype(jnp.bfloat16)
        h = _rmsnorm(x, blk["ln2"])
        x = x + jax.nn.gelu(h @ blk["up"].astype(jnp.bfloat16)) @ blk["down"].astype(jnp.bfloat16)
    x = _rmsnorm(x, params["ln_f"])
    return x @ params["embed"].T.astype(jnp.bfloat16)          # tied head


def _rmsnorm(x, g):
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(var + 1e-6)).astype(jnp.bfloat16) * g.astype(jnp.bfloat16)


def loss_fn(params, batch, heads):
    logits = forward(params, batch[:, :-1], heads)
    targets = batch[:, 1:]
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, targets[..., None], axis=-1))


# --------------------------------------------------------------------------- #
# sharding
# --------------------------------------------------------------------------- #
def shard_params(params, mesh, mode):
    """dp: replicate everything. fsdp: shard every weight's axis 0 across 'data'."""
    def spec(path_leaf):
        if mode == "dp":
            return P()
        arr = path_leaf
        if arr.ndim >= 2 or (arr.ndim == 1 and arr.shape[0] % mesh.shape["data"] == 0):
            return P("data", *([None] * (arr.ndim - 1)))
        return P()
    return jax.tree.map(
        lambda a: jax.device_put(a, NamedSharding(mesh, spec(a))), params)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dp", "fsdp"], default="fsdp")
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-per-device", type=int, default=1)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--profile-dir", default="", help="jax.profiler trace dir (optional)")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    devices = jax.devices()
    n_dev = len(devices)
    mesh = Mesh(devices, axis_names=("data",))
    print(f"[bench] backend={jax.default_backend()} devices={n_dev} "
          f"mode={args.mode} dim={args.dim} layers={args.layers} seq={args.seq_len}")

    key = jax.random.PRNGKey(0)
    params = init_params(key, args.vocab, args.dim, args.layers, args.heads)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"[bench] params={n_params/1e9:.2f}B")
    params = shard_params(params, mesh, args.mode)

    B = args.batch_per_device * n_dev
    batch = jax.device_put(
        jax.random.randint(key, (B, args.seq_len + 1), 0, args.vocab),
        NamedSharding(mesh, P("data", None)))

    heads = args.heads
    @partial(jax.jit, donate_argnums=0)
    def train_step(params, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch, heads)
        new_params = jax.tree.map(lambda p_, g: p_ - args.lr * g, params, grads)
        return new_params, loss

    if args.profile_dir:
        jax.profiler.start_trace(args.profile_dir)

    times = []
    for step in range(args.warmup + args.steps):
        t0 = time.perf_counter()
        params, loss = train_step(params, batch)
        jax.block_until_ready(loss)
        dt = time.perf_counter() - t0
        if step >= args.warmup:
            times.append(dt)
        if step == 0:
            print(f"[bench] first step (compile) {dt:.1f}s")

    if args.profile_dir:
        jax.profiler.stop_trace()

    step_s = statistics.median(times)
    tokens_per_step = B * args.seq_len
    mem = {}
    try:
        stats = devices[0].memory_stats()
        mem = {"peak_bytes_per_device": stats.get("peak_bytes_in_use"),
               "limit_bytes": stats.get("bytes_limit")}
    except Exception:
        pass

    result = {
        "engine": f"jax-{args.mode}",
        "backend": jax.default_backend(),
        "devices": n_dev,
        "params_b": round(n_params / 1e9, 3),
        "dim": args.dim, "layers": args.layers, "seq_len": args.seq_len,
        "batch_global": B,
        "step_ms_median": round(step_s * 1000, 1),
        "step_ms_p10": round(sorted(times)[max(0, len(times)//10)] * 1000, 1),
        "step_ms_p90": round(sorted(times)[min(len(times)-1, 9*len(times)//10)] * 1000, 1),
        "tokens_per_sec_global": int(tokens_per_step / step_s),
        "peak_gb_per_device": round((mem.get("peak_bytes_per_device") or 0) / 2**30, 2),
        "tag": args.tag,
    }
    print(json.dumps(result, indent=2))
    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    name = f"jax-{args.mode}-{jax.default_backend()}-{n_dev}dev.jsonl"
    with (out / name).open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[bench] appended -> results/{name}")


if __name__ == "__main__":
    main()
