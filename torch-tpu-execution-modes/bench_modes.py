#!/usr/bin/env python3
"""TorchTPU 0.2.0 execution-mode benchmark on TPU v6e-1.

Important: TorchTPU locks environment variables at process initialization
(`import torch` / `torch.device("tpu")`). Therefore, each execution mode MUST
be invoked in its own separate Python process.

Architecture & parameters match `jax-sharding-bench/bench.py` (0.87B Pre-LN
Transformer LM, vocab=32000, dim=2048, layers=16, heads=16, ff_mult=4, tied
embedding head, fp32 master weights with bf16 compute, seq_len=2048,
batch_per_device=1, lr=1e-4, warmup=5, steps=30).

Modes:
  eager              Default strict eager (`EagerMode.DEFER_NEVER`)
  defer_fuse         `TORCH_TPU_DEFER_AND_FUSE=1` (set BEFORE `import torch`)
  defer_fuse_ctx     Context manager `torch.tpu.set_eager_mode(EagerMode.DEFER_AND_FUSE)`
  compile            `torch.compile(model)` for forward + loss, eager backward + SGD
  compile_step       `torch.compile(step_fn)` wrapping fwd + loss + bwd + SGD
  single_trace       `SingleTraceTrainer(..., compile_fn=torch.compile)`
  single_trace_static `SingleTraceTrainer` with `StaticCompiler` + buffer donation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import platform
import statistics
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TorchTPU 0.2.0 execution mode benchmark")
    p.add_argument(
        "--mode",
        choices=[
            "eager",
            "defer_fuse",
            "defer_fuse_ctx",
            "compile",
            "compile_step",
            "single_trace",
            "single_trace_static",
        ],
        required=True,
        help="Execution mode (each mode must run in its own process)",
    )
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--tier2-cache",
        default="",
        help="Value for TORCH_TPU_TIER2_COMPILATION_CACHE (e.g. 'tpu_tier2_cache' or 'disabled')",
    )
    p.add_argument(
        "--precision",
        choices=["", "DEFAULT", "HIGH", "HIGHEST"],
        default="",
        help="Optional torch.tpu.Precision context override",
    )
    p.add_argument(
        "--dynamic-seqs",
        default="",
        help="Comma-separated sequence lengths to test dynamic shapes (e.g. '512,1024,2048')",
    )
    p.add_argument(
        "--compile-dynamic",
        choices=["default", "none", "false"],
        default="default",
        help="dynamic=... argument for torch.compile when testing dynamic shapes",
    )
    p.add_argument("--dump-versions", default="", help="Write versions JSON to path")
    p.add_argument("--output", default="", help="Output JSON path")
    p.add_argument("--tag", default="")
    return p.parse_args()


def configure_pre_import_env(args: argparse.Namespace) -> dict[str, str]:
    """Configure TorchTPU env vars BEFORE importing torch."""
    applied: dict[str, str] = {}
    if args.mode == "defer_fuse":
        os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"
        applied["TORCH_TPU_DEFER_AND_FUSE"] = "1"
    else:
        # Ensure strict isolation if caller environment had it set
        os.environ.pop("TORCH_TPU_DEFER_AND_FUSE", None)

    os.environ.pop("TORCH_TPU_LAUNCH_BLOCKING", None)

    if args.tier2_cache:
        os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = args.tier2_cache
        applied["TORCH_TPU_TIER2_COMPILATION_CACHE"] = args.tier2_cache
        if args.tier2_cache != "disabled":
            cache_dir = pathlib.Path("/dev/shm/torch_tpu_cache") / args.tier2_cache
            cache_dir.mkdir(parents=True, exist_ok=True)

    return applied


def main() -> None:
    args = parse_args()
    env_applied = configure_pre_import_env(args)

    # CRITICAL: import torch only AFTER environment variables are set
    import contextlib
    import importlib.metadata
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("tpu")
    torch.manual_seed(0)
    if hasattr(torch, "tpu") and hasattr(torch.tpu, "manual_seed_all"):
        torch.tpu.manual_seed_all(0)

    def sync_tpu() -> None:
        if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
            torch.accelerator.synchronize()
        elif hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
            torch.tpu.synchronize()

    # Collect device & package metadata
    props = torch.tpu.get_device_properties(device) if hasattr(torch, "tpu") else None
    device_info: dict[str, Any] = {}
    if props is not None:
        for attr in (
            "name",
            "multi_processor_count",
            "total_memory",
            "devices_per_chip",
            "hbm_bytes_per_chip",
            "hbm_bytes_per_device",
            "bf16_flops_per_chip",
            "bf16_flops_per_device",
            "sparse_cores_per_device",
            "sparse_core_simd_width",
        ):
            if hasattr(props, attr):
                device_info[attr] = getattr(props, attr)

    def pkg_ver(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except Exception:
            return "not_installed"

    versions_meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_tpu": pkg_ver("torch-tpu"),
        "libtpu": pkg_ver("libtpu"),
        "numpy": pkg_ver("numpy"),
        "device_count": (
            torch.accelerator.device_count()
            if hasattr(torch, "accelerator")
            else 1
        ),
        "default_eager_mode": str(getattr(torch.tpu, "eager_mode", "unknown")),
        "device_properties": device_info,
    }

    if args.dump_versions:
        vpath = pathlib.Path(args.dump_versions)
        vpath.parent.mkdir(parents=True, exist_ok=True)
        vpath.write_text(json.dumps(versions_meta, indent=2) + "\n")
        print(f"[bench] wrote versions -> {vpath}")

    # ----------------------------------------------------------------------- #
    # Model definition: exact match to jax-sharding-bench/bench.py
    # ----------------------------------------------------------------------- #
    def rmsnorm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.to(torch.float32).pow(2), dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + 1e-6)).to(torch.bfloat16) * g.to(torch.bfloat16)

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, heads: int, ff_mult: int = 4) -> None:
            super().__init__()
            self.dim = dim
            self.heads = heads
            self.head_dim = dim // heads
            scale = 1.0 / math.sqrt(dim)
            self.ln1 = nn.Parameter(torch.ones((dim,), dtype=torch.float32))
            self.qkv = nn.Parameter(torch.randn((dim, 3 * dim), dtype=torch.float32) * scale)
            self.proj = nn.Parameter(torch.randn((dim, dim), dtype=torch.float32) * scale)
            self.ln2 = nn.Parameter(torch.ones((dim,), dtype=torch.float32))
            self.up = nn.Parameter(torch.randn((dim, ff_mult * dim), dtype=torch.float32) * scale)
            self.down = nn.Parameter(torch.randn((ff_mult * dim, dim), dtype=torch.float32) * scale)

        def forward(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
            B, T, D = x.shape
            h = rmsnorm(x, self.ln1)
            qkv = torch.matmul(h, self.qkv.to(torch.bfloat16))
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = torch.where(causal, att, torch.tensor(-1e9, dtype=att.dtype, device=att.device))
            att = F.softmax(att.to(torch.float32), dim=-1).to(torch.bfloat16)
            o = torch.matmul(att, v).transpose(1, 2).reshape(B, T, D)
            x = x + torch.matmul(o, self.proj.to(torch.bfloat16))
            h = rmsnorm(x, self.ln2)
            ff = F.gelu(torch.matmul(h, self.up.to(torch.bfloat16)), approximate="tanh")
            x = x + torch.matmul(ff, self.down.to(torch.bfloat16))
            return x

    class PreLNTransformerLM(nn.Module):
        def __init__(
            self, vocab: int, dim: int, layers: int, heads: int, ff_mult: int = 4
        ) -> None:
            super().__init__()
            scale = 1.0 / math.sqrt(dim)
            self.vocab = vocab
            self.dim = dim
            self.embed = nn.Parameter(torch.randn((vocab, dim), dtype=torch.float32) * scale)
            self.blocks = nn.ModuleList(
                [TransformerBlock(dim, heads, ff_mult) for _ in range(layers)]
            )
            self.ln_f = nn.Parameter(torch.ones((dim,), dtype=torch.float32))

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            x = F.embedding(tokens, self.embed).to(torch.bfloat16)
            B, T, D = x.shape
            idx = torch.arange(T, device=tokens.device)
            causal = idx[:, None] >= idx[None, :]
            for blk in self.blocks:
                x = blk(x, causal)
            x = rmsnorm(x, self.ln_f)
            logits = torch.matmul(x, self.embed.t().to(torch.bfloat16))
            return logits

    def compute_lm_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        logp = F.log_softmax(logits.to(torch.float32), dim=-1)
        gathered = torch.gather(logp, dim=-1, index=targets.unsqueeze(-1))
        return -torch.mean(gathered)

    print(
        f"[bench] mode={args.mode} eager_mode={getattr(torch.tpu, 'eager_mode', None)} "
        f"device={device_info.get('name', 'tpu')} dim={args.dim} layers={args.layers} "
        f"heads={args.heads} seq={args.seq_len} batch={args.batch_size}"
    )

    model = PreLNTransformerLM(
        vocab=args.vocab,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        ff_mult=args.ff_mult,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] params={n_params / 1e9:.3f}B ({n_params:,})")

    batch = torch.randint(
        0,
        args.vocab,
        (args.batch_size, args.seq_len + 1),
        dtype=torch.long,
        device=device,
    )
    sync_tpu()

    # Optional precision context
    def make_exec_ctx():
        stack = contextlib.ExitStack()
        if args.mode == "defer_fuse_ctx":
            stack.enter_context(
                torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE)
            )
        if args.precision:
            level = getattr(torch.tpu.Precision, args.precision)
            stack.enter_context(torch.tpu.precision(level))
        return stack

    compile_kwargs: dict[str, Any] = {}
    if args.compile_dynamic == "none":
        compile_kwargs["dynamic"] = None
    elif args.compile_dynamic == "false":
        compile_kwargs["dynamic"] = False

    # Build step callable according to mode
    setup_t0 = time.perf_counter()
    if args.mode in ("eager", "defer_fuse", "defer_fuse_ctx"):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        def run_step(inp: torch.Tensor) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_lm_loss(model, inp)
            loss.backward()
            optimizer.step()
            return loss

    elif args.mode == "compile":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        compiled_fwd_loss = torch.compile(
            lambda inp: compute_lm_loss(model, inp), **compile_kwargs
        )

        def run_step(inp: torch.Tensor) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = compiled_fwd_loss(inp)
            loss.backward()
            optimizer.step()
            return loss

    elif args.mode == "compile_step":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        @torch.compile(**compile_kwargs)
        def _compiled_full_step(inp: torch.Tensor) -> torch.Tensor:
            loss = compute_lm_loss(model, inp)
            grads = torch.autograd.grad(loss, list(model.parameters()))
            with torch.no_grad():
                for p, g in zip(model.parameters(), grads):
                    p.sub_(g, alpha=args.lr)
            return loss

        def run_step(inp: torch.Tensor) -> torch.Tensor:
            return _compiled_full_step(inp)

    elif args.mode == "single_trace":
        # Note: default Sgd(fused=True) emits aten._fused_sgd_ which fails under
        # torch.compile's aot_autograd (assert_functional_graph). Setting fused=False
        # makes it compatible with compile_fn=torch.compile.
        tpu_opt = torch.tpu.optimizers.Sgd(lr=args.lr, fused=False)

        def _stt_loss_fn(m: nn.Module, call_args: Any, call_kwargs: Any) -> torch.Tensor:
            inp = call_args[0]
            return compute_lm_loss(m, inp)

        trainer = torch.tpu.optimizers.SingleTraceTrainer(
            model=model,
            optimizer=tpu_opt,
            compile_fn=lambda gm: torch.compile(gm, **compile_kwargs),
            compute_loss_fn=_stt_loss_fn,
        )
        compiled_train_step = trainer.make_compiled_train_step(batch)

        def run_step(inp: torch.Tensor) -> torch.Tensor:
            return compiled_train_step(inp)

    elif args.mode == "single_trace_static":
        from torch.utils import _pytree
        from torch_tpu._internal.compile import compiler as tpu_compiler

        # StaticCompiler with buffer donation works with default Sgd(fused=True)
        tpu_opt = torch.tpu.optimizers.Sgd(lr=args.lr, fused=True)

        def _stt_loss_fn(m: nn.Module, call_args: Any, call_kwargs: Any) -> torch.Tensor:
            inp = call_args[0]
            return compute_lm_loss(m, inp)

        static_compiler = tpu_compiler.StaticCompiler()
        trainer = torch.tpu.optimizers.SingleTraceTrainer(
            model=model,
            optimizer=tpu_opt,
            compute_loss_fn=_stt_loss_fn,
        )

        def _static_compile_fn(graph_module, flat_inputs):
            flat_pg, _ = _pytree.tree_flatten(trainer.param_group)
            donated = [
                i for i, x in enumerate(flat_pg) if isinstance(x, torch.Tensor)
            ]
            return static_compiler(
                graph_module,
                flat_inputs,
                donated_inputs=donated,
            )

        compiled_train_step = trainer.make_compiled_train_step(
            batch, compile_fn=_static_compile_fn
        )

        def run_step(inp: torch.Tensor) -> torch.Tensor:
            return compiled_train_step(inp)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    sync_tpu()
    setup_sec = time.perf_counter() - setup_t0
    print(f"[bench] step callable setup took {setup_sec:.3f}s")

    # ----------------------------------------------------------------------- #
    # Optional dynamic sequence length sweep
    # ----------------------------------------------------------------------- #
    dynamic_records: list[dict[str, Any]] = []
    if args.dynamic_seqs:
        seq_list = [int(s.strip()) for s in args.dynamic_seqs.split(",") if s.strip()]
        print(f"[bench] running dynamic sequence sweep: {seq_list}")
        with make_exec_ctx():
            for s_len in seq_list:
                dyn_batch = torch.randint(
                    0,
                    args.vocab,
                    (args.batch_size, s_len + 1),
                    dtype=torch.long,
                    device=device,
                )
                sync_tpu()
                seq_step_ms: list[float] = []
                for rep in range(4):
                    t0 = time.perf_counter()
                    loss = run_step(dyn_batch)
                    sync_tpu()
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    seq_step_ms.append(round(dt_ms, 2))
                    print(
                        f"[bench][dyn] seq={s_len} iter={rep} dt={dt_ms:.2f}ms loss={float(loss.detach().cpu()):.4f}"
                    )
                dynamic_records.append(
                    {
                        "seq_len": s_len,
                        "step_ms_all": seq_step_ms,
                        "first_step_ms": seq_step_ms[0],
                        "steady_step_ms_median": round(statistics.median(seq_step_ms[1:]), 2),
                    }
                )

    # ----------------------------------------------------------------------- #
    # Main steady-state benchmark loop (warmup + measured steps)
    # ----------------------------------------------------------------------- #
    all_step_sec: list[float] = []
    first_loss_val: float | None = None
    final_loss_val: float | None = None

    total_iters = args.warmup + args.steps
    with make_exec_ctx():
        for step in range(total_iters):
            t0 = time.perf_counter()
            loss = run_step(batch)
            sync_tpu()
            dt = time.perf_counter() - t0
            all_step_sec.append(dt)
            if step == 0:
                first_loss_val = float(loss.detach().cpu())
                print(
                    f"[bench] step 0 (first step + compile): {dt * 1000.0:.1f} ms "
                    f"({dt:.2f} s), setup+step0={setup_sec + dt:.2f} s, loss={first_loss_val:.4f}"
                )
            elif step < args.warmup:
                print(f"[bench] warmup step {step}: {dt * 1000.0:.2f} ms")
            elif step == total_iters - 1:
                final_loss_val = float(loss.detach().cpu())

    measured_sec = all_step_sec[args.warmup :]
    measured_ms = [t * 1000.0 for t in measured_sec]
    sorted_ms = sorted(measured_ms)
    median_s = statistics.median(measured_sec)
    median_ms = median_s * 1000.0
    p10_ms = sorted_ms[max(0, len(sorted_ms) // 10)]
    p90_ms = sorted_ms[min(len(sorted_ms) - 1, (9 * len(sorted_ms)) // 10)]
    mean_ms = statistics.mean(measured_ms)
    stdev_ms = statistics.stdev(measured_ms) if len(measured_ms) > 1 else 0.0
    tokens_per_step = args.batch_size * args.seq_len
    tok_per_sec = tokens_per_step / median_s

    result = {
        "mode": args.mode,
        "tag": args.tag,
        "env_applied": env_applied,
        "effective_eager_mode": str(getattr(torch.tpu, "eager_mode", "unknown")),
        "precision_override": args.precision or "default",
        "compile_dynamic": args.compile_dynamic,
        "params_b": round(n_params / 1e9, 3),
        "params_exact": n_params,
        "vocab": args.vocab,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "ff_mult": args.ff_mult,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "setup_sec": round(setup_sec, 4),
        "first_step_sec": round(all_step_sec[0], 4),
        "first_step_plus_setup_sec": round(setup_sec + all_step_sec[0], 4),
        "first_step_ms": round(all_step_sec[0] * 1000.0, 2),
        "step_ms_median": round(median_ms, 2),
        "step_ms_mean": round(mean_ms, 2),
        "step_ms_stdev": round(stdev_ms, 2),
        "step_ms_p10": round(p10_ms, 2),
        "step_ms_p90": round(p90_ms, 2),
        "step_ms_min": round(min(measured_ms), 2),
        "step_ms_max": round(max(measured_ms), 2),
        "tokens_per_sec": round(tok_per_sec, 1),
        "first_loss": round(first_loss_val, 5) if first_loss_val is not None else None,
        "final_loss": round(final_loss_val, 5) if final_loss_val is not None else None,
        "warmup_step_ms": [round(t * 1000.0, 2) for t in all_step_sec[: args.warmup]],
        "measured_step_ms": [round(t * 1000.0, 2) for t in measured_sec],
        "dynamic_sweep": dynamic_records,
        "device_name": device_info.get("name", "unknown"),
        "hbm_gb_static_per_device": (
            round(
                (
                    device_info.get("hbm_bytes_per_device")
                    or device_info.get("total_memory")
                    or 0
                )
                / (1024**3),
                2,
            )
            if (
                "hbm_bytes_per_device" in device_info
                or "total_memory" in device_info
            )
            else None
        ),
        "peak_hbm_measured": None,
        "peak_hbm_note": (
            "torch_tpu 0.2.0 does not expose a runtime max_memory_allocated API; "
            "get_device_properties() in 0.2.0 only exposes static total_memory (32 GiB)."
        ),
    }

    print(json.dumps(result, indent=2))

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[bench] wrote result -> {out_path}")


if __name__ == "__main__":
    main()
