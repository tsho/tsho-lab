#!/usr/bin/env python3
"""Introspect the installed TorchTitan TPU stack before committing to a runner.

The previous experiment drove everything from CLI flags like
--tpu_config.use_graph_trainer. Those keys are gone on main, and the GraphTrainer
is now selected by building TpuGraphTrainer.Config. Rather than guess at the new
flag spelling and burn TPU time on failed launches, print what is actually there.

Run on the TPU VM inside the venv:
    python discover.py 2>&1 | tee ~/discover.txt
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import sys


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show_fields(cls, name: str) -> None:
    print(f"\n--- {name} ({cls.__module__}.{cls.__qualname__}) ---")
    if not dataclasses.is_dataclass(cls):
        print("  not a dataclass")
        return
    for f in dataclasses.fields(cls):
        default = f.default
        if default is dataclasses.MISSING:
            default = "<factory>" if f.default_factory is not dataclasses.MISSING else "<required>"
        print(f"  {f.name}: {f.type} = {default!r}")


def try_import(path: str):
    try:
        return importlib.import_module(path)
    except Exception as e:  # noqa: BLE001
        print(f"  IMPORT FAILED {path}: {type(e).__name__}: {e}")
        return None


rule("1. versions")
for m in ("torch", "torch_tpu", "torchtitan"):
    mod = try_import(m)
    if mod:
        print(f"  {m}: {getattr(mod, '__version__', '?')} at {getattr(mod, '__file__', '?')}")

rule("2. optimizer_compile and memory_policy: the actual accepted values")
gt_cfg = try_import("torchtitan.experiments.tpu.graph_trainer.config")
if gt_cfg:
    for sym in ("TpuGraphTrainerOptimizerCompileMode", "TpuGraphTrainerMemoryPolicy"):
        print(f"  {sym} = {getattr(gt_cfg, sym, 'MISSING')}")
    for sym in ("TpuGraphTrainerCompileConfig", "GraphSplitterConfig"):
        cls = getattr(gt_cfg, sym, None)
        if cls:
            show_fields(cls, sym)

job = try_import("torchtitan.experiments.tpu.tpu_job_config")
if job:
    print(f"\n  TpuTrainerOptimizerCompileMode = {getattr(job, 'TpuTrainerOptimizerCompileMode', 'MISSING')}")
    for sym in ("TpuTrainerCompileConfig", "TPUConfig"):
        cls = getattr(job, sym, None)
        if cls:
            show_fields(cls, sym)

base = try_import("torchtitan.experiments.graph_trainer.configs")
if base:
    print(f"\n  GraphTrainerMemoryPolicy = {getattr(base, 'GraphTrainerMemoryPolicy', 'MISSING')}")
    cls = getattr(base, "GraphTrainerCompileConfig", None)
    if cls:
        show_fields(cls, "GraphTrainerCompileConfig")

rule("3. how a shipped recipe builds a GraphTrainer config (the pattern to copy)")
for reg_path, fn in (("torchtitan.experiments.tpu.qwen3.config_registry", "qwen3_8b"),
                     ("torchtitan.experiments.tpu.llama3.config_registry", "graph_trainer_llama3_debugmodel")):
    reg = try_import(reg_path)
    if reg and hasattr(reg, fn):
        print(f"\n--- {reg_path}.{fn} ---")
        try:
            print(inspect.getsource(getattr(reg, fn)))
        except Exception as e:  # noqa: BLE001
            print(f"  source unavailable: {e}")

rule("4. llama3_1b, the model this experiment uses")
reg = try_import("torchtitan.experiments.tpu.llama3.config_registry")
if reg:
    print("  available entries:", [n for n in dir(reg) if n.startswith(("llama3", "graph_trainer"))])
    if hasattr(reg, "llama3_1b"):
        try:
            print(inspect.getsource(reg.llama3_1b))
        except Exception as e:  # noqa: BLE001
            print(f"  source unavailable: {e}")

rule("5. TpuGraphTrainer.Config and from_tpu_trainer_config")
tgt = try_import("torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer")
if tgt and hasattr(tgt, "TpuGraphTrainer"):
    C = getattr(tgt.TpuGraphTrainer, "Config", None)
    if C:
        show_fields(C, "TpuGraphTrainer.Config")
    f = getattr(tgt.TpuGraphTrainer, "from_tpu_trainer_config", None)
    if f:
        print(f"\n  from_tpu_trainer_config{inspect.signature(f)}")

rule("6. the train entrypoint: which trainer does it construct?")
for path in ("torchtitan.experiments.tpu.train",):
    mod = try_import(path)
    if mod:
        try:
            src = inspect.getsource(mod)
            print(f"  {path}: {len(src.splitlines())} lines")
            for i, line in enumerate(src.splitlines(), 1):
                if any(k in line for k in ("GraphTrainer", "Trainer(", "from_tpu_trainer_config",
                                           "argparse", "tyro", "parse", "--")):
                    print(f"    {i}: {line.rstrip()[:150]}")
        except Exception as e:  # noqa: BLE001
            print(f"  source unavailable: {e}")

rule("7. XProf: which capture path exists?")
print("  torch.tpu.profiler.TpuProfilerConfig:")
try:
    from torch.tpu.profiler import TpuProfilerConfig  # noqa: F401
    print("    OK — documented path works")
    print(f"    {inspect.signature(TpuProfilerConfig)}")
except Exception as e:  # noqa: BLE001
    print(f"    FAILED: {type(e).__name__}: {e}")
print("  torch_tpu.profiler:")
try:
    import torch_tpu.profiler as tp
    print(f"    OK — members: {[n for n in dir(tp) if not n.startswith('_')][:20]}")
except Exception as e:  # noqa: BLE001
    print(f"    FAILED: {type(e).__name__}: {e}")
print("  torch.profiler.ProfilerActivity members:")
try:
    import torch.profiler as P
    print(f"    {[n for n in dir(P.ProfilerActivity) if not n.startswith('_')]}")
except Exception as e:  # noqa: BLE001
    print(f"    FAILED: {type(e).__name__}: {e}")

rule("8. grad norm: confirm the in_graph path skips _clip_grad_norm")
if tgt and hasattr(tgt, "TpuGraphTrainer"):
    for meth in ("_clip_grad_norm", "_train_step_optimizer_in_graph", "train_step"):
        m = getattr(tgt.TpuGraphTrainer, meth, None)
        if m:
            try:
                src = inspect.getsource(m)
                calls = "_clip_grad_norm" in src
                print(f"  {meth}: {len(src.splitlines())} lines, mentions _clip_grad_norm={calls}")
            except Exception:  # noqa: BLE001
                print(f"  {meth}: source unavailable")
        else:
            print(f"  {meth}: MISSING")

print("\nDONE")
