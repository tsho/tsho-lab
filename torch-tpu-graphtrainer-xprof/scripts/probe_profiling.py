#!/usr/bin/env python3
"""Find the profiling hooks TorchTitan actually exposes, before wiring a capture."""
import dataclasses, importlib, inspect, os

def show(cls, name):
    print(f"\n--- {name} ---")
    if not dataclasses.is_dataclass(cls):
        print("  not a dataclass"); return
    for f in dataclasses.fields(cls):
        d = f.default
        if d is dataclasses.MISSING:
            d = "<factory>" if f.default_factory is not dataclasses.MISSING else "<required>"
        print(f"  {f.name}: {f.type} = {d!r}")

print("=== 1. is there a profiling section in the job config? ===")
for path, syms in (
    ("torchtitan.config.configs", ("ProfilingConfig", "Profiling", "JobConfig")),
    ("torchtitan.experiments.tpu.tpu_job_config", ("TPUTrainerConfig", "TPUConfig")),
):
    try:
        m = importlib.import_module(path)
    except Exception as e:
        print(f"  import failed {path}: {e}"); continue
    print(f"  {path}: {[n for n in dir(m) if 'rofil' in n or 'race' in n]}")
    for s in syms:
        c = getattr(m, s, None)
        if c and dataclasses.is_dataclass(c):
            names = [f.name for f in dataclasses.fields(c)]
            hits = [n for n in names if "profil" in n or "trace" in n]
            if hits:
                print(f"    {s} profiling-ish fields: {hits}")

print("\n=== 2. TPUConfig.use_jax_profiler: who reads it? ===")
import subprocess
r = subprocess.run(["grep","-rn","use_jax_profiler","--include=*.py","."],
                   capture_output=True, text=True, cwd=os.environ.get("TITAN_DIR", os.path.expanduser("~/torchtitan")))
print(r.stdout[:1500] or "  (no hits)")

print("\n=== 3. does the trainer expose a per-step hook we can wrap? ===")
try:
    tgt = importlib.import_module("torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer")
    T = tgt.TpuGraphTrainer
    print("  TpuGraphTrainer methods:", [n for n in dir(T) if not n.startswith("__")][:30])
    for meth in ("train", "train_step", "run"):
        f = getattr(T, meth, None)
        if f:
            try:
                print(f"\n  --- {meth} source ---")
                print(inspect.getsource(f)[:1200])
            except Exception as e:
                print(f"  {meth}: {e}")
except Exception as e:
    print("  failed:", e)

print("\n=== 4. torch.tpu.profiler surface ===")
import torch.tpu.profiler as tp
print("  members:", [n for n in dir(tp) if not n.startswith("_")])
