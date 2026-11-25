#!/usr/bin/env python3
"""Construct every configuration in bench_cfg before spending TPU time on a run."""
import traceback

import bench_cfg

names = sorted(n for n in dir(bench_cfg) if n.startswith("cfg_"))
ok = fail = 0
for n in names:
    try:
        c = bench_cfg.__dict__[n]()
        comp = getattr(c, "compile", None)
        bits = []
        for k in ("enable", "mode", "optimizer_compile", "memory_policy", "compile_grad_norm"):
            if comp is not None and hasattr(comp, k):
                bits.append("%s=%r" % (k, getattr(comp, k)))
        print("OK   %-28s %s" % (n, type(c).__qualname__))
        print("     " + " ".join(bits))
        ok += 1
    except Exception as e:
        print("FAIL %-28s %s: %s" % (n, type(e).__name__, e))
        traceback.print_exc(limit=3)
        fail += 1
print("\n%d ok, %d failed" % (ok, fail))
