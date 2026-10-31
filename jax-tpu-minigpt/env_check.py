#!/usr/bin/env python3
"""Verify the pinned JAX/Flax stack on a Colab TPU runtime.

Colab's preinstalled pair (jax 0.11.2 + flax 0.11.2) cannot import flax.nnx at
all, so the notebook pins a newer stack. Run this after the install cell and a
kernel restart; a version mismatch here means the restart was skipped.
"""

from __future__ import annotations

import importlib
import sys

EXPECTED = {
    "jax": "0.12.0.1",
    "jaxlib": "0.12.0.1",
    "flax": "0.12.12",
    "optax": "0.2.8",
    "grain": "0.2.18",
    "tiktoken": "0.14.0",
}


def main() -> int:
    print("python:", sys.version.split()[0])
    bad = []
    for name, want in EXPECTED.items():
        try:
            got = getattr(importlib.import_module(name), "__version__", "(none)")
        except Exception as e:  # noqa: BLE001
            got = f"import failed: {type(e).__name__}"
        ok = got == want
        print(f"  {name:9s} {got:<24s} {'ok' if ok else f'expected {want}'}")
        if not ok:
            bad.append(name)

    import jax

    devices = jax.devices()
    print("devices:", devices)
    if not any(d.platform == "tpu" for d in devices):
        print("\nNo TPU device. Pick a TPU runtime: Runtime > Change runtime type.")
        return 1

    from flax import nnx

    missing = [a for a in ("Optimizer", "value_and_grad", "jit", "List") if not hasattr(nnx, a)]
    print("flax.nnx:", "ok" if not missing else f"missing {missing}")

    if bad:
        print(
            "\nVersions differ from the pins. If you just ran the install cell, "
            "restart the runtime (Runtime > Restart session) and run this again: "
            "pip writes to disk, but the already-imported modules stay loaded."
        )
        return 1
    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
