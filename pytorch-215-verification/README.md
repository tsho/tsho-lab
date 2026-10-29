# pytorch-215-verification

Runs PyTorch 2.15.0's release-note claims against the actual wheels, on both
2.14.1 and 2.15.0, and diffs the results. Nothing here is taken on trust: each
check records what the build *did*, so the comparison shows the change rather
than asserting it.

Host: Apple Silicon (arm64) macOS, `python3.14`, CPU and MPS. That makes the
AArch64 numerical items directly observable. CUDA, ROCm, XPU and
distributed-backend claims are **out of scope** and are not checked here.

```bash
./run.sh                    # 2.14.1 vs 2.15.0
OLD=2.13.0 ./run.sh         # different baseline
python3 compare.py results/torch-2.14.1.json results/torch-2.15.0.json
```

`torch==2.15.0` dropped the cp310 wheel, so `PY` must be Python 3.11 or newer.

## Confirmed changes

| Check | 2.14.1 | 2.15.0 |
|---|---|---|
| `importlib.reload(torch)` | `RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace triton` | `ImportError: torch can only be initialized once per process` |
| `sort` / `argsort` / `topk` / `argmax` / `argmin` on complex | `RuntimeError` | **`TypeError`** |
| `torch.optim.SGD({param}, ...)` (bare set) | no warning | `FutureWarning` |
| `torch.load(weights_only=True)` with a nested tensor | loads | `UnpicklingError` |
| `FunctionEvent.metadata_json` | present | **removed** (`metadata` works in both) |
| `torch.compiler.precompile` | `_PrecompileApi` object, no `capture` | module with `capture` and `load` |
| `torch.foreach`, `torch.optim.functional` | missing | module |
| `torch.mps.Stream` / `current_stream` / `default_stream` / `set_stream` | missing | present |

## Numerics on arm64

Float32 `sin`/`cos` moved to faster kernels with a looser bound. Measured as
max error in **float32 ULPs** against a float64 reference over
`linspace(-20, 20, 20001)`, restricted to `|result| >= 0.5` — near a zero of
`sin` the result is tiny while the argument-reduction error is not, so the ULP
ratio there says nothing about kernel quality.

| | 2.14.1 | 2.15.0 | Release note |
|---|---:|---:|---|
| `sin` | 0.71 ULP | **1.57 ULP** | 1.0 ULP -> 3.5 ULP |
| `cos` | 0.67 ULP | **1.77 ULP** | 1.0 ULP -> 3.5 ULP |

Both stay inside the documented 3.5 ULP bound; this sample never reached it.
The output bits differ, so stored golden outputs will not match:

```
sin  sha256[:16]  7e4de91a766b5749 -> 0fee8f6820a836d9
cos  sha256[:16]  f9daed17ff4a8e42 -> 32d9e3252c814cb0
```

Float32 reduction order changed on NEON as well, over `randn(65536)` with
`manual_seed(0)`:

| | 2.14.1 | 2.15.0 | Difference |
|---|---:|---:|---|
| `sum()` | -243.913223 | -243.913177 | 3 ULP (1.9e-07 relative) |
| `norm()` | 255.679230 | 255.679153 | 5 ULP (3.0e-07 relative) |
| `view(4096,16).sum(0).sum()` | -243.913208 | -243.913193 | 1 ULP |

## Claims this host could not confirm

Reported as-is rather than dropped, since "not observed" is a result:

- **`split` / `split_with_sizes` / `unbind` rebuilt as `slice`/`select`.** The
  note concerns *regenerated* outputs. In eager, and under
  `FunctionalTensorMode`, `grad_fn` stayed `SplitBackward0`,
  `SplitWithSizesBackward0` and `UnbindBackward0` in both versions.
- **`pipeline_per_direction_p2p` renamed to `pipeline_per_edge_p2p`.** Neither
  name is exported anywhere under `torch.distributed.pipelining` in either
  version.
- **`linspace`/`logspace` precision for float16/bfloat16.** Identical outputs
  on CPU in both versions; the improvement may be CUDA-only.

## Files

| File | Purpose |
|---|---|
| `verify.py` | the checks; prints one JSON document |
| `compare.py` | diffs two result files |
| `run.sh` | builds a venv per version, runs both, diffs |
| `results/` | raw output per version |
