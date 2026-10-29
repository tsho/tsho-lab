# pytorch-215-verification

Runs PyTorch 2.15.0's release-note claims against the actual wheels, on both
2.14.1 and 2.15.0, and diffs the results. Nothing here is taken on trust: each
check records what the build *did*, so the comparison shows the change rather
than asserting it.

Two hosts:

- **arm64 macOS**, `python3.14`, CPU and MPS — makes the AArch64 numerical items
  directly observable (`verify.py`).
- **2x NVIDIA L4 on GCP**, driver 580.178.04, `python3.12` — the CUDA and
  distributed items (`gpu_verify.py`, `dist_verify.py`).

Both torch builds on the GPU host are **cu132**, the only CUDA build published
for 2.14.1 and 2.15.0 alike, so the comparison is not confounded by a CUDA
version change. The driver reports CUDA 13.0 and runs the cu132 wheels through
CUDA minor-version compatibility. ROCm and XPU claims are out of scope.

```bash
./run.sh                    # CPU/arm64: 2.14.1 vs 2.15.0
./setup_gpu.sh              # GPU host: build both cu132 venvs
~/pt2150/bin/python gpu_verify.py > results/gpu-torch-2.15.0.json
~/pt2150/bin/torchrun --nproc_per_node=2 dist_verify.py --out results/dist-torch-2.15.0.json
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

Float32 `sin`/`cos` moved to faster kernels with a looser bound.

| Condition | Value |
|---|---|
| Input | `linspace(-20, 20, 20001)` rounded to float32 (20,001 points) |
| Reference | the same float32 inputs evaluated in float64 |
| ULP size | `numpy.spacing(np.float32(reference))` — the float32 ULP |
| Filter | `abs(result) >= 0.5` (near a zero of `sin` the result is tiny while the argument-reduction error is not, so the ratio says nothing about kernel quality) |
| Points used | 13,194 for `sin`, 13,613 for `cos` |
| **Statistic quoted** | **maximum** |

| | 2.14.1 | 2.15.0 | Release note |
|---|---:|---:|---|
| `sin` | 0.7098 ULP | **1.5675 ULP** | 1.0 ULP -> 3.5 ULP |
| `cos` | 0.6707 ULP | **1.7700 ULP** | 1.0 ULP -> 3.5 ULP |

Maximum against maximum: both stay inside the documented 3.5 ULP bound, and
this sample never reached it. Mean and median are both under 0.04 ULP, so most
points match the correctly rounded float64 value bit for bit and the error is
concentrated in a few. `verify.py` reports max, mean, median and the point
counts together so the quoted figure cannot be mistaken for another statistic.
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

## CUDA and distributed (2x L4, cu132)

| Check | 2.14.1 | 2.15.0 |
|---|---|---|
| `torch.histc(x, out=)` with a different `out` dtype | no error | `RuntimeError: torch.histogram: input tensor and hist tensor should have the same dtype, but got input float and hist double` |
| `libtorch_cuda.so` links `libcurand` | yes (`libcurand.so.10`) | **no** |
| `torch.cuda.graph_annotations` members | `clear_kernel_annotations`, `get_kernel_annotations`, `is_available`, `mark_kernels` | adds `dump_kernel_py_stacks`, `get_kernel_py_stacks` |
| `all_gather(async_op=True).get_future().wait()` on **NCCL** | `[tensor(2, 4)]` — one stacked tensor | **`[tensor(4), tensor(4)]`** — one per rank |
| the same on **gloo** (`migration_check.py`, CPU) | `[tensor(4), tensor(4)]` | `[tensor(4), tensor(4)]` — unchanged |
| `all_reduce` on a tensor belonging to another rank's device | **no error** | `RuntimeError: Expected tensor on cuda:0 but found tensor on cuda:1` |

The `all_gather` change is the one most likely to break working code silently:
anything that took `wait()[0]` and treated it as the stacked result now gets
rank 0's slice instead.

**It is backend-specific.** gloo already returned a per-rank list in 2.14.1, so
2.15's NCCL moved to what gloo was already doing. The documented migration —
read the output list after `work.wait()` rather than the future — was run on
gloo against both versions and produced the full `(2, 4)` result each time
(`migration_check.py`). The `nccl-legacy` workaround in the notes was **not**
executed here.

The cross-device result deserves a caveat. Run on its own, 2.14.1 accepted the
wrong-device tensor and returned without error in 6 seconds. Run as part of the
full check sequence, the same 2.14.1 job hung and had to be killed, while 2.15.0
raised immediately. So "no error" there is not the same as "harmless"; the exact
trigger for the hang was not isolated.

## Claims that did not reproduce on CUDA

- **`linspace`/`logspace` precision for float16/bfloat16.** Identical output on
  CPU *and* on CUDA in both versions, for float16, bfloat16 and float32.
- **cuBLAS/cuBLASLt workspaces moving from cached to per-operation.** Reserved
  memory around repeated 2048x2048 matmuls was 70.0 MiB in both versions, and
  peak allocation matched to 0.01 MiB.
- **MAGMA deprecated for pivoted LU in favour of cuSOLVER.** `lu_factor` on a
  256x256 float64 matrix gave bit-identical factors, pivots and reconstruction
  error in both.
- **The rewritten NCCL backend becoming the default.** `get_backend()` returns
  `nccl` and `torch.cuda.nccl.version()` is 2.30.7 in both. Note that
  `dist.Backend("nccl-legacy")` resolving on 2.14.1 only reflects unknown names
  being accepted as-is, and is **not** evidence that the legacy implementation
  exists there. What was rewritten is PyTorch's C++ layer, not the NCCL library,
  so matching names and library versions cannot distinguish the two.
- **FSDP2 gradients moving to `grad_dtype` under `reduce_dtype=None`.** With
  fp32 parameters and `param_dtype=bfloat16`, gradients came back `float32` in
  both versions. The test probably cannot separate the two: `grad_dtype`
  defaults to the parameter's dtype, which is already fp32 here, so old and new
  behaviour coincide. A case where `grad_dtype` differs from the parameter dtype
  would be needed.

## Claims the CPU host could not confirm

Reported as-is rather than dropped, since "not observed" is a result:

- **`split` / `split_with_sizes` / `unbind` rebuilt as `slice`/`select`.** The
  note concerns *regenerated* outputs. In eager, and under
  `FunctionalTensorMode`, `grad_fn` stayed `SplitBackward0`,
  `SplitWithSizesBackward0` and `UnbindBackward0` in both versions.
- **`pipeline_per_direction_p2p` renamed to `pipeline_per_edge_p2p`.** Neither
  name is exported anywhere under `torch.distributed.pipelining` in either
  version.

## Files

| File | Purpose |
|---|---|
| `verify.py` | CPU/arm64 checks; prints one JSON document |
| `gpu_verify.py` | single-GPU CUDA checks |
| `dist_verify.py` | two-GPU distributed checks (`--skip`/`--only` to isolate one) |
| `migration_check.py` | the recommended `all_gather` migration, on gloo / CPU |
| `setup_gpu.sh` | builds both cu132 venvs on a CUDA host |
| `compare.py` | diffs two result files |
| `run.sh` | builds a venv per version, runs both, diffs |
| `results/` | raw output per version |
