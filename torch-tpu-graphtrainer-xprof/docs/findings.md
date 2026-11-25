# Findings

All throughput figures are tokens per second per chip, taken as the median of
steps 11-30 of a 30-step run. Unless stated otherwise the model is the
TorchTitan `llama3_1b` recipe on a Cloud TPU v6e-4 (four chips, FSDP through
`--parallelism.use_simple_fsdp`), sequence length 2048, and TorchTitan
`experiments/tpu` at commit `e78271f7`. Each configuration was run once. Differences below about 1% are treated here
as small in practice, but run-to-run variance was not evaluated: none of the
differences is shown to be statistically significant, and the 1.6%
difference in section 1 may also lie within that variance. The same
configuration measured in two separate runs differed by 0.6% (E: 11,772 and
11,843). Models are trained from random initialization; no pretrained
weights are used. "Current commit" means TorchTitan main as of 2026-11-23.

Each section separates what was observed from what is inferred.

## 1. Throughput by optimizer compilation mode

2048 tokens per device per step.

| ID | Trainer | `optimizer_compile` | tok/s/chip | vs A | Allocator peak (GiB) |
|---|---|---|---:|---:|---:|
| A | TPUTrainer, eager | eager | 6,296 | 1.00x | 0.98 |
| B | TPUTrainer, per-block compile | eager | 10,464 | 1.66x | 0.98 |
| F | TPUTrainer, per-block compile | `separate_graph` | 10,403 | 1.65x | 0.98 |
| C | TpuGraphTrainer | eager | 11,581 | 1.84x | 13.40 |
| D | TpuGraphTrainer | `separate_graph` | 11,528 | 1.83x | 13.40 |
| E | TpuGraphTrainer | `in_graph` | 11,772 | 1.87x | 13.42 |
| C2 | C + `compile_grad_norm=True` | eager | 11,647 | 1.85x | 13.40 |
| D2 | D + `compile_grad_norm=True` | `separate_graph` | 11,574 | 1.84x | 13.40 |

Observed:

- Moving the optimizer into a separate compiled graph shows no clear
  improvement on either trainer (B to F: -0.6%, C to D: -0.5%).
- Moving the optimizer into the training graph adds 1.6% over the graph
  trainer with an eager optimizer (C to E).
- Compiling the gradient-norm computation adds 0.6% (C to C2) and 0.4%
  (D to D2).
- With `in_graph`, the logged `grad_norm` is 0.0000 on every step. The
  `in_graph` step function does not call the Python-side clipping helper
  that produces the logged value. From reading the source, clipping is
  instead computed inside the traced graph
  (`clip_coef = max_norm / (total_norm + 1e-6)`), so the zero is a logging
  artefact rather than evidence that clipping is skipped. The in-graph
  clipping was not checked numerically. Final losses after 30 steps are
  3.3629 (C) and 3.3645 (E).

## 2. Comparison with the earlier ladder

The earlier measurement (TorchTitan `89ef29b5`) reported 10,339 for per-block
compile, 10,584 for the graph trainer, and 13,131 with the optimizer in the
graph: a 24% step for the optimizer and 2.06x over eager in total.

| Configuration | Published (`89ef29b5`) | `89ef29b5` re-run | `e78271f7` |
|---|---:|---:|---:|
| Per-block compile | 10,339 | 9,497 | 10,464 |
| Graph trainer, eager optimizer | 10,584 | 9,901 | 11,581 |
| Graph trainer, optimizer in graph | 13,131 | 11,838 | 11,772 |
| Optimizer step (last row / row above) | +24.1% | +19.6% | +1.6% |

Observed:

- On the old commit the gain from moving the optimizer into the graph
  reproduces (about 20%). Eager was not re-measured on the old commit, so
  the 2.06x over eager itself was not reproduced.
- On the current commit the graph trainer with an eager optimizer has caught
  up, and the optimizer step has shrunk to 1.6%. The best configuration is
  1.87x over eager, not 2.06x. Which change between the two commits made
  the graph trainer alone faster was not identified.
- The old commit re-run is about 6-10% below the published numbers in every
  row (-8.1%, -6.5%, -9.9%).

Not explained: the 6-10% drop on the same TorchTitan commit. The earlier
run recorded Python and seven packages. Python (3.14.10), torch
(2.15.0+cpu), torch_tpu (0.2.0) and libtpu (0.0.50) match that record. Four
of the recorded packages differ in the environment of the re-run: tokamax,
the kernel library (0.0.14 then, 0.0.17 now), numpy (2.5.4, 2.5.5),
transformers (5.21.0, 5.24.1) and tyro (1.0.16, 1.0.17). Whether any of
these, or a package the earlier run did not record, causes the drop was not
tested.

## 3. What the profiles show (2048 tokens per device)

Profiles were captured with the TorchTitan built-in profiler
(five active steps, steps 15-19). Values are per step: the median over the
profiled steps on each rank, then the median across the four ranks. Step time
under the profiler was 0.6-1.2% longer than without it.

| Metric | B per-block | C graph | D graph + separate | E graph + in_graph |
|---|---:|---:|---:|---:|
| Step time without profiler (ms) | 195.7 | 176.8 | 177.7 | 174.0 |
| Device programs per step | 110 | 34 | 34 | 6 |
| Device busy in programs (ms) | 189.0 | 167.6 | 169.2 | 165.2 |
| Device idle between programs (ms) | 8.8 | 9.1 | 10.6 | 11.7 |
| Device idle (% of step) | 4.5 | 5.1 | 5.8 | 6.6 |
| Synchronous collectives (ms) | 75.2 | 67.2 | 67.1 | 67.6 |
| Host time in PJRT execute calls, excluding queue wait (ms) | 9.1 | 3.3 | 3.7 | 1.7 |
| Host waiting for an execution-queue slot (ms) | 142.7 | 0.03 | 0.02 | 0.01 |
| Host blocked in `.item()` (ms) | 165.9 | 157.4 | 145.9 | 146.6 |
| Optimizer program on device (ms) | 8.16 | 8.09 | 8.15 | in the step graph |

Observed:

- The number of device programs per step falls from 110 to 34 to 6, and the
  device idle time between programs does not fall with it. It rises slightly,
  from 8.8 ms to 11.7 ms.
- The step gets shorter because the device spends less time executing
  programs: 189.0 ms for per-block compile against 165-169 ms for the graph
  trainer.
- The eager optimizer and the separately compiled optimizer take the same
  time on the device (8.09 ms and 8.15 ms).
- Collective operations (all-gather, reduce-scatter and the fused
  reduce-scatter) occupy 67-75 ms of each step, about 38% of the step.
- In every configuration the host main thread spends most of the step blocked
  in a device-to-host scalar read.

- Host time in dispatch calls, excluding queue wait, falls by about 7 ms
  from B to E. In per-block mode the host spends 142.7 ms per step waiting
  for a free execution-queue slot, i.e. it is ahead of the device.

How the figures are computed: device execution time is the length of the
union of program intervals within the host-side step window, so overlapping
intervals are not counted twice. Idle is the step window minus that union.
Synchronous collectives lie inside program execution time and are not a
separate quantity. Step windows come from host-side timestamps; any offset
from device-side step boundaries was not evaluated.

Inferred:

- The hypothesis that the speedup comes from less device idle time between
  programs is not supported: idle does not fall as the program count falls.
- Idle time between programs is not the same thing as host-side dispatch
  cost. Host work can overlap device execution without appearing as device
  idle. Dispatch-call time did fall by about 7 ms; how much of the roughly
  22 ms shorter step (B to E) it accounts for was not separated, so it
  cannot be concluded that host dispatch is not a bottleneck.
- What corresponds to the per-block to graph-trainer gain is device
  execution time. In per-block mode, 59 small eager programs take 21.9 ms
  and 12 collective programs take 32.0 ms; in the graph trainer those are
  0.2 ms and 2.8 ms, with the collectives running inside the single
  forward/backward graph. The profiles show the reduction in time; which
  compiler-level changes produce it (fusion, fewer copies, different
  collective scheduling) was not determined.
- That `separate_graph` shows no improvement is consistent with the eager
  optimizer already being a single device program of the same duration.

Not obtainable:

- MXU utilisation: the profile reports 0.0 for every file; hardware
  performance counters were not captured.
- A breakdown of host-side Python time: Python stack tracing was off, so host
  time outside recorded operators is not attributed.

## 4. Allocator metric versus device memory

TorchTitan logs a peak-memory figure per step. Device HBM was sampled
separately with `tpu-info` every two seconds during the run, and the XProf
memory profile gives heap and stack-reserved figures.

| Configuration | TorchTitan log (GiB) | XProf stack reserved (GiB) | XProf peak in use (GiB) | `tpu-info` peak (GiB) |
|---|---:|---:|---:|---:|
| B per-block | 0.98 | 0.98 | 15.01 | 11.09 |
| C graph | 13.40 | 13.40 | 19.31 | not sampled |
| D graph + separate | 13.40 | 13.40 | 23.50 | not sampled |
| E graph + in_graph | 13.42 | 13.42 | 21.87 | 21.87 |

HBM capacity is 31.24 GiB per chip.

Observed:

- The figure in the TorchTitan log numerically equals the stack-reserved
  (program scratch) part of the XProf memory profile in all four
  configurations. It is not the total HBM in use. Whether the two refer to
  the same internal region was not confirmed in the implementation.
- The device HBM ratio between the graph trainer with the optimizer in the
  graph and per-block compile is about 1.97x by `tpu-info` sampled peak
  (21.87 against 11.09 GiB) and about 1.46x by XProf peak (21.87 against
  15.01 GiB). Both are well below the 13.7x that the logged figure suggests
  (13.42 against 0.98 GiB).
- For per-block compile the two-second `tpu-info` sampling reads 11.09 GiB
  while XProf reports a 15.01 GiB peak. The two were taken in different runs,
  and a two-second sampler can miss a short peak.

## 5. Memory policies (graph trainer, optimizer in graph)

| `memory_policy` | tok/s/chip | vs default | Allocator peak (GiB) | `tpu-info` peak (GiB) |
|---|---:|---:|---:|---:|
| `default` | 11,843 | - | 13.42 | 21.87 |
| `full` | 11,806 | -0.3% | 13.42 | 21.87 |
| `tpu_selective` (`save_regions=feed_forward.w2`) | 11,811 | -0.3% | 13.42 | 21.87 |
| `full_and_offload` | 11,020 | -6.9% | 5.21 | 13.97 |

Observed: no clear difference for `full` and `tpu_selective` under these
conditions. The `default` row is configuration E measured in a separate run. `full_and_offload` frees about 8 GiB of HBM for a 7% loss in
throughput.

Notes: `tpu_selective` requires `save_regions`. Only the region
`feed_forward.w2` was measured.

## 6. Batch size (8192 tokens per device)

| ID | tok/s/chip | vs A | `tpu-info` peak (GiB) |
|---|---:|---:|---:|
| A eager | 11,272 | 1.00x | 17.72 |
| B per-block | 15,498 | 1.37x | 28.09 |
| C graph | 18,785 | 1.67x | 31.06 |
| E graph + in_graph | 18,807 | 1.67x | 28.14 |

| Profile metric | B per-block | C graph | E graph + in_graph |
|---|---:|---:|---:|
| Step time without profiler (ms) | 528.6 | 436.1 | 435.6 |
| Device programs per step | 110 | 34 | 6 |
| Device idle between programs (ms) | 15.3 | 16.2 | 17.6 |
| Device idle (% of step) | 2.9 | 3.7 | 4.0 |
| Synchronous collectives (ms) | 82.4 | 71.1 | 71.8 |

Observed:

- At four times the batch, the optimizer-in-graph step disappears (C and E
  differ by 0.1%), while the per-block to graph-trainer gap grows from 11% to
  21%.
- Collective time barely changes with batch size (67 to 71 ms), so its share
  of the step falls from about 38% to about 16%.
- The graph trainer with an eager optimizer peaks at 31.06 GiB of a
  31.24 GiB chip at this batch size.

## 7. Llama 3 8B on v6e-8

Eight chips, 2048 tokens per device, Spot capacity.

| Configuration | Result | tok/s/chip | `tpu-info` peak (GiB) |
|---|---|---:|---:|
| Per-block compile | completed | 2,537 | 26.78 |
| Graph trainer + `in_graph`, `default` | compile-time HBM error (35.09 GiB of temporaries against 31.24 GiB) | - | - |
| Graph trainer + `in_graph`, `full` | same compile-time HBM error | - | - |
| Graph trainer + eager optimizer, `default` | runtime `RESOURCE_EXHAUSTED` | - | - |
| Graph trainer + eager optimizer, `full_and_offload` | runtime allocation failure | - | - |
| Graph trainer + `in_graph`, `full_and_offload` | completed | 2,444 | 31.22 |

Observed:

- With the default memory policy the graph trainer did not fit on a v6e
  chip for this model at this batch size. Only the configurations in the
  table were tried.
- The one graph-trainer configuration that ran is 3.7% slower than per-block
  compile and peaks within 0.02 GiB of HBM capacity.
- There is no eager baseline for 8B. The `llama3_8b` recipe enables per-block
  compile by default, so the run intended as the eager baseline measured the
  same configuration as the per-block run (2,536 and 2,537).

The 1B ordering (graph trainer ahead of per-block compile) therefore does not
carry over to 8B on this hardware. Smaller batches or other memory policies
were not explored.

### Profile of the 8B per-block run

Only the per-block run was profiled on 8B (eight ranks, same profiler
settings as section 3). The two graph-trainer profile runs, with the
`default` memory policy, failed with the HBM errors above; the
`full_and_offload` configuration that did run was not profiled. Values are per step, median over the profiled
steps on each rank, then the median across the eight ranks.

| Metric | 8B per-block |
|---|---:|
| Step time without profiler (ms) | 807.3 |
| Device programs per step | 94 |
| Device idle between programs (ms) | 0.6 |
| Device idle (% of step) | 0.1 |
| Synchronous collectives (ms) | 481.8 |
| Synchronous collectives (% of step without profiler) | 59.7 |
| XProf peak HBM in use (GiB) | 27.58 |
| XProf stack reserved (GiB) | 2.98 |

Observed:

- The device is almost never idle between programs (0.6 ms of an
  800 ms step), against 8.8 ms of a 196 ms step for 1B per-block.
- Synchronous collectives take about 60% of the step, against about 38%
  for 1B per-block at the same tokens per device.
- The TorchTitan memory figure (2.98 GiB) again equals the XProf
  stack-reserved part; XProf peak in use is 27.58 GiB (`tpu-info` read
  26.78 GiB in the separate timing run).

## 8. Hypotheses

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Putting the optimizer in the graph gives about +24% | Supported on `89ef29b5` (+20%), not on `e78271f7` (+1.6%) | Section 2 |
| The gain comes from less device idle time between programs | Not supported; host dispatch cost itself is not separated | Section 3: programs 110 to 6, idle 8.8 to 11.7 ms |
| `separate_graph` recovers part of the `in_graph` gain | Not supported | Sections 1 and 3 |
| Compiling the gradient norm matters | Not supported (+0.4-0.6%) | Section 1 |
| The logged memory figure is device HBM occupancy | Not supported | Section 4 |
| A larger batch reduces the relative dispatch overhead | Partly: idle share falls from 4.5-6.6% to 2.9-4.0%, but idle was never the main cost | Section 6 |
| The graph trainer's advantage holds for 8B | Not supported on v6e-8 at 2048 tokens per device | Section 7 |

## 9. Not verified

- The cause of the 6-10% drop when re-running the old commit.
- Which compiler-level changes make the single graph cheaper on the device.
- That the in-graph gradient clipping of `in_graph` is numerically
  equivalent to the clipping used by the other modes (read from source only).
- Run-to-run variance: every configuration was run once.
- MXU utilisation and host Python-level time.
- 8B with a smaller batch.
- A profile of any 8B graph-trainer run: none exists (the `default`-policy
  profile runs failed and the `full_and_offload` run was not profiled).
- Any hardware other than v6e.
- How much host-side dispatch contributes to step time.
- The effect of host/device step-boundary offset on the idle figure.
- Whether parameter updates are correct with `in_graph`.
- Whether collective time on 8B is driven by model size or by chip count
  (1B ran on four chips, 8B on eight).
