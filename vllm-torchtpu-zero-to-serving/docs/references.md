# References

Every URL below was fetched and checked on 2026-11-24. Where a page contradicts
another, both are recorded rather than reconciled.

## vllm-torchtpu

| What | URL |
|---|---|
| Repository | https://github.com/vllm-project/vllm-torchtpu |
| Releases (v0.33.0, 2026-11-17, tag commit `46ae4b7`) | https://github.com/vllm-project/vllm-torchtpu/releases |
| Installation | https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/getting_started/installation/ |
| TPU setup | https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/getting_started/tpu_setup/ |
| Quickstart | https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/getting_started/quickstart/ |
| **v6e recipes** (the configuration used here) | https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/deployment/v6e_recipes/ |
| Supported models | https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/user_guide/supported_models/ |
| Compilation cache | https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/user_guide/compilation_cache/ |
| Environment variables | https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/reference/env_vars/ |
| v6e nightly support matrix | https://github.com/vllm-project/vllm-torchtpu/blob/main/support_matrices/nightly/v6e/model_support_matrix.csv |

## Announcements

| What | URL |
|---|---|
| Introducing vLLM TorchTPU (2026-10-20) | https://vllm.ai/blog/2026-10-20-vllm-torchtpu |
| Google: new vLLM and SGLang inference engines on TPU (2026-10-20) | https://developers.googleblog.com/announcing-new-vllm-and-sglang-inference-engines-on-tpu/ |
| vLLM JAX (the former `tpu-inference`) | https://docs.vllm.ai/projects/tpu/en/latest/ |

## Google Cloud

| What | URL |
|---|---|
| Run inference on Cloud TPU | https://docs.cloud.google.com/tpu/docs/tpu-inference |
| PyTorch on Cloud TPU | https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu |
| TPU v6e (topologies, machine types, HBM) | https://docs.cloud.google.com/tpu/docs/v6e |
| TPU runtime versions (`v2-alpha-tpuv6e`) | https://docs.cloud.google.com/tpu/docs/runtimes |
| Regions and zones | https://docs.cloud.google.com/tpu/docs/regions-zones |
| Create a TPU VM (`--spot`) | https://docs.cloud.google.com/tpu/docs/create-tpu-vm |
| Pricing | https://cloud.google.com/tpu/pricing |

## URLs that do not resolve

Recorded because they are easy to reach for by analogy and both waste time.

| URL | Result |
|---|---|
| `https://docs.vllm.ai/projects/torchtpu/en/latest/` | 404. The project slug is `vllm-torchtpu`, not `torchtpu` |
| `https://docs.cloud.google.com/tpu/docs/spot-vms` | 404 |
| `https://docs.cloud.google.com/ai-hypercomputer/docs/tutorials/tpu/serve-qwen3-8b-base` | HTTP 200, but serves the tutorials index. There is no Qwen3-8B tutorial |

## Documented facts that conflict

1. **v6e support status.** The repository README lists TPU v6e as "Supported — not
   covered by CI". The launch blog calls v6e "experimental". The v6e recipes page
   describes its own commands as "reference configuration examples sized for v6e
   HBM capacity", unlike the v7x recipes, which "mirror dedicated CI benchmark
   scripts".

2. **Compilation cache variables.** The compilation-cache page names
   `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` (native PJRT binaries) and
   `VLLM_CACHE_ROOT` (Python-level artifacts), and calls them "completely
   independent". The environment-variable reference, the quickstart and the
   supported-models page instead use `VLLM_XLA_CACHE_PATH`. No page states how
   the three relate.

3. **Provisioning API.** The vllm-torchtpu TPU setup page routes TPU v6e through
   the Compute Engine API (`gcloud compute instances create`) and offers only
   Standard and Flex-start capacity, never mentioning Spot. The Cloud TPU API
   documents `--spot` but carries a notice that the API is "no longer under
   active development".

4. **Model validation.** The v6e-1 recipe names `Qwen/Qwen3-8B` as a recommended
   model, but that checkpoint appears in neither the Validated Models table (CI
   covers v7x only) nor the v6e nightly support matrix, whose single row is
   `Qwen/Qwen3.5-397B-A17B` marked untested in every column.
