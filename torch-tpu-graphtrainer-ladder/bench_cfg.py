"""Benchmark config: llama3_1b with weight tying disabled.

llama3_1b sets enable_weight_tying=True, which fails on TPU inside
Decoder.init_states():
    TypeError: cannot assign torch.tpu.BFloat16Tensor as parameter weight
Untying alone is not enough: the tied config gives tok_embeddings
_EMBEDDING_SKIP_INIT (the weight is meant to be re-pointed at lm_head), so an
untied model starts uninitialized and the loss is non-finite at step 1. We
therefore also restore a real embedding init. Everything else is llama3_1b.
"""
from torchtitan.experiments.tpu.llama3 import config_registry as _r
from torchtitan.models.llama3 import _EMBEDDING_INIT


def llama3_1b_untied():
  config = _r.llama3_1b()
  model = config.model_spec.model
  model.enable_weight_tying = False
  model.tok_embeddings.param_init = _EMBEDDING_INIT
  return config
