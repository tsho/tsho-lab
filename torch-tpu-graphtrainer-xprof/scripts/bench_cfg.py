"""Configurations for the GraphTrainer optimizer-compilation study.

Base model is llama3_1b with weight tying disabled, as in the previous
experiment: llama3_1b sets enable_weight_tying=True, which fails on TPU inside
Decoder.init_states() with "cannot assign torch.tpu.BFloat16Tensor as parameter
weight". Untying alone is not enough -- the tied config gives tok_embeddings
_EMBEDDING_SKIP_INIT, so an untied model starts uninitialised and the loss is
non-finite at step 1. The embedding init is restored too.

Configurations:

  A  eager                         plain TPUTrainer
  B  per-block torch.compile       plain TPUTrainer
  C  GraphTrainer, optimizer eager
  D  GraphTrainer, optimizer separate_graph
  E  GraphTrainer, optimizer in_graph   (the configuration the earlier ladder measured)
  F  per-block compile + optimizer separate_graph (plain TPUTrainer)
  C2 C + compile_grad_norm=True    }  isolates how much of E's gain is the
  D2 D + compile_grad_norm=True    }  gradient-norm path rather than the optimizer

Why C2/D2 exist: only the in_graph path skips _clip_grad_norm and clips inside
the traced graph instead (verified on this build: _train_step_optimizer_in_graph
never calls _clip_grad_norm). So E is not doing the same work as A-D, and part of
its speedup may be the grad-norm path. compile_grad_norm cannot be combined with
in_graph, so the comparison is made on C and D instead.
"""
from torchtitan.experiments.tpu.llama3 import config_registry as _r
from torchtitan.experiments.tpu.llama3.config_registry import (
    graph_trainer_model_registry,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
)
from torchtitan.models.llama3 import _EMBEDDING_INIT

try:  # the flag helper has moved between releases
    from torchtitan.experiments.tpu import tpu_flags
    _XLA_FLAGS = tpu_flags.merge_flags(xla_enable_hlo_sharding_v3=False)
except Exception:  # noqa: BLE001
    _XLA_FLAGS = {"xla_enable_hlo_sharding_v3": False}


def _base():
    """llama3_1b, untied, with a working embedding init."""
    config = _r.llama3_1b()
    model = config.model_spec.model
    model.enable_weight_tying = False
    model.tok_embeddings.param_init = _EMBEDDING_INIT
    return config


def _graph(optimizer_compile, memory_policy="default", compile_grad_norm=False,
           save_regions=(), base_fn=None):
    """Wrap the base config in a GraphTrainer config.

    xla_enable_hlo_sharding_v3=False mirrors the shipped qwen3_8b recipe: Shardy
    HLO sharding v3 is on by default from libtpu 0.0.48 and segfaults in
    mlir::sdy::buildTensorFactorShardings under GraphTrainer. This VM has 0.0.50.
    """
    base = (base_fn or _base)()
    base.xla_flags = _XLA_FLAGS
    config = TpuGraphTrainer.Config.from_tpu_trainer_config(
        base, graph_trainer_model_registry
    )
    config.compile = TpuGraphTrainerCompileConfig(
        enable=True,
        components=["model"],
        mode="aot_fx_trace",          # required by optimizer_compile="in_graph"
        memory_policy=memory_policy,
        optimizer_compile=optimizer_compile,
        compile_grad_norm=compile_grad_norm,
        save_regions=tuple(save_regions),
    )
    return config


# --- plain TPUTrainer configurations -------------------------------------
def cfg_a_eager():
    return _base()


def cfg_b_perblock():
    config = _base()
    config.compile.enable = True
    # Without this the default backend is inductor, which torch_tpu rejects:
    # "torch.compile(..., backend='inductor') on TPU is experimental and does
    # not yet have full TPU support". The GraphTrainer config defaults to "tpu";
    # the plain TPUTrainer one does not.
    config.compile.backend = "tpu"
    return config


def cfg_f_perblock_separate():
    config = _base()
    config.compile.enable = True
    config.compile.backend = "tpu"      # see cfg_b_perblock
    config.compile.optimizer_compile = "separate_graph"
    return config


# --- GraphTrainer configurations -----------------------------------------
def cfg_c_gt_eager():
    return _graph("eager")


def cfg_d_gt_separate():
    return _graph("separate_graph")


def cfg_e_gt_ingraph():
    return _graph("in_graph")


def cfg_c2_gt_eager_cgn():
    return _graph("eager", compile_grad_norm=True)


def cfg_d2_gt_separate_cgn():
    return _graph("separate_graph", compile_grad_norm=True)


# --- memory policies, on the in_graph configuration ----------------------
def cfg_mp_default():
    return _graph("in_graph", memory_policy="default")


def cfg_mp_full():
    return _graph("in_graph", memory_policy="full")


def cfg_mp_tpu_selective():
    # tpu_selective refuses to start without a region: "needs a region in
    # save_regions or save_last_n_layers; known regions: attention.inner_attention,
    # feed_forward.w2, moe.ep_dispatch, moe.routing".
    return _graph("in_graph", memory_policy="tpu_selective",
                  save_regions=("feed_forward.w2",))


def cfg_mp_full_and_offload():
    return _graph("in_graph", memory_policy="full_and_offload")


# --- Llama 3 8B on v6e-8 --------------------------------------------------
def _base_8b():
    """llama3_8b, with the same weight-tying workaround if the recipe ties."""
    config = _r.llama3_8b()
    model = config.model_spec.model
    if getattr(model, "enable_weight_tying", False):
        model.enable_weight_tying = False
        model.tok_embeddings.param_init = _EMBEDDING_INIT
    return config


def cfg_8b_a_eager():
    return _base_8b()


def cfg_8b_b_perblock():
    config = _base_8b()
    config.compile.enable = True
    config.compile.backend = "tpu"
    return config


def cfg_8b_c_gt_eager():
    return _graph("eager", base_fn=_base_8b)


def cfg_8b_e_gt_ingraph():
    return _graph("in_graph", base_fn=_base_8b)


# 8B GraphTrainer does not fit with memory_policy="default" on v6e-8 at 2048
# tokens/device (compile-time HBM OOM: 35.09G of HLO temporaries against 31.24G).
# These try the policies that cut HBM on the 1B model.
def cfg_8b_e_offload():
    return _graph("in_graph", memory_policy="full_and_offload", base_fn=_base_8b)


def cfg_8b_c_offload():
    return _graph("eager", memory_policy="full_and_offload", base_fn=_base_8b)


def cfg_8b_e_full():
    return _graph("in_graph", memory_policy="full", base_fn=_base_8b)
