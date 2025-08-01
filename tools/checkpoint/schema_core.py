# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Core model schemas."""

import typing as T

from schema_base import ModelSchema


def get_core_transformer_block_key(model_key):
    return {
        "GPT" : "decoder",
        "BERT" : "encoder",
    }[model_key]


class CoreSchema(ModelSchema):

    def __init__(self, model_type, layer_schema):
        block_key = get_core_transformer_block_key(model_type)
        super().__init__({
            "embeddings" : {
                "pos" : "embedding.position_embeddings.weight",
                "word" : "embedding.word_embeddings.weight",
            },
            "layer_prefix" : f"{block_key}.layers",
            "layer" : layer_schema,
            "final_norm" : {
                "weight" : f"{block_key}.final_layernorm.weight",
                "bias" : f"{block_key}.final_layernorm.bias",
            },
            "output_layer" : {
                "weight" : "output_layer.weight",
            },
            "pooler" : {
                "weight" : "pooler.dense.weight",
                "bias" : "pooler.dense.bias",
            },
            "lm_head" : {
                "dense_weight" : "lm_head.dense.weight",
                "dense_bias" : "lm_head.dense.bias",
                "norm_weight" : "lm_head.layer_norm.weight",
                "norm_bias" : "lm_head.layer_norm.bias",
            },
            "binary_head" : {
                "weight" : "binary_head.weight",
                "bias" : "binary_head.bias",
            },
        })


class CoreLocalSchema(CoreSchema):

    def __init__(self, model_type):
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "input_layernorm.weight",
            "self_attn_norm_bias" : "input_layernorm.bias",
            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",
            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",
            "self_attn_q_layernorm_weight" : "self_attention.q_layernorm.weight",
            "self_attn_q_layernorm_bias" : "self_attention.q_layernorm.bias",
            "self_attn_k_layernorm_weight" : "self_attention.k_layernorm.weight",
            "self_attn_k_layernorm_bias" : "self_attention.k_layernorm.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias",
            # xielu weights
            "mlp_xielu_alpha_p" : "mlp.activation_func.alpha_p",
            "mlp_xielu_alpha_n" : "mlp.activation_func.alpha_n",
        })


class CoreTESchema(CoreSchema):

    def __init__(self, model_type):
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "input_layernorm.weight",
            "self_attn_norm_bias" : "input_layernorm.bias",
            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",

            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",
            "self_attn_q_layernorm_weight" : "self_attention.q_layernorm.weight",
            "self_attn_q_layernorm_bias" : "self_attention.q_layernorm.bias",
            "self_attn_k_layernorm_weight" : "self_attention.k_layernorm.weight",
            "self_attn_k_layernorm_bias" : "self_attention.k_layernorm.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",
            "mlp_fc1_weight" : "mlp.linear_fc1.weight",
            "mlp_fc1_bias" : "mlp.linear_fc1.bias",
            "mlp_fc2_weight" : "mlp.linear_fc2.weight",
            "mlp_fc2_bias" : "mlp.linear_fc2.bias",
            # xielu weights
            "mlp_xielu_alpha_p" : "mlp.activation_func.alpha_p",
            "mlp_xielu_alpha_n" : "mlp.activation_func.alpha_n",

        })


class CoreMoETESchema(CoreSchema):

    def __init__(self, model_type, num_experts, expert_model_parallel_size):
        num_local_experts = num_experts // expert_model_parallel_size
        super().__init__(model_type, layer_schema={

            # Self attention.
            "self_attn_norm_weight" : "self_attention.linear_qkv.layer_norm_weight",
            "self_attn_norm_bias" : "self_attention.linear_qkv.layer_norm_bias",

            "self_attn_qkv_weight" : "self_attention.linear_qkv.weight",
            "self_attn_qkv_bias" : "self_attention.linear_qkv.bias",

            "self_attn_proj_weight" : "self_attention.linear_proj.weight",
            "self_attn_proj_bias" : "self_attention.linear_proj.bias",
            "self_attn_q_layernorm_weight" : "self_attention.q_layernorm.weight",
            "self_attn_q_layernorm_bias" : "self_attention.q_layernorm.bias",
            "self_attn_k_layernorm_weight" : "self_attention.k_layernorm.weight",
            "self_attn_k_layernorm_bias" : "self_attention.k_layernorm.bias",

            # MLP.
            "mlp_norm_weight" : "pre_mlp_layernorm.weight",
            "mlp_norm_bias" : "pre_mlp_layernorm.bias",

            "router_weight" : "mlp.router.weight",

            **{f"mlp_fc1_weight.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.linear_fc1.weight" for expert_idx in range(num_local_experts) },
            **{f"mlp_fc2_weight.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.linear_fc2.weight" for expert_idx in range(num_local_experts) },
            
            # xielu weights
            **{f"mlp_xielu_alpha_p.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.activation_func.alpha_p.weight" for expert_idx in range(num_local_experts) },
            **{f"mlp_xielu_alpha_n.{expert_idx}" : f"mlp.experts.local_experts.{expert_idx}.activation_func.alpha_n.weight" for expert_idx in range(num_local_experts) },
        })


def get_model_schema(
    model_type: T.Literal["GPT", "BERT"],
    transformer_impl: T.Literal["transformer_engine", "local"],
    num_experts: T.Optional[int] = None,
    expert_model_parallel_size: T.Optional[int] = None,
) -> CoreSchema:
    if num_experts is not None and num_experts > 0:
        # Only support TE setter for MOE
        assert transformer_impl == "transformer_engine"
        assert isinstance(expert_model_parallel_size, int)
        return CoreMoETESchema(model_type, num_experts, expert_model_parallel_size)
    return {
        "local" : CoreLocalSchema,
        "transformer_engine" : CoreTESchema,
    }[transformer_impl](model_type)
