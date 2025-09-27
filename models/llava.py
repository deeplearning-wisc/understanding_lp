import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_llava(model_id="llava-hf/llava-1.5-7b-hf", attn_implementation="eager"):
    # Load the model in half-precision
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        attn_implementation=attn_implementation
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor

# LlavaForConditionalGeneration(
#   (model): LlavaModel(
#     (vision_tower): CLIPVisionModel(
#       (vision_model): CLIPVisionTransformer(
#         (embeddings): CLIPVisionEmbeddings(
#           (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
#           (position_embedding): Embedding(577, 1024)
#         )
#         (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#         (encoder): CLIPEncoder(
#           (layers): ModuleList(
#             (0-23): 24 x CLIPEncoderLayer(
#               (self_attn): CLIPAttention(
#                 (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
#                 (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
#                 (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
#                 (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
#               )
#               (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#               (mlp): CLIPMLP(
#                 (activation_fn): QuickGELUActivation()
#                 (fc1): Linear(in_features=1024, out_features=4096, bias=True)
#                 (fc2): Linear(in_features=4096, out_features=1024, bias=True)
#               )
#               (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#         )
#         (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#       )
#     )
#     (multi_modal_projector): LlavaMultiModalProjector(
#       (linear_1): Linear(in_features=1024, out_features=4096, bias=True)
#       (act): GELUActivation()
#       (linear_2): Linear(in_features=4096, out_features=4096, bias=True)
#     )
#     (language_model): LlamaModel(
#       (embed_tokens): Embedding(32064, 4096)
#       (layers): ModuleList(
#         (0-31): 32 x LlamaDecoderLayer(
#           (self_attn): LlamaAttention(
#             (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
#             (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
#             (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           )
#           (mlp): LlamaMLP(
#             (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
#             (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
#             (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
#           (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
#         )
#       )
#       (norm): LlamaRMSNorm((4096,), eps=1e-05)
#       (rotary_emb): LlamaRotaryEmbedding()
#     )
#   )
#   (lm_head): Linear(in_features=4096, out_features=32064, bias=False)
# )