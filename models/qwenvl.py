from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import os

def load_qwenvl(model_name="Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="eager", min_pixels=256*28*28, max_pixels=1280*28*28):
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", attn_implementation=attn_implementation
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    
    return model, processor

def load_qwenvl_lmhead(model_name="Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="eager", min_pixels=256*28*28, max_pixels=1280*28*28):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", attn_implementation=attn_implementation
    )
    lm_head = model.lm_head
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    return lm_head, processor

# Qwen2_5_VLForConditionalGeneration(
#   (model): Qwen2_5_VLModel(
#     (visual): Qwen2_5_VisionTransformerPretrainedModel(
#       (patch_embed): Qwen2_5_VisionPatchEmbed(
#         (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
#       )
#       (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
#       (blocks): ModuleList(
#         (0-31): 32 x Qwen2_5_VLVisionBlock(
#           (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
#           (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
#           (attn): Qwen2_5_VLVisionAttention(
#             (qkv): Linear(in_features=1280, out_features=3840, bias=True)
#             (proj): Linear(in_features=1280, out_features=1280, bias=True)
#           )
#           (mlp): Qwen2_5_VLMLP(
#             (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
#             (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
#             (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
#             (act_fn): SiLU()
#           )
#         )
#       )
#       (merger): Qwen2_5_VLPatchMerger(
#         (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
#         (mlp): Sequential(
#           (0): Linear(in_features=5120, out_features=5120, bias=True)
#           (1): GELU(approximate='none')
#           (2): Linear(in_features=5120, out_features=3584, bias=True)
#         )
#       )
#     )
#     (language_model): Qwen2_5_VLTextModel(
#       (embed_tokens): Embedding(152064, 3584)
#       (layers): ModuleList(
#         (0-27): 28 x Qwen2_5_VLDecoderLayer(
#           (self_attn): Qwen2_5_VLAttention(
#             (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
#             (k_proj): Linear(in_features=3584, out_features=512, bias=True)
#             (v_proj): Linear(in_features=3584, out_features=512, bias=True)
#             (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
#             (rotary_emb): Qwen2_5_VLRotaryEmbedding()
#           )
#           (mlp): Qwen2MLP(
#             (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
#             (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
#             (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
#           (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
#         )
#       )
#       (norm): Qwen2RMSNorm((3584,), eps=1e-06)
#       (rotary_emb): Qwen2_5_VLRotaryEmbedding()
#     )
#   )
#   (lm_head): Linear(in_features=3584, out_features=152064, bias=False)
# )