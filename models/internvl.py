from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

def load_internvl(model_name = "OpenGVLab/InternVL3-8B-hf", attn_implementation = "eager"):
    model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor