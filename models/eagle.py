from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

def load_eagle(model_id="nvidia/Eagle2.5-8B"):
    model = AutoModel.from_pretrained(model_id,trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left"

    return model, processor

if __name__ == "__main__":
    model, processor = load_eagle()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://www.ilankelman.org/stopsigns/australia.jpg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text_list = [processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )]
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)