from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

def load_llavanext(model_id="llava-hf/llava-v1.6-vicuna-7b-hf"):
    processor = LlavaNextProcessor.from_pretrained(model_id)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto") 

    return model, processor

if __name__ == "__main__":
    model, processor = load_llavanext()

    # prepare image and text prompt, using the appropriate prompt template
    # url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    path = "YOUR_SAMPLE_IMAGE.jpeg"
    image = Image.open(path)

    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": "What is shown in this image?"},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))