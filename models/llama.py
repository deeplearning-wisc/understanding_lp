import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image


def load_llama(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor

if __name__ == "__main__":
    model, processor = load_llama()
    
    print(model.config._attn_implementation)
    print(model)
    
    path = "YOUR_SAMPLE_IMAGE.jpg"
    image = Image.open(path).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]}
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    print(input_text)
    
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)
    print(processor.decode(output[0]))