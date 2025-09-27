from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def load_gemma(model_name = "google/gemma-3-4b-it", attn_implementation = "eager"):
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", attn_implementation=attn_implementation
    ).eval()

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor
