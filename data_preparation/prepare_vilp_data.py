from datasets import load_dataset
from io import BytesIO
import base64
import os
import json

def extract_samples(split):
    def convert_image_to_base64(image):
        format = image.format
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_url = f"data:image/{format.lower()};base64,{img_base64}"
        return img_url
    
    samples = []
    for sample in split:
        image1 = convert_image_to_base64(sample["image1"])
        image2 = convert_image_to_base64(sample["image2"])
        image3 = convert_image_to_base64(sample["image3"])
        samples.append({
            "question": sample["question"],
            "answers": [sample["answer1"], sample["answer2"], sample["answer3"]],
            "images": [image1, image2, image3],
        })
    return samples


if __name__ == "__main__":
    save_base_dir = os.getenv("DATA_PATH")
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("ViLP/ViLP")
    
    samples = []
    samples.extend(extract_samples(ds["train"]))
    print(len(samples))
    with open(os.path.join(save_base_dir, "vilp.jsonl"), "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Data saved to {os.path.join(save_base_dir, 'vilp.jsonl')}")