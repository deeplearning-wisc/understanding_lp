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
        image = convert_image_to_base64(sample["image"])
        samples.append({
            "question": sample["question"],
            "answer": sample["answer"],
            "image": image,
        })
    return samples

if __name__ == "__main__":
    save_base_dir = os.getenv("DATA_PATH")
    ds = load_dataset("Lin-Chen/MMStar")
    samples = []
    samples.extend(extract_samples(ds["val"]))
    print(len(samples))
    with open(os.path.join(save_base_dir, "mmstar.jsonl"), "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Data saved to {os.path.join(save_base_dir, 'mmstar.jsonl')}")