import json
import os

data_path = os.getenv("DATA_PATH")

# usage:
# prepare your data in the data_path, with the naming convention: {dataset}.jsonl
# each line in the jsonl file should be a dictionary with the following keys:
# - image: the image path
# - instruction: the instruction
# - target_tokens: the target tokens (e.g. ["Yes", "No"] or ["A", "B", "C", "D"])
# - other keys: other keys you want to add

def get_dataset(dataset):
    samples = []
    data_info = []
    with open(os.path.join(data_path, f"{dataset}.jsonl"), "r") as f:
        for line in f:
            sample = json.loads(line)
            samples.append({
                "image": sample["image"],
                "instruction": sample["instruction"],
            })
            data_info.append(sample)
            target_tokens = sample["target_tokens"]
    
    print(f"loaded {len(samples)} samples")
    
    return samples, data_info, target_tokens