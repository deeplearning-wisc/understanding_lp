import ast
from IPython.display import display
from PIL import Image
import base64
from io import BytesIO
from datasets import load_dataset
import os
import json

def extract_samples(split):
    def count_image_num(sample):
        keys = ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]
        num = 0
        for key in keys:
            if sample[key] is not None:
                num += 1
        return num
    
    def remove_image_tag_from_question(question):
        # sample question: '<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?'
        question = question.replace("<image 1>", "").replace("<image 2>", "").replace("<image 3>", "").replace("<image 4>", "").replace("<image 5>", "").replace("<image 6>", "").replace("<image 7>", "")
        return question

    def split_options(options):
        # sample options: "['$6', '$7', '$8', '$9']"
        options = ast.literal_eval(options)
        return options
    
    def convert_image_to_base64(image):
        format = image.format
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_url = f"data:image/{format.lower()};base64,{img_base64}"
        return img_url
        
    samples = []
    for sample in split:
        new_sample = {}
        image_num = count_image_num(sample)
        if image_num > 1:
            continue
        options = split_options(sample["options"])
        if not len(options):
            continue
        new_sample["question"] = remove_image_tag_from_question(sample["question"])
        new_sample["options"] = options
        new_sample["answer"] = sample["answer"]
        new_sample["image"] = convert_image_to_base64(sample["image_1"])
        samples.append(new_sample)

    return samples

if __name__ == "__main__":
    save_base_dir = os.getenv("DATA_PATH")
    
    disciplines =['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']

    all_samples = []

    for discipline in disciplines:
        ds = load_dataset("MMMU/MMMU", discipline)
        samples = []
        samples.extend(extract_samples(ds['validation']))
        # samples.extend(extract_samples(ds['dev']))
        all_samples.extend(samples)

    print(len(all_samples))
    
    with open(os.path.join(save_base_dir, "mmmu.jsonl"), "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Data saved to {os.path.join(save_base_dir, 'mmmu.jsonl')}")