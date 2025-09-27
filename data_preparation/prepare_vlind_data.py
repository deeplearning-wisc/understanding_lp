from utils.api import get_response
import json
import os
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()

data_path = os.getenv("DATA_PATH")

vlind_data_path = os.path.join(data_path, "VLind-Bench/VLind-Bench-Dataset/data.json")
vlind_imgs_dir = os.path.join(data_path, "VLind-Bench/VLind-Bench-Dataset/images")
save_base_dir = data_path

def form_instruction(data):
    filtered_data = []
    for sample in data:
        # if sample['concept'] not in ['habitat', 'history', 'landmark', 'location']:
        #     continue
        image_path = os.path.join(vlind_imgs_dir, "counterfactual", sample['concept'], f"{sample['context_id']}_{sample['context']}", f"{sample['best_img_id']}.jpg")
        if not os.path.exists(image_path):
            continue
        filtered_data.append({
            "true_statement": sample['true_statement'],
            "false_statement": sample['false_statement'],
            "concept": sample['concept'],
            "image_path": image_path,
            "object": sample['existent_noun']
        })
    return filtered_data

def generate_question(data):
    instruction = "Generate a question based on the counterfactual information in the given statement. The question should be answered by yes.\n Here are some examples:\n Statement: The Statue of Liberty is holding a sword instead of a torch. Question: Is the Statue of Liberty holding a sword?\n Statement: The Sydney Opera House is illustrated as an underwater aquarium, with fish swimming around its structures. Question: Is the Sydney Opera House underwater?\n Statement: The Leaning Tower of Pisa is perfectly vertical in the image, without any tilt. Question: Is the Leaning Tower of Pisa perfectly vertical?\n Now generate a question for the following statement: {statement}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction.format(statement=data['true_statement'])}
    ]
    response = get_response(messages)
    return response

def process_data(data):
    testing_data = data.copy()
    question = generate_question(data)
    testing_data['question'] = question
    return testing_data

if __name__ == "__main__":
    vlind_data = json.load(open(vlind_data_path))

    filtered_data = form_instruction(vlind_data)
    print(len(filtered_data))
    
    testing_data = []
    
    for data in tqdm(filtered_data):
        try:
            testing_data.append(process_data(data))
        except Exception as e:
            continue
    print(len(testing_data))
    
    with open(os.path.join(save_base_dir, "vlind.jsonl"), "w") as f:
        for sample in testing_data:
            f.write(json.dumps(sample) + "\n")