import os
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device_num", type=int, default=1)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--dataset", type=str, default="mme")
parser.add_argument("--test", action="store_true")
parser.add_argument("--data_path", type=str, default="data")
args = parser.parse_args()

load_dotenv()

from models.eagle import load_eagle
from utils.attention_ops import *
from utils.plot import *
from utils.load_data import get_dataset

def forward(messages, model, processor, target_tokens):
    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device, dtype=torch.bfloat16)
    
    # torch.set_printoptions(threshold=float('inf'))
    # print(inputs.input_ids)
        
    image_start_idx = torch.where(inputs.input_ids == 151665)[1]
    image_end_idx = torch.where(inputs.input_ids == 151666)[1]
    
    
    # print(f"image_start_idx: {image_start_idx}, image_end_idx: {image_end_idx}, image_tokens_num: {image_end_idx - image_start_idx + 1}")

    # Perform inference to generate the output
    with torch.no_grad():
        # outputs = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
        outputs = model.generate(**inputs, max_new_tokens=1024, output_attentions=True, output_hidden_states=True, output_logits=True, do_sample=False, return_dict_in_generate=True)
        
    # Calculate the probability of the target token
    target_tokens_probs = []
    for target_token in target_tokens:
        target_token_id = processor.tokenizer.encode(target_token, add_special_tokens=False)[0]
        target_token_probs = F.softmax(outputs.logits[0], dim=-1)
        target_token_prob = target_token_probs[:, target_token_id].mean().item()
        target_tokens_probs.append(target_token_prob)
    
    hidden_states = outputs.hidden_states[0]
    hidden_states = [layer.detach().cpu() for layer in hidden_states]
        
    attention_maps = outputs.attentions[0]
    if attention_maps is not None and attention_maps[0] is not None:
        attention_maps = [layer.detach().cpu() for layer in attention_maps]
    
    if len(image_start_idx) > 0 and len(image_end_idx) > 0:
        image_start_idx = image_start_idx[0]
        image_end_idx = image_end_idx[0]
        last_token_hidden_states = get_hidden_states(hidden_states, last_token_only=True)
        if attention_maps is not None and attention_maps[0] is not None:
            last_token_visual_attention_scores = get_visual_attention_scores(attention_maps, image_start_idx, image_end_idx, last_token_only=True)
        else:
            last_token_visual_attention_scores = [torch.ones(5) * -1 for _ in range(len(hidden_states))]
    else:
        image_start_idx = None
        image_end_idx = None
        last_token_hidden_states = get_hidden_states(hidden_states, last_token_only=True)
        last_token_visual_attention_scores = None
    
    return last_token_hidden_states, last_token_visual_attention_scores, target_tokens_probs

def get_generation_pattern(image, instruction, model, processor, target_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": instruction},
            ],
        },
    ]
    messages_without_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
            ],
        },
    ]

    
    last_token_hidden_states_vl, last_token_visual_attention_scores_vl, target_tokens_probs_vl = forward(messages, model, processor, target_tokens)    
    last_token_hidden_states_l, last_token_visual_attention_scores_l, target_tokens_probs_l = forward(messages_without_image, model, processor, target_tokens)
    
    assert last_token_hidden_states_vl[0].shape == last_token_hidden_states_l[0].shape, f"last_token_hidden_states_vl[0].shape: {last_token_hidden_states_vl[0].shape}, last_token_hidden_states_l[0].shape: {last_token_hidden_states_l[0].shape}"
    assert len(last_token_hidden_states_vl) == len(last_token_hidden_states_l), f"len(last_token_hidden_states_vl): {len(last_token_hidden_states_vl)}, len(last_token_hidden_states_l): {len(last_token_hidden_states_l)}"
    
    return last_token_hidden_states_vl, last_token_hidden_states_l, target_tokens_probs_vl, target_tokens_probs_l, last_token_visual_attention_scores_vl

if __name__ == "__main__":
    dataset = args.dataset
    device_num = args.device_num
    device_id = args.device_id
    data_path = args.data_path
    
    model, processor = load_eagle()
    model.eval()
    
    samples, data_info, target_tokens = get_dataset(dataset)
    batch_size = len(samples) // device_num + 1
    samples = samples[batch_size * device_id:batch_size * (device_id + 1)]
    if args.test:
        samples = samples[:10]
        
    output_dir = os.path.join(data_path, "eagle")
    os.makedirs(output_dir, exist_ok=True)
        
    probs_vl = []
    probs_l = []
    hidden_states_vl = []
    hidden_states_l = []
    visual_attention_weights = []
    for sample in tqdm(samples):
        hidden_states_vl_, hidden_states_l_, probs_vl_, probs_l_, visual_attention_weights_ = get_generation_pattern(sample["image"], sample["instruction"], model, processor, target_tokens=target_tokens)
        probs_vl.append(torch.tensor(probs_vl_))
        probs_l.append(torch.tensor(probs_l_))
        hidden_states_vl.append(torch.stack(hidden_states_vl_))
        hidden_states_l.append(torch.stack(hidden_states_l_))
        visual_attention_weights_ = torch.stack(visual_attention_weights_)
        avg_visual_attention_weight = torch.mean(visual_attention_weights_, dim=1)
        visual_attention_weights.append(avg_visual_attention_weight)
    
    probs_vl = torch.stack(probs_vl)
    probs_l = torch.stack(probs_l)
    hidden_states_vl = torch.stack(hidden_states_vl)
    hidden_states_l = torch.stack(hidden_states_l)
    visual_attention_weights = torch.stack(visual_attention_weights)
    
    if args.test:
        print(probs_vl.shape)
        print(probs_l.shape)
        print(hidden_states_vl.shape)
        print(hidden_states_l.shape)
        print(visual_attention_weights.shape)
        print(visual_attention_weights.mean(dim=1))
        print(hidden_states_vl[0][0])
        print(hidden_states_l[0][0])
        
        exit()
        
    torch.save(probs_vl, os.path.join(output_dir, f"probs_vl_{dataset}_device_{device_id}.pt"))
    torch.save(probs_l, os.path.join(output_dir, f"probs_l_{dataset}_device_{device_id}.pt"))
    torch.save(hidden_states_vl, os.path.join(output_dir, f"hidden_states_vl_{dataset}_device_{device_id}.pt"))
    torch.save(hidden_states_l, os.path.join(output_dir, f"hidden_states_l_{dataset}_device_{device_id}.pt"))
    torch.save(visual_attention_weights, os.path.join(output_dir, f"visual_attention_weights_{dataset}_device_{device_id}.pt"))
    with open(os.path.join(output_dir, f"data_info_{dataset}_device_{device_id}.json"), "w") as f:
        json.dump(data_info, f)