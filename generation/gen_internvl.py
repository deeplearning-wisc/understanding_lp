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
parser.add_argument("--data_path", type=str, default="data")
args = parser.parse_args()

load_dotenv()

from models.internvl import load_internvl
from utils.attention_ops import *
from utils.plot import *
from utils.load_data import get_dataset

def forward(messages, model, processor, target_tokens):
    # Prepare for inference
    inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    image_start_idx = torch.where(inputs.input_ids == 151665)[1] - 1
    image_end_idx = torch.where(inputs.input_ids == 151666)[1]

    # Perform inference to generate the output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
        
    # Calculate the probability of the target token
    target_tokens_probs = []
    for target_token in target_tokens:
        target_token_id = processor.tokenizer.encode(target_token, add_special_tokens=False)[0]
        target_token_probs = F.softmax(outputs.logits, dim=-1)
        target_token_prob = target_token_probs[:, :, target_token_id].mean().item()
        target_tokens_probs.append(target_token_prob)
    
    hidden_states = outputs.hidden_states
    hidden_states = [layer.detach().cpu() for layer in hidden_states]
        
    attention_maps = outputs.attentions
    attention_maps = [layer.detach().cpu() for layer in attention_maps]
    
    if len(image_start_idx) > 0 and len(image_end_idx) > 0:
        image_start_idx = image_start_idx[0]
        image_end_idx = image_end_idx[0]
        last_token_hidden_states = get_hidden_states(hidden_states, last_token_only=True)
        last_token_visual_attention_scores = get_visual_attention_scores(attention_maps, image_start_idx, image_end_idx, last_token_only=True)
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
    model, processor = load_internvl(model_name="OpenGVLab/InternVL3-8B-hf")
    model.eval()
    
    samples, data_info, target_tokens = get_dataset(dataset)
    batch_size = len(samples) // device_num + 1
    samples = samples[batch_size * device_id:batch_size * (device_id + 1)]
    
    # samples = samples[:10]
    
    output_dir = os.path.join(data_path, "internvl")
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
    
    # print(probs_vl.shape)
    # print(probs_l.shape)
    # print(hidden_states_vl.shape)
    # print(hidden_states_l.shape)
    # print(visual_attention_weights.shape)
    # print(visual_attention_weights.mean(dim=1))
    
    # exit()
    
    torch.save(probs_vl, os.path.join(output_dir, f"probs_vl_{dataset}_device_{device_id}.pt"))
    torch.save(probs_l, os.path.join(output_dir, f"probs_l_{dataset}_device_{device_id}.pt"))
    torch.save(hidden_states_vl, os.path.join(output_dir, f"hidden_states_vl_{dataset}_device_{device_id}.pt"))
    torch.save(hidden_states_l, os.path.join(output_dir, f"hidden_states_l_{dataset}_device_{device_id}.pt"))
    torch.save(visual_attention_weights, os.path.join(output_dir, f"visual_attention_weights_{dataset}_device_{device_id}.pt"))
    with open(os.path.join(output_dir, f"data_info_{dataset}_device_{device_id}.json"), "w") as f:
        json.dump(data_info, f)