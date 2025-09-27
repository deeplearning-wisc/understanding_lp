import torch
from tqdm import tqdm
import argparse
import pandas as pd
import math
import os
import json
from verify import verify
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mme,commonsense_qa,mmbench,vlind,mmstar,mmmu,vilp", help="Dataset name or comma-separated list of datasets")
    parser.add_argument("--model", type=str, default="qwenvl,llava,gemma,llama,fuyu,llavaov,llavanext,smolvlm,eagle,internvl", help="Model name or comma-separated list of models")
    parser.add_argument("--device_num", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="data")
    args = parser.parse_args()
    
    # Parse dataset argument - can be single dataset or comma-separated list
    if "," in args.dataset:
        datasets = [ds.strip() for ds in args.dataset.split(",")]
    else:
        datasets = [args.dataset]
    
    # Parse model argument - can be single model or comma-separated list
    if "," in args.model:
        models = [m.strip() for m in args.model.split(",")]
    else:
        models = [args.model]
    
    device_num = args.device_num
    data_path = args.data_path
    for model in models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model}")
        print(f"{'='*60}")
        
        for dataset in datasets:
            print(f"\n{'-'*50}")
            print(f"Processing dataset: {dataset}")
            print(f"{'-'*50}")
            
            all_probs_vl = []
            all_probs_l = []
            all_hidden_states_vl = []
            all_hidden_states_l = []
            all_visual_attention_weights = []
            all_data_info = []
            
            print(f"Loading {model} {dataset} data...")
            
            for device_id in tqdm(range(device_num)):
                try:
                    probs_vl = torch.load(os.path.join(data_path, f"{model}/probs_vl_{dataset}_device_{device_id}.pt"))
                    probs_l = torch.load(os.path.join(data_path, f"{model}/probs_l_{dataset}_device_{device_id}.pt"))
                    hidden_states_vl = torch.load(os.path.join(data_path, f"{model}/hidden_states_vl_{dataset}_device_{device_id}.pt"))
                    hidden_states_l = torch.load(os.path.join(data_path, f"{model}/hidden_states_l_{dataset}_device_{device_id}.pt"))
                    visual_attention_weights = torch.load(os.path.join(data_path, f"{model}/visual_attention_weights_{dataset}_device_{device_id}.pt"))
                    data_info = json.load(open(os.path.join(data_path, f"{model}/data_info_{dataset}_device_{device_id}.json")))
                    all_probs_vl.append(probs_vl)
                    all_probs_l.append(probs_l)
                    all_hidden_states_vl.append(hidden_states_vl)
                    all_hidden_states_l.append(hidden_states_l)
                    all_visual_attention_weights.append(visual_attention_weights)
                    all_data_info = data_info
                except Exception as e:
                    print(f"Error loading file {device_id}: {e}")
                    continue
                
            
            print(f"Concatenating {model} {dataset} data...")
            
            all_probs_vl = torch.cat(all_probs_vl, dim=0)
            all_probs_l = torch.cat(all_probs_l, dim=0)
            all_hidden_states_vl = torch.cat(all_hidden_states_vl, dim=0)
            all_hidden_states_l = torch.cat(all_hidden_states_l, dim=0)
            all_visual_attention_weights = torch.cat(all_visual_attention_weights, dim=0)
            
            assert len(all_probs_vl) == len(all_probs_l) == len(all_hidden_states_vl) == len(all_hidden_states_l) == len(all_visual_attention_weights) == len(all_data_info)
            
            print(f"All probs VL shape: {all_probs_vl.shape}")
            print(f"All probs L shape: {all_probs_l.shape}")
            print(f"All hidden states VL shape: {all_hidden_states_vl.shape}")
            print(f"All hidden states L shape: {all_hidden_states_l.shape}")
            print(f"All visual attention weights shape: {all_visual_attention_weights.shape}")
            print(f"All data info shape: {len(all_data_info)}")
            
            print(f"Verifying {model} {dataset} data...")
            
            all_correctness_vl = verify(dataset, all_probs_vl, all_data_info)
            all_correctness_l = verify(dataset, all_probs_l, all_data_info)
            
            print(f"Accuracy for VL: {torch.mean(all_correctness_vl.float())}")
            print(f"Accuracy for L: {torch.mean(all_correctness_l.float())}")
            
            print(f"Saving {model} {dataset} data...")
            
            torch.save(all_probs_vl, os.path.join(data_path, f"{model}/probs_vl_{dataset}.pt"))
            torch.save(all_probs_l, os.path.join(data_path, f"{model}/probs_l_{dataset}.pt"))
            torch.save(all_hidden_states_vl, os.path.join(data_path, f"{model}/hidden_states_vl_{dataset}.pt"))
            torch.save(all_hidden_states_l, os.path.join(data_path, f"{model}/hidden_states_l_{dataset}.pt"))
            torch.save(all_visual_attention_weights, os.path.join(data_path, f"{model}/visual_attention_weights_{dataset}.pt"))
            torch.save(all_correctness_vl, os.path.join(data_path, f"{model}/correctness_vl_{dataset}.pt"))
            torch.save(all_correctness_l, os.path.join(data_path, f"{model}/correctness_l_{dataset}.pt"))
            json.dump(all_data_info, open(os.path.join(data_path, f"{model}/data_info_{dataset}.json"), "w"))
            
            print(f"Done processing {model} {dataset}!")
    
    print(f"\n{'='*60}")
    print(f"All models and datasets processed successfully!")
    print(f"{'='*60}")