import matplotlib.pyplot as plt
from utils.visualize_utils import plot_divergences_list, plot_label_distributions
import torch
import torch.nn.functional as F
import json
from scipy.stats import spearmanr
import os
import argparse
from models.qwenvl import load_qwenvl_lmhead

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mme,mmbench,vlind,mmstar,mmmu,vilp", help="Dataset name or comma-separated list of datasets")
parser.add_argument("--model", type=str, default="qwenvl,llava,gemma,llama,fuyu,llavaov,llavanext,smolvlm,eagle,internvl,gemma-3-12b-it,gemma-3-27b-it", help="Model name or comma-separated list of models")
parser.add_argument("--data_path", type=str, default="data", help="Path to the data")

args = parser.parse_args()


# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'

def js_distance(p: torch.Tensor, q: torch.Tensor, eps=1e-12):
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)

    m = 0.5 * (p + q) + eps

    kl_pm = F.kl_div((p + eps).log(), m, reduction='none').sum(dim=-1)
    kl_qm = F.kl_div((q + eps).log(), m, reduction='none').sum(dim=-1)

    js_div = 0.5 * (kl_pm + kl_qm)
    js_div = torch.where(js_div < 0, torch.tensor(0.0, device=js_div.device), js_div)
    js_dist = torch.sqrt(js_div)

    return js_dist

def logit_lens(model, hidden_states_list):
    if model == "qwenvl":
        lm_head, processor = load_qwenvl_lmhead()
        target_tokens = ["A", "B", "C", "D"]
        target_token_ids = processor.tokenizer(target_tokens, add_special_tokens=False, return_tensors="pt").input_ids[:,0]
        logit_lens_vl = []
        with torch.no_grad():
            for hidden_states in hidden_states_list:
                hidden_states = hidden_states.to(next(lm_head.parameters()).device)
                logit_lens_vl.append(lm_head(hidden_states).detach().cpu())
        logit_lens_vl = torch.stack(logit_lens_vl)
        del lm_head
        logit_lens_vl = logit_lens_vl[:, :, :, target_token_ids]
        
        return logit_lens_vl.to(torch.float32)
    else:
        return None
    
def get_model_name(model):
    if model == "qwenvl":
        return "Qwen2.5-VL-7B"
    elif model == "llava":
        return "LLaVA-v1.5-7B"
    elif model == "gemma":
        return "Gemma-3-4B"
    elif model == "gemma-3-12b-it":
        return "Gemma-3-12B"
    elif model == "gemma-3-27b-it":
        return "Gemma-3-27B"
    elif model == "internvl":
        return "InternVL3-8B"
    elif model == "llama":
        return "Llama-3.2-11B-Vision"
    elif model == "llavaov":
        return "LLaVA-OV-Qwen2-7B"
    elif model == "llavanext":
        return "LLaVA-NeXT-Vicuna-7B"
    elif model == "smolvlm":
        return "SmolVLM"
    elif model == "eagle":
        return "Eagle2.5-8B"
    else:
        raise ValueError(f"Invalid model: {model}")
    
def get_dataset_name(dataset):
    if dataset == "mme":
        return "MME"
    elif dataset == "commonsense_qa":
        return "CommonsenseQA"
    elif dataset == "mmbench":
        return "MMBench"
    elif dataset == "vlind":
        return "VLind-Bench"
    elif dataset == "mmmu":
        return "MMMU"
    elif dataset == "mmstar":
        return "MMStar"
    elif dataset == "vilp":
        return "ViLP"

def get_data_from_pt(dataset, model):
    print(f"Loading {model} {dataset} data...")
    probs_vl = torch.load(os.path.join(args.data_path, f"{model}/probs_vl_{dataset}.pt"))
    probs_l = torch.load(os.path.join(args.data_path, f"{model}/probs_l_{dataset}.pt"))
    hidden_states_vl = torch.load(os.path.join(args.data_path, f"{model}/hidden_states_vl_{dataset}.pt"))
    hidden_states_l = torch.load(os.path.join(args.data_path, f"{model}/hidden_states_l_{dataset}.pt"))
    visual_attention_weights = torch.load(os.path.join(args.data_path, f"{model}/visual_attention_weights_{dataset}.pt"))
    correctness_vl = torch.load(os.path.join(args.data_path, f"{model}/correctness_vl_{dataset}.pt"))
    correctness_l = torch.load(os.path.join(args.data_path, f"{model}/correctness_l_{dataset}.pt"))
    data_info = json.load(open(os.path.join(args.data_path, f"{model}/data_info_{dataset}.json"), "r"))
    
    confidence_vl = torch.max(probs_vl, dim=1)[0] / torch.sum(probs_vl, dim=1)
    confidence_l = torch.max(probs_l, dim=1)[0] / torch.sum(probs_l, dim=1)

    print(f"Loaded {model} {dataset} data with length {len(probs_vl)}...")
    
    print("=========Basic Info=========")
    print(f"Blind confidence: {torch.mean(confidence_l.to(torch.float32)):.4f}")
    print(f"Blind accuracy: {torch.mean(correctness_l.to(torch.float32)):.4f}")
    print(f"Accuracy: {torch.mean(correctness_vl.to(torch.float32)):.4f}")
    print(f"Accuracy when blind guess is correct: {torch.mean(correctness_vl[torch.where(correctness_l == 1)[0]].to(torch.float32)):.4f}")
    print("=========Basic Info=========")
    
    return probs_vl, probs_l, hidden_states_vl, hidden_states_l, visual_attention_weights, correctness_vl, correctness_l, data_info, confidence_vl, confidence_l

def calculate_trace_similarity(traces_vl, traces_l, velocity=True, selected_layer=None, metric="cosine"):
    """Calculate similarity of vectors at the same positions in trajectories"""
    # Ensure both trajectories have the same length
    assert len(traces_vl) == len(traces_l), "Trajectory lengths must be the same"
    if selected_layer is not None:
        traces_vl = traces_vl[selected_layer, :]
        traces_l = traces_l[selected_layer, :]
    
    if velocity:
        traces_vl = torch.diff(traces_vl, dim=0)
        traces_l = torch.diff(traces_l, dim=0)
    
    if metric == "cosine":
        similarities = torch.nn.functional.cosine_similarity(traces_vl, traces_l, dim=-1)
    elif metric == "euclidean":
        similarities = -torch.norm(traces_vl - traces_l, dim=-1)
    elif metric == "js_distance":
        traces_vl = traces_vl.to("cuda")
        traces_l = traces_l.to("cuda")
        with torch.no_grad():
            similarities = -js_distance(traces_vl, traces_l).to("cpu")
    elif metric == "kl_divergence":
        traces_vl = traces_vl.to("cuda")
        traces_l = traces_l.to("cuda")
        with torch.no_grad():
            traces_vl = torch.log(traces_vl + 1e-12)
            traces_l = torch.log(traces_l + 1e-12)
            similarities = torch.nn.functional.kl_div(traces_vl, traces_l, reduction="none", log_target=True).to("cpu")
            similarities = -torch.sum(similarities, dim=-1).to("cpu")
    else:
        raise ValueError(f"Invalid metric: {metric}")
    
    return similarities

def get_divergences(model, dataset, metric="cosine", dim_norm=False):    
    probs_vl, probs_l, hidden_states_vl, hidden_states_l, visual_attention_weights, correctness_vl, correctness_l, data_info, confidence_vl, confidence_l = get_data_from_pt(dataset, model)
    
    sample_num = len(probs_vl)

    similarity_per_layer_list = []
    
    if model == "qwenvl" and (metric == "js_distance" or metric == "kl_divergence"): 
        logits = logit_lens(model, torch.stack([hidden_states_vl, hidden_states_l]))
        hidden_states_vl = torch.softmax(logits[0], dim=-1)
        hidden_states_l = torch.softmax(logits[1], dim=-1)
    
    for idx in range(sample_num):
        traces_vl = hidden_states_vl[idx]
        traces_l = hidden_states_l[idx]
        
        similarity_per_layer = calculate_trace_similarity(traces_vl, traces_l, velocity=False, selected_layer=None, metric=metric)
        
        similarity_per_layer_list.append(similarity_per_layer)

    similarity_per_layer_list = torch.stack(similarity_per_layer_list)
    
    divergences_per_layer = 1 - similarity_per_layer_list
    visual_attention_weights = visual_attention_weights.mean(dim=1)
    
    if dim_norm:
        divergences_per_layer = divergences_per_layer * torch.sqrt(torch.tensor(hidden_states_vl.shape[-1], device=divergences_per_layer.device))
    
    return divergences_per_layer.to(torch.float32), visual_attention_weights.to(torch.float32), confidence_vl.to(torch.float32), confidence_l.to(torch.float32), correctness_vl.to(torch.float32), correctness_l.to(torch.float32), probs_vl.to(torch.float32), probs_l.to(torch.float32)

def get_baseline_divergences(model):
    baseline_divergences, _, _, _, _, _, _, _ = get_divergences(model, "commonsense_qa")
    return baseline_divergences.to(torch.float32)

def calculate_spearman_r(model, dataset, divergences_per_layer, correctness_vl, correctness_l, visual_attention_weights, confidence_vl, confidence_l, probs_vl, probs_l, idx=None):
    if idx is not None:
        divergences_per_layer = divergences_per_layer[idx]
        correctness_vl = correctness_vl[idx]
        correctness_l = correctness_l[idx]
        visual_attention_weights = visual_attention_weights[idx]
        confidence_vl = confidence_vl[idx]
        confidence_l = confidence_l[idx]
        probs_vl = probs_vl[idx]
        probs_l = probs_l[idx]
    print(f"Calculating Spearman R for {model} - {dataset}")
    
    vip = get_vip(model)
    
    
    divergences_all_layer = torch.mean(divergences_per_layer, axis=1)
    divergences_after_vip = torch.mean(divergences_per_layer[:,vip:], axis=1)
    divergences_before_vip = torch.mean(divergences_per_layer[:,:vip], axis=1)
    
    divergences_output = divergences_per_layer[:,-1]
    
    
    # Calculate Spearman R and P values
    spearman_divergences_all_layer, p_divergences_all_layer = spearmanr(correctness_vl, divergences_all_layer)
    spearman_divergences_before_vip, p_divergences_before_vip = spearmanr(correctness_vl, divergences_before_vip)
    spearman_divergences_after_vip, p_divergences_after_vip = spearmanr(correctness_vl, divergences_after_vip)
    spearman_divergences_output, p_divergences_output = spearmanr(correctness_vl, divergences_output)
    spearman_vis, p_vis = spearmanr(correctness_vl, visual_attention_weights)
    
    print(f"Spearman R (Div_All_Layer): {spearman_divergences_all_layer:.4f}, P-value: {p_divergences_all_layer:.8f}")
    print(f"Spearman R (Div_Pre_VIP): {spearman_divergences_before_vip:.4f}, P-value: {p_divergences_before_vip:.8f}")
    print(f"Spearman R (Div_Post_VIP): {spearman_divergences_after_vip:.4f}, P-value: {p_divergences_after_vip:.8f}")
    print(f"Spearman R (Div_Output): {spearman_divergences_output:.4f}, P-value: {p_divergences_output:.8f}")
    print(f"Spearman R (VA): {spearman_vis:.4f}, P-value: {p_vis:.8f}")
    
    print(f"✓ Successfully processed {model} - {dataset}")

def get_vip(model):
    if model == "llava":
        return 9
    elif model == "gemma":
        return 20
    elif model == "internvl":
        return 16
    elif model == "qwenvl":
        return 18
    elif model == "llama":
        return 12
    elif model == "gemma-3-12b-it":
        return 26
    elif model == "llavaov":
        return 15
    elif model == "llavanext":
        return 12
    elif model == "gemma-3-27b-it":
        return 35
    elif model == "eagle":
        return 15
    elif model == "smolvlm":
        return 15
    else:
        print(f"Unknown model: {model}")
        return 0

def process_model_dataset_combination(model, dataset):
    """Process single model and dataset combination"""
    print(f"\n{'='*60}")
    print(f"Processing: {model.upper()} - {dataset.upper()}")
    print(f"{'='*60}")
    
    vip = get_vip(model)
    
    output_dir = os.path.join(args.data_path, "analysis/divergence_curve/paper", model)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
    
        divergences_per_layer, visual_attention_weights, confidence_vl, confidence_l, correctness_vl, correctness_l, probs_vl, probs_l = get_divergences(model, dataset)
        
        same = divergences_per_layer[torch.argmax(probs_vl, dim=1) == torch.argmax(probs_l, dim=1)]
        diff = divergences_per_layer[torch.argmax(probs_vl, dim=1) != torch.argmax(probs_l, dim=1)]

        
        scale = range(same.shape[1])
        if model.startswith("gemma"):
            scale = range(same.shape[1] - 1)
        
        # Plot charts
        plot_divergences_list([same[:,scale], diff[:,scale]], 
                            [r'$D_\text{T}$' + ' ' + f'(Avg TVI={torch.mean(same[:,vip:]).item():.3f})', 
                            r'$D_\text{VT}$' + ' ' + f'(Avg TVI={torch.mean(diff[:,vip:]).item():.3f})'], 
                            f'{get_model_name(model)} on {get_dataset_name(dataset)}', 
                            save_path=os.path.join(output_dir, f'{dataset}.pdf'))

        calculate_spearman_r(model, dataset, divergences_per_layer, correctness_vl, correctness_l, visual_attention_weights, confidence_vl, confidence_l, probs_vl, probs_l)
        
    except Exception as e:
        print(f"✗ Error processing {model} - {dataset}: {e}")
        return False
    
    return True

if __name__ == "__main__":    
    # Parse model argument - can be single model or comma-separated list
    if "," in args.model:
        models = [m.strip() for m in args.model.split(",")]
    else:
        models = [args.model]
    
    # Parse dataset argument - can be single dataset or comma-separated list
    if "," in args.dataset:
        datasets = [ds.strip() for ds in args.dataset.split(",")]
    else:
        datasets = [args.dataset]
    
    print(f"Models to process: {models}")
    print(f"Datasets to process: {datasets}")
    print(f"Total combinations: {len(models) * len(datasets)}")
    
    success_count = 0
    total_count = len(models) * len(datasets)
    
    # Process each combination iteratively
    for model in models:
        for dataset in datasets:
            if process_model_dataset_combination(model, dataset):
                success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Success: {success_count}/{total_count}")
    print(f"{'='*60}")