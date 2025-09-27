import torch
import torch.nn.functional as F
import os
import pickle

data_dir = os.getenv("DATA_DIR")

def get_visual_attention_scores(attention_maps, image_start_idx, image_end_idx, last_token_only=False):
    visual_attention_scores_per_layer = []
    image_start_idx = image_start_idx.to(attention_maps[0].device)
    image_end_idx = image_end_idx.to(attention_maps[0].device)
    for layer in attention_maps:
        token_num = layer.size(-1)
        all_idx = torch.arange(token_num)
        keep_idx_visual = all_idx[(all_idx >= image_start_idx) & (all_idx <= image_end_idx)]
        if last_token_only:
            layer_visual = layer.index_select(-1, keep_idx_visual)[0, :, -1:, :]
        else:
            layer_visual = layer.index_select(-1, keep_idx_visual)[0, :, :, :]
        visual_attention_scores = torch.sum(layer_visual, dim=-1)
        visual_attention_scores_per_layer.append(visual_attention_scores)
    return visual_attention_scores_per_layer

def get_hidden_states(hidden_states, last_token_only=False):
    hidden_states_per_layer = []
    for layer in hidden_states:
        if last_token_only:
            layer_hidden_states = layer[0, -1, :]
        else:
            layer_hidden_states = layer[0, :, :]
        hidden_states_per_layer.append(layer_hidden_states)
    return hidden_states_per_layer

# Calculate the similarity of two attention maps
def compute_attention_maps_similarity(attention_maps_1, attention_maps_2, token_idx, similarity_func):
    similarities = torch.zeros(len(attention_maps_1), len(attention_maps_1[0][0]))
    for layer_idx in range(len(attention_maps_1)):
        heads_1 = attention_maps_1[layer_idx][0]
        heads_2 = attention_maps_2[layer_idx][0]
        for head_idx in range(heads_1.size(0)):
            distribution_1 = heads_1[head_idx, token_idx, :token_idx+1]
            distribution_2 = heads_2[head_idx, token_idx, :token_idx+1]
            similarity = similarity_func(distribution_1, distribution_2)
            similarities[layer_idx, head_idx] = similarity
    return similarities

def normalize_distribution(distribution):
    eps = 1e-12
    distribution = torch.clamp(distribution, min=eps)
    distribution = distribution / (distribution.sum(dim=-1, keepdim=True) + eps)
    return distribution

def total_variation_distance(p, q):
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return torch.sum(torch.abs(p - q)) / 2

# def jensen_shannon_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
#     """
#     Jensen-Shannon distance (0~1)
#     JS(P,Q) = sqrt( 0.5*KL(P||M) + 0.5*KL(Q||M) ) / sqrt(log(2))
#     Avoid nan: ensure the denominator is non-zero and js_div is non-negative
#     """
#     eps = 1e-12
#     p = normalize_distribution(p)
#     q = normalize_distribution(q)
#     m = 0.5 * (p + q)

#     # Add eps to the denominator when calculating KL divergence to avoid log(0)
#     kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=-1)
#     kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=-1)

#     js_div = 0.5 * (kl_pm + kl_qm)
#     # Clamp js_div to avoid negative values leading to nan in sqrt
#     js_div = torch.clamp(js_div, min=0.0)
#     js_dist = torch.sqrt(js_div / torch.log(torch.tensor(2.0)))
#     # If js_div is 0, the result is 0; if the denominator is 0, the result is also 0
#     js_dist = torch.nan_to_num(js_dist, nan=0.0, posinf=1.0, neginf=0.0)
#     return js_dist

def kl_divergence_torch(p: torch.Tensor, q: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """
    KL(P||Q) with safe clamp.
    p, q: (..., d)
    """
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * (torch.log2(p) - torch.log2(q)), dim=-1)

def vanilla_js_distance(p: torch.Tensor, q: torch.Tensor):
    """JS distance between batches p,q shape (batch,d)"""
    m = 0.5 * (p + q)
    js_div = 0.5 * kl_divergence_torch(p, m) + 0.5 * kl_divergence_torch(q, m)
    return torch.sqrt(torch.clamp(js_div, min=0.0))

def js_baseline_mu_batch(d: int, n: int = 2000, 
                         batch_size: int = 128, 
                         device="cuda:0") -> float:
    """
    Sample n random d-dim distributions ~ Dirichlet(1),
    compute their mean pairwise JS distance (mu_d) in memory-efficient batches.
    
    Args:
        d: dimension
        n: number of random distributions
        batch_size: how many rows per block to compute pairwise
        device: cpu or cuda
    
    Returns:
        mu_d: float
    """
    # 1. Sample n*d and normalize
    X = torch.rand(n, d, device=device)
    X = X / X.sum(dim=-1, keepdim=True)
    
    total_js = 0.0
    total_pairs = 0
    
    # 2. Traverse upper triangle in blocks
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        A = X[i_start:i_end]  # (bs1,d)
        
        # Only compute with blocks that come after, avoid duplication/diagonal
        for j_start in range(i_start+1, n, batch_size):
            j_end = min(j_start + batch_size, n)
            B = X[j_start:j_end]  # (bs2,d)
            
            # Broadcast to pairwise (bs1*bs2,d)
            AA = A.unsqueeze(1).expand(-1, B.shape[0], -1).reshape(-1, d)
            BB = B.unsqueeze(0).expand(A.shape[0], -1, -1).reshape(-1, d)
            
            dist = vanilla_js_distance(AA, BB)
            total_js += dist.sum().item()
            total_pairs += dist.numel()
    
    # Note: Upper triangle traversal doesn't compute i<j pairs within the same block, need to compute separately
    # Upper triangle within the same block
    for k_start in range(0, n, batch_size):
        k_end = min(k_start + batch_size, n)
        blk = X[k_start:k_end]  # (bs,d)
        bs = blk.shape[0]
        if bs > 1:
            P = blk.unsqueeze(1).expand(-1, bs, -1)
            Q = blk.unsqueeze(0).expand(bs, -1, -1)
            M = 0.5 * (P + Q)
            js_div = 0.5 * kl_divergence_torch(P, M) + 0.5 * kl_divergence_torch(Q, M)
            js_dist = torch.sqrt(js_div)
            
            # Upper triangle mask
            mask = torch.triu(torch.ones(bs, bs, dtype=torch.bool, device=device), diagonal=1)
            total_js += js_dist[mask].sum().item()
            total_pairs += mask.sum().item()
    
    del X
    torch.cuda.empty_cache()
    
    return total_js / total_pairs

def jensen_shannon_distance(p: torch.Tensor, 
                      q: torch.Tensor, 
                      normalize: str = 'baseline',
                      cache_dir: str = os.path.join(data_dir, "cache", "js_dist_cache.pkl")) -> torch.Tensor:
    """
    Jensen-Shannon Distance with optional dimension normalization.
    
    Args:
        p, q: Tensor with shape (..., d), need not normalized
        normalize:
            - 'none' (default) standard JS distance
            - 'dim'  divide by sqrt(d)
            - 'log'  divide by log2(d+1)
    Returns:
        Tensor shape (...,)
    """
    # normalize distributions along last dim
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    m = 0.5 * (p + q)
    
    js_div = 0.5 * kl_divergence_torch(p, m) + 0.5 * kl_divergence_torch(q, m)
    js_dist = torch.sqrt(torch.clamp(js_div, min=0.0))  # standard Jensen-Shannon distance ∈ [0,1]
    
    d = p.shape[-1]
    if normalize == 'dim':
        js_dist = js_dist / torch.sqrt(torch.tensor(float(d), device=js_dist.device))
    elif normalize == 'log':
        js_dist = js_dist / torch.log2(torch.tensor(float(d+1), device=js_dist.device))
    elif normalize == 'baseline':
        if os.path.exists(cache_dir):
            with open(cache_dir, "rb") as f:
                cache = pickle.load(f)
            if d in cache:
                mu_d = cache[d]
            else:
                mu_d = js_baseline_mu_batch(d, n=10000)
                cache[d] = mu_d
                with open(cache_dir, "wb") as f:
                    pickle.dump(cache, f)
        else:
            mu_d = js_baseline_mu_batch(d, n=10000)
            with open(cache_dir, "wb") as f:
                pickle.dump({d: mu_d}, f)
        js_dist = js_dist - mu_d
    
    return js_dist

def jensen_shannon_distance_non_normalized(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon distance (0~1)
    JS(P,Q) = sqrt( 0.5*KL(P||M) + 0.5*KL(Q||M) ) / sqrt(log(2))
    Avoid nan: ensure the denominator is non-zero and js_div is non-negative
    """
    eps = 1e-12
    # p = torch.clamp(p, min=eps)
    # q = torch.clamp(q, min=eps)
    m = 0.5 * (p + q)

    # Add eps to the denominator when calculating KL divergence to avoid log(0)
    kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=-1)

    js_div = 0.5 * (kl_pm + kl_qm)
    # Clamp js_div to avoid negative values leading to nan in sqrt
    js_div = torch.clamp(js_div, min=0.0)
    js_dist = torch.sqrt(js_div / torch.log(torch.tensor(2.0)))
    # If js_div is 0, the result is 0; if the denominator is 0, the result is also 0
    js_dist = torch.nan_to_num(js_dist, nan=0.0, posinf=1.0, neginf=0.0)
    return js_dist

def hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Hellinger distance (0~1)
    H(P,Q) = sqrt(0.5 * sum( (sqrt(p_i)-sqrt(q_i))^2 ))
    """
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q))**2, dim=-1))

def cosine_similarity(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity (-1~1)
    Suitable for observing distribution shape similarity rather than absolute differences
    """
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return F.cosine_similarity(p, q, dim=-1)

def l2_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    L2 distance (Euclidean distance)
    Note: May scale with dimension, not necessarily dimension-stable
    """
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return torch.norm(p - q, p=2, dim=-1)

# Calculate visual attention weight (at last token)
def calculate_visual_attention_weights(visual_attention_maps):
    visual_attention_weight = torch.zeros(len(visual_attention_maps), len(visual_attention_maps[0][0]))
    for layer_idx in range(len(visual_attention_maps)):
        heads = visual_attention_maps[layer_idx][0]
        for head_idx in range(heads.size(0)):
            distribution = heads[head_idx, -1, :]
            visual_attention_weight[layer_idx, head_idx] = distribution.sum()
    return visual_attention_weight