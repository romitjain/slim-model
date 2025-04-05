import torch
import numpy as np
from typing import Dict, List, Tuple

def compute_activation_variance(activation_tensor: torch.Tensor, attention_mask: torch.Tensor) -> float:
    """
    Compute average variance of activations over tokens.
    Expected tensor shape: (batch, seq_len, hidden_dim)
    """
    assert activation_tensor.ndim == 3, "Activation tensor must have 3 dimensions"
    assert attention_mask.ndim == 2, "Attention mask must have 2 dimensions"

    activation_tensor = activation_tensor * attention_mask.unsqueeze(2)

    # Variance along token dimension for each hidden unit
    # (if batch >1, variance computed per sample then averaged)
    var = activation_tensor.var(dim=1, unbiased=False)  # shape: (batch, hidden_dim)
    return var.mean().item() # shape: (hidden_dim,)

def compute_attention_entropy(attention_tensor: torch.Tensor, eps: float = 1e-9) -> List[float]:
    """
    Compute average entropy per head.
    Expected tensor shape: (batch, num_heads, seq_len, seq_len)
    """

    # Entropy: -sum(p * log(p)) over the key dimension (last dim)
    entropy = - (attention_tensor * torch.log(attention_tensor + eps)).sum(dim=-1)  # (batch, num_heads, seq_len)
    # Average over batch and query tokens
    avg_entropy = entropy.mean(dim=(0, 2))  # shape: (num_heads,)
    return avg_entropy.tolist()

def compute_attention_kl(attention_tensor: torch.Tensor, eps: float = 1e-9) -> List[float]:
    """
    Compute KL divergence KL(Attention || Uniform) for each head's attention distribution.
    Measures how much the attention distribution P deviates from a uniform distribution Q.
    KL(P || Q) = sum(P * log(P/Q)) = log(N) - Entropy(P)
    """
    B, H, L_query, L_key = attention_tensor.shape

    if L_key <= 0:
        raise ValueError(f"Length of attention distribution (L_key) must be positive, but got L_key={L_key}")
    # Decode phase
    elif L_key == 1:
        log_L = 0.0
    else:
        log_L = torch.log(torch.tensor(L_key, device=attention_tensor.device, dtype=torch.float64)).to(attention_tensor.dtype)

    # Entropy calculation H(P) = -sum(P * log(P + eps))
    # Sum over the distribution dimension L_key (the last dimension, -1)
    log_term = torch.log(attention_tensor + eps)
    entropy = - (attention_tensor * log_term).sum(dim=-1)  # Shape: (B, H, L_query)

    # Resulting kl_div tensor has shape corresponding to entropy: (B, H, L_query)
    kl_div = log_L - entropy

    avg_kl_div = kl_div.mean(dim=(0, 2)) # Shape: (num_heads,)
    return avg_kl_div.tolist()

def compute_composite_score(
        attention_entropies: Dict[str, Dict[int, float]],
        attention_kl: Dict[str, Dict[int, float]],
        entropy_weight: float = 0.5,
        kl_weight: float = 0,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
    """
    Compute composite score for each head.
    High entropy and low kl divergence is good.
    """

    def normalize(x):
        return (x - min(x)) / (max(x) - min(x))
    
    # normalize entropy and kl across layers
    normalized_entropies = {}
    normalized_kl = {}

    for layer in attention_entropies.keys():
        if layer not in normalized_entropies:
            normalized_entropies[layer] = []
        normalized_entropies[layer].append(
            np.mean([h for h in attention_entropies[layer].values()])
        )

        if layer not in normalized_kl:
            normalized_kl[layer] = []
        normalized_kl[layer].append(
            np.mean([h for h in attention_kl[layer].values()])
        )

    candidates = {}
    for layer in attention_entropies.keys():
        candidates[layer] = entropy_weight * normalized_entropies[layer][0]# + (1-kl_weight) * normalized_kl[layer][0]

    # Currently outputs layer number for self attn
    return [(k.split("_")[1], v) for k, v in sorted(candidates.items(), key=lambda x: x[1], reverse=True)][:top_k]
