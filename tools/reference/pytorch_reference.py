#!/usr/bin/env python3
"""
PyTorch Reference Model for validating ScriptC LLM Engine
==========================================================

Creates a small Llama-like Transformer in PyTorch, exports weights to JSON,
runs inference, and dumps intermediate activations for numerical comparison
with the ScriptC implementation.

Usage:
    python tools/reference/pytorch_reference.py
"""

import json
import math
import sys
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch is required for reference validation.")
    print("Install with: pip install torch")
    sys.exit(1)


# ─── Configuration (matches ScriptC's createDefaultConfig) ───────────────────

CONFIG = {
    "vocabSize": 50,
    "hiddenDim": 16,
    "numLayers": 2,
    "numHeads": 4,
    "numKVHeads": 4,
    "intermediateDim": 32,
    "maxSeqLen": 64,
    "normEps": 1e-6,
    "ropeBase": 10000.0,
}


# ─── RMSNorm (Reference) ────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ─── RoPE (Reference) ───────────────────────────────────────────────────────

def precompute_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # [max_seq_len, head_dim // 2]
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    return cos_freqs, sin_freqs


def apply_rope(x: torch.Tensor, start_pos: int, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor):
    """x: [seq_len, num_heads, head_dim]"""
    seq_len = x.shape[0]
    head_dim = x.shape[2]
    half = head_dim // 2
    out = x.clone()
    for s in range(seq_len):
        pos = start_pos + s
        for h in range(x.shape[1]):
            for i in range(half):
                c = cos_freqs[pos, i].item()
                si = sin_freqs[pos, i].item()
                x0 = x[s, h, 2 * i].item()
                x1 = x[s, h, 2 * i + 1].item()
                out[s, h, 2 * i] = x0 * c - x1 * si
                out[s, h, 2 * i + 1] = x0 * si + x1 * c
    return out


# ─── SwiGLU MLP (Reference) ─────────────────────────────────────────────────

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ─── Multi-Head Attention (Reference) ───────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_rep = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, start_pos: int, cos_freqs, sin_freqs):
        seq_len = x.shape[0]

        Q = self.q_proj(x).view(seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(x).view(seq_len, self.num_kv_heads, self.head_dim)

        Q = apply_rope(Q, start_pos, cos_freqs, sin_freqs)
        K = apply_rope(K, start_pos, cos_freqs, sin_freqs)

        attn_out = torch.zeros(seq_len, self.hidden_dim)

        for h in range(self.num_heads):
            kv_head = h // self.num_rep
            for i in range(seq_len):
                query_pos = start_pos + i
                valid_key_len = query_pos + 1

                scores = []
                max_score = float('-inf')
                for j in range(valid_key_len):
                    dot = 0.0
                    for d in range(self.head_dim):
                        dot += Q[i, h, d].item() * K[j, kv_head, d].item()
                    s = dot * self.scale
                    scores.append(s)
                    if s > max_score:
                        max_score = s

                exps = []
                exp_sum = 0.0
                for j in range(valid_key_len):
                    e = math.exp(scores[j] - max_score)
                    exps.append(e)
                    exp_sum += e

                for d in range(self.head_dim):
                    sum_val = 0.0
                    for j in range(valid_key_len):
                        sum_val += (exps[j] / exp_sum) * V[j, kv_head, d].item()
                    attn_out[i, h * self.head_dim + d] = sum_val

        return self.o_proj(attn_out)


# ─── Transformer Block (Reference) ──────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config["hiddenDim"], config["normEps"])
        self.attention = MultiHeadAttention(
            config["hiddenDim"], config["numHeads"], config["numKVHeads"]
        )
        self.ffn_norm = RMSNorm(config["hiddenDim"], config["normEps"])
        self.feed_forward = SwiGLUMLP(config["hiddenDim"], config["intermediateDim"])

    def forward(self, x, start_pos, cos_freqs, sin_freqs):
        h = x + self.attention(self.attention_norm(x), start_pos, cos_freqs, sin_freqs)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ─── Complete Transformer Model (Reference) ─────────────────────────────────

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config["vocabSize"], config["hiddenDim"])
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config["numLayers"])
        ])
        self.norm = RMSNorm(config["hiddenDim"], config["normEps"])
        self.lm_head = nn.Linear(config["hiddenDim"], config["vocabSize"], bias=False)

        head_dim = config["hiddenDim"] // config["numHeads"]
        self.cos_freqs, self.sin_freqs = precompute_freqs(
            head_dim, config["maxSeqLen"], config["ropeBase"]
        )

    def forward(self, token_ids: list, start_pos: int = 0):
        tokens = torch.tensor(token_ids, dtype=torch.long)
        h = self.tok_embeddings(tokens)  # [seq_len, hidden_dim]

        activations = {"embeddings": h.detach().numpy().tolist()}

        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, self.cos_freqs, self.sin_freqs)
            activations[f"layer_{i}_output"] = h.detach().numpy().tolist()

        h = self.norm(h)
        activations["final_norm"] = h.detach().numpy().tolist()

        logits = self.lm_head(h)
        activations["logits"] = logits.detach().numpy().tolist()

        return logits, activations


# ─── Weight Export (PyTorch → JSON for ScriptC) ─────────────────────────────

def export_weights(model: TransformerModel, filepath: str):
    """Export model weights to JSON format compatible with ScriptC loader."""
    weights = {}

    weights["tok_embeddings"] = model.tok_embeddings.weight.detach().numpy().flatten().tolist()
    weights["norm"] = model.norm.weight.detach().numpy().flatten().tolist()
    weights["lm_head"] = model.lm_head.weight.detach().numpy().flatten().tolist()

    for l, layer in enumerate(model.layers):
        weights[f"layer_{l}.attn_norm"] = layer.attention_norm.weight.detach().numpy().flatten().tolist()
        weights[f"layer_{l}.q_proj"] = layer.attention.q_proj.weight.detach().numpy().flatten().tolist()
        weights[f"layer_{l}.k_proj"] = layer.attention.k_proj.weight.detach().numpy().flatten().tolist()
        weights[f"layer_{l}.v_proj"] = layer.attention.v_proj.weight.detach().numpy().flatten().tolist()
        weights[f"layer_{l}.o_proj"] = layer.attention.o_proj.weight.detach().numpy().flatten().tolist()
        weights[f"layer_{l}.ffn_norm"] = layer.ffn_norm.weight.detach().numpy().flatten().tolist()
        weights[f"layer_{l}.gate_proj"] = layer.feed_forward.gate_proj.weight.detach().numpy().flatten().tolist()
        weights[f"layer_{l}.up_proj"] = layer.feed_forward.up_proj.weight.detach().numpy().flatten().tolist()
        weights[f"layer_{l}.down_proj"] = layer.feed_forward.down_proj.weight.detach().numpy().flatten().tolist()

    data = {"config": model.config, "weights": weights}

    with open(filepath, "w") as f:
        json.dump(data, f)

    print(f"✅ Weights exported to {filepath}")
    return data


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PyTorch Reference Model for ScriptC LLM Validation")
    print("=" * 60)

    torch.manual_seed(42)

    # 1. Create model
    model = TransformerModel(CONFIG)
    model.eval()
    print(f"\n[1/4] Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # 2. Export weights
    os.makedirs("tools/reference/output", exist_ok=True)
    weights_path = "tools/reference/output/reference_model.json"
    export_weights(model, weights_path)

    # 3. Run inference
    test_tokens = [1, 5, 12, 3, 8]
    print(f"\n[2/4] Running inference on tokens: {test_tokens}")

    with torch.no_grad():
        logits, activations = model(test_tokens)

    print(f"      Logits shape: [{logits.shape[0]}, {logits.shape[1]}]")

    # 4. Save activations for comparison
    activations_path = "tools/reference/output/reference_activations.json"
    with open(activations_path, "w") as f:
        json.dump(activations, f)
    print(f"✅ Activations saved to {activations_path}")

    # 5. Print logits summary
    print(f"\n[3/4] Logits for last token (argmax = {logits[-1].argmax().item()}):")
    last_logits = logits[-1].detach().numpy()
    top5_indices = np.argsort(last_logits)[-5:][::-1]
    for idx in top5_indices:
        print(f"      token {idx:4d}: {last_logits[idx]:.6f}")

    # 6. Greedy decode 5 tokens
    print(f"\n[4/4] Greedy generation (5 tokens):")
    generated = list(test_tokens)
    with torch.no_grad():
        for _ in range(5):
            logits_step, _ = model(generated)
            next_token = logits_step[-1].argmax().item()
            generated.append(next_token)
            print(f"      → token {next_token}")

    print(f"\n      Full sequence: {generated}")
    print("=" * 60)


if __name__ == "__main__":
    main()
