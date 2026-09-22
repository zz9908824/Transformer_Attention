import torch
import torch.nn.functional as F
import math

def attention(q, k, v):
    """
    计算注意力权重
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    output = scores @ v
    return output, scores

def self_attention(X):
    """
    计算自注意力权重
    """
    d_model = X.size(-1)
    W_Q = torch.randn(d_model, d_model)
    W_K = torch.randn(d_model, d_model)
    W_V = torch.randn(d_model, d_model)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    output, scores = attention(Q, K, V)
    return output, scores

seq_len = 4
d_k = 8
batch_size = 1
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)
output, scores = attention(Q, K, V)

# 自注意力
x = torch.randn(batch_size, seq_len, d_k)
output, scores = self_attention(x)
print(output.shape)
print(scores.shape)
