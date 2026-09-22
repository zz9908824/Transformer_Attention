import torch
import torch.nn.functional as F
import math
import torch.nn as nn

scores = torch.tensor(
    [
        [1.0,2.0,3.0],
        [4.0,5.0,6.0],
        [7.0,8.0,9.0]
    ]
)
mask = torch.tensor(
    [
        [1,0,0],
        [1,1,0],
        [1,1,1]
    ]
)
scores = scores.masked_fill(mask == 0, float('-inf'))
print(scores)
attention = torch.softmax(scores, dim=-1)
print("attention:", attention)
print("attention.shape:", attention.shape)

seq_len = 3
d_model = 8
W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)
x = torch.randn(seq_len, d_model)

Q = x @ W_Q
K = x @ W_K
V = x @ W_V

weights = Q @ K.transpose(-2, -1) / math.sqrt(d_model)
weights = F.softmax(weights, dim=-1)
output = weights @ V
print("weights.shape:", weights.shape)
print("weights:", weights)
print("output.shape:", output.shape)
print("output:", output)
