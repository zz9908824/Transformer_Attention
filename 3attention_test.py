import torch
import torch.nn.functional as F
import math
import torch.nn as nn

torch.manual_seed(123)
torch.cuda.manual_seed(123)

tokens = ["我","喜欢","nlp"]
seq_len = len(tokens)
d_model = 8
x = torch.randn(seq_len, d_model)

# W_Q = torch.randn(d_model, d_model)
# W_K = torch.randn(d_model, d_model)
# W_V = torch.randn(d_model, d_model)

# Q = x @ W_Q
# K = x @ W_K
# V = x @ W_V

# scores = Q @ K.transpose(-2, -1) / math.sqrt(d_model)
# scores = F.softmax(scores, dim=-1)
# output = scores @ V
# print(output.shape)
# print(scores.shape)
# print("X.shape:", x.shape)
# print("Q.shape:", Q.shape)
# print("K.shape:", K.shape)
# print("V.shape:", V.shape)
# print("score:", scores)
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
    def forward(self, X):
        """
        前向传播
        """
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        scores = F.softmax(scores, dim=-1)
        output = scores @ V
        return output, scores

selfattention = SelfAttentionLayer(d_model)
output, scores = selfattention(x)
print(output.shape)
print(scores.shape)
print("score:", scores)
