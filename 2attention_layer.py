import torch
import torch.nn.functional as F
import math
import torch.nn as nn

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

batch_size = 2
seq_len = 4
d_k = 8
x = torch.randn(batch_size, seq_len, d_k)
selfattention = SelfAttentionLayer(d_k)
output, scores = selfattention(x)
print(output.shape)
print(scores.shape)