import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttentionMasked(nn.Module):
    def __init__(self, d_model, num_heads) :
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, X):
        """
        前向传播
        """
        B,L,_ = X.size()
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        mask = torch.tril(torch.ones(L, L,device = X.device))

        Q = Q.view(B, L, self.num_heads, self.head_dim)
        K = K.view(B, L, self.num_heads, self.head_dim)
        V = V.view(B, L, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        output = attention @ V
        output = output.transpose(1,2)
        output = output.contiguous()
        output = output.view(B, L, self.d_model)
        output = self.W_O(output)
        return output,attention

x = torch.randn(2,5,8)
model = MultiHeadAttentionMasked(
    d_model = 8,
    num_heads = 2
)
output,scores = model(x)
print(output.shape)
print(scores.shape)
print(scores)