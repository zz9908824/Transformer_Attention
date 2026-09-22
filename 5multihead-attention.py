import torch
import torch.nn.functional as F
import math
import torch.nn as nn
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
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
        B,L,d_model = X.size()
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        
        # 分头
        Q_heads = Q.view(B, L, self.num_heads, self.head_dim)
        K_heads = K.view(B, L, self.num_heads, self.head_dim)
        V_heads = V.view(B, L, self.num_heads, self.head_dim)
        Q = Q_heads.transpose(1, 2)
        K = K_heads.transpose(1, 2)
        V = V_heads.transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        output = scores @ V
        output = output.transpose(1,2)
        output = output.contiguous()
        output = output.view(B, L, self.d_model)
        output = self.W_O(output)
        return output

x = torch.randn(2,5,8)
model = MultiHeadAttentionLayer(
    d_model = 8,
    num_heads = 2
)
output = model(x)
print(output.shape)