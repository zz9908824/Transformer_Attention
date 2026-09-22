import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class MyLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        # x[batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        output = self.weight * x_norm + self.bias
        return output

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, max_len=5000):
       super().__init__()
       self.d_model = d_model
       pe = torch.zeros(max_len, d_model)
       position = torch.arange(max_len).unsqueeze(1)
       div_term = torch.exp(
           torch.arange(0,d_model,2)
           * (-math.log(10000.0)/d_model)
       )
       # 偶数维
       pe[:,0::2] = torch.sin(position * div_term)
       # 奇数维
       pe[:,1::2] = torch.cos(position * div_term)
       pe = pe.unsqueeze(0)
       self.register_buffer('pe', pe)
    def forward(self,x):
        # x[batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x+ self.pe[:, :seq_len, :]

class MHA_Residual(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.layer_norm = MyLayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        # x[batch_size, seq_len, d_model]
        residual = x
        seq_len = x.size(1)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # 分头 [batch_size, seq_len, n_heads, head_dim]
        Q = Q.view(-1, seq_len, self.n_heads, self.head_dim)
        K = K.view(-1, seq_len, self.n_heads, self.head_dim)
        V = V.view(-1, seq_len, self.n_heads, self.head_dim)
        # [batch_size,n_heads,seq_len,head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 计算注意力分布
        attn_dist = torch.softmax(attn_scores, dim=-1)
        # 计算输出
        output = torch.matmul(attn_dist, V)
        # 合并头 [batch_size, seq_len, n_heads, head_dim]
        output = output.transpose(1, 2)
        output = output.contiguous()
        output = output.view(-1, seq_len, self.d_model)
        # 残差连接
        output = self.W_O(output)
        output = self.dropout(output)
        output = residual + output
        # 层归一化
        output_norm = self.layer_norm(output)
        return output_norm

d_model = 8
n_heads = 2
seq_len = 5
x = torch.randn(1, seq_len, d_model)
pe = PositionalEncoding(d_model)
x_pos = pe(x)
print(x_pos)
print(x_pos.shape)
mha = MHA_Residual(d_model, n_heads)
output = mha(x_pos)
print(output)
print(output.shape)


