import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
        return x+self.pe[:, :seq_len, :]
        
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
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self,x):
        residual = x
        # x[batch_size, seq_len, d_model]
        seq_len = x.size(1)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # [batch_size, seq_len, n_heads, head_dim]
        Q = Q.view(-1, seq_len, self.n_heads, self.head_dim)
        K = K.view(-1, seq_len, self.n_heads, self.head_dim)
        V = V.view(-1, seq_len, self.n_heads, self.head_dim)
        # [batch_size, n_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # [batch_size, n_heads, seq_len, seq_len]
        QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(QK, dim=-1)
        # [batch_size, n_heads, seq_len, seq_len]
        output = torch.matmul(attention, V)
        # [batch_size, seq_len, n_heads, head_dim]
        output = output.transpose(1, 2)
        # [batch_size, seq_len, d_model]
        output = output.contiguous()
        output = output.view(-1, seq_len, self.n_heads * self.head_dim)
        # residual connection
        output = self.linear_out(output)
        output = self.dropout(output)
        output = output + residual
        return output

d_model = 8
seq_len = 5
n_heads = 2

x = torch.randn(1, seq_len, d_model)
pe = PositionalEncoding(d_model, max_len=1000)
xwithpos = pe(x)
print(xwithpos)
print(xwithpos.shape)
mha = MHA_Residual(d_model, n_heads)
output = mha(xwithpos)
print(output)
print(output.shape)
