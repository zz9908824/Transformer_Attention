import torch
import torch.nn as nn
import math

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

pe = PositionalEncoding(8, max_len=1000)
x = torch.randn(1, 10, 8)
x_pos = pe(x)
print(x_pos)
print(x_pos.shape)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.imshow(x_pos[0, :, :])
plt.show()
plt.figure(figsize=(12, 8))
plt.imshow(pe.pe[0, :, :],aspect='auto')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.show()
       