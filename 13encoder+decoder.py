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
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    def forward(self,x):
        # x[batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True ,unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return self.weight * x + self.bias
class MHA_Encoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.layer_norm = LayerNorm(d_model)
    def forward(self,x):
        # x[batch_size, seq_len, d_model]
        residual = x
        seq_len = x.size(1)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # 分头 [batch_size, seq_len, n_heads, head_dim]
        Q = Q.view(-1,seq_len , self.n_heads, self.head_dim)
        K = K.view(-1,seq_len , self.n_heads, self.head_dim)
        V = V.view(-1,seq_len, self.n_heads, self.head_dim)
        # [batch_size, n_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # 计算注意力权重
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # 合并头
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(-1, seq_len, self.n_heads * self.head_dim)
        # residual connection
        attn_output = self.W_O(attn_output)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output + residual
        # layer norm
        attn_output = self.layer_norm(attn_output)
        return attn_output
class Masked_SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.layer_norm = LayerNorm(d_model)
    def forward(self,x):
        # x[batch_size, seq_len, d_model]
        residual = x
        seq_len = x.size(1)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # 创建下三角掩码
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        # 分头 [batch_size, seq_len, n_heads, head_dim]
        Q = Q.view(-1,seq_len , self.n_heads, self.head_dim)
        K = K.view(-1,seq_len , self.n_heads, self.head_dim)
        V = V.view(-1,seq_len, self.n_heads, self.head_dim)
        # [batch_size, n_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # 计算注意力权重
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 应用掩码
        masked_attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        masked_attn_weights = torch.softmax(masked_attn_weights, dim=-1)
        attn_output = torch.matmul(masked_attn_weights, V)
        # 合并头
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(-1, seq_len, self.n_heads * self.head_dim)
        # residual connection
        attn_output = self.W_O(attn_output)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output + residual
        # layer norm
        attn_output = self.layer_norm(attn_output)
        return attn_output
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.layer_norm = LayerNorm(d_model)
    def forward(self,x,encoder_output):
        # x[batch_size, seq_len, d_model]
        residual = x
        seq_len = x.size(1)
        encoder_seq_len = encoder_output.size(1)
        # 交叉注意力
        Q = self.W_Q(x)
        K = self.W_K(encoder_output)
        V = self.W_V(encoder_output)
        # 分头 [batch_size, seq_len, n_heads, head_dim]
        Q = Q.view(-1,seq_len , self.n_heads, self.head_dim)
        K = K.view(-1,encoder_seq_len , self.n_heads, self.head_dim)
        V = V.view(-1,encoder_seq_len, self.n_heads, self.head_dim)
        # [batch_size, n_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        # 计算注意力权重
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # 合并头
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(-1, seq_len, self.n_heads * self.head_dim)
        # residual connection
        attn_output = self.W_O(attn_output)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output + residual
        # layer norm
        attn_output = self.layer_norm(attn_output)
        return attn_output

class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, hidden_dim)
        self.W_2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
    def forward(self,x):
        # x[batch_size, seq_len, d_model]
        residual = x
        x = self.W_1(x)
        x = torch.relu(x)
        x = self.W_2(x)
        # x = torch.relu(x)
        # dropout and activation
        x = self.dropout(x)
        # residual connection
        x = x + residual
        # layer norm
        x = self.layer_norm(x)
        return x

class DecodeBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.masked_self_attention = Masked_SelfAttention(d_model, n_heads, dropout)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, hidden_dim, dropout)
    def forward(self,x, encoder_output):
        # x[batch_size, seq_len, d_model]
        x = self.masked_self_attention(x)
        x = self.cross_attention(x, encoder_output)
        x = self.ffn(x)
        return x
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1) :
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=1000)
        self.decode_blocks = nn.ModuleList([
                DecodeBlock(d_model, n_heads, hidden_dim, dropout) for _ in range(6)
            ])
    def forward(self,x, encoder_output):
        x = self.positional_encoding(x)
        for block in self.decode_blocks:
            x = block(x, encoder_output)
        return x
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.mha = MHA_Encoder(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, hidden_dim, dropout)
    def forward(self,x):
        # x[batch_size, seq_len, d_model]
        x = self.mha(x)
        x = self.ffn(x)
        return x
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=1000)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, hidden_dim, dropout) for _ in range(6)
        ])
    def forward(self, x):
        x = self.positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x

# d_model = 8
# head_nums = 2
# seq_len = 5
# hidden_dim = 16
# dropout = 0.1
# x = torch.randn(2, seq_len, d_model)

# encode = Encoder(d_model, head_nums, hidden_dim, dropout)
# encoded_output = encode(x)
# print("Encoded output shape:", encoded_output.shape)
# decoder = Decoder(d_model, head_nums, hidden_dim, dropout)
# decoded_output = decoder(x,encoded_output)
# print("Decoded output shape:", decoded_output.shape)

d_model = 8
head_nums = 2
hidden_dim = 16
dropout = 0.1

src_len = 7
tgt_len = 5

src = torch.randn(2, src_len, d_model)
tgt = torch.randn(2, tgt_len, d_model)

encoder = Encoder(
    d_model,
    head_nums,
    hidden_dim,
    dropout
)

encoder_output = encoder(src)

print(
    "Encoder output:",
    encoder_output.shape
)

decoder = Decoder(
    d_model,
    head_nums,
    hidden_dim,
    dropout
)

decoder_output = decoder(
    tgt,
    encoder_output
)

print(
    "Decoder output:",
    decoder_output.shape
)