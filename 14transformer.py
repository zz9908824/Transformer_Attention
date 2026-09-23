import torch
import torch.nn as nn
import math
import re
#transformer with padding mask: 1 for real tokens, 0 for padding tokens
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
    def forward(self,x,padding_mask):
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
        # padding mask
        # 计算注意力权重
        if padding_mask is not None:
            # padding_mask: [batch_size, seq_len]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = attn_weights.masked_fill(padding_mask == 0, float('-inf'))
        else:
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
    def forward(self,x,tgt_padding_mask):
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
        if tgt_padding_mask is not None:
            masked_attn_weights = masked_attn_weights.masked_fill(tgt_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
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
    def forward(self,x,encoder_output,src_padding_mask=None):
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
        if src_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(src_padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
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
    def forward(self,x, encoder_output, src_padding_mask, tgt_padding_mask=None):
        # x[batch_size, seq_len, d_model]
        x = self.masked_self_attention(x,tgt_padding_mask)
        x = self.cross_attention(x, encoder_output, src_padding_mask)
        x = self.ffn(x)
        return x
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1) :
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=1000)
        self.decode_blocks = nn.ModuleList([
                DecodeBlock(d_model, n_heads, hidden_dim, dropout) for _ in range(6)
            ])
    def forward(self,x, encoder_output, src_padding_mask, tgt_padding_mask=None):
        x = self.positional_encoding(x)
        for block in self.decode_blocks:
            x = block(x, encoder_output, src_padding_mask, tgt_padding_mask)
        return x
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.mha = MHA_Encoder(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, hidden_dim, dropout)
    def forward(self,x,padding_mask):
        # x[batch_size, seq_len, d_model]
        x = self.mha(x,padding_mask)
        x = self.ffn(x)
        return x
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=1000)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, hidden_dim, dropout) for _ in range(6)
        ])
    def forward(self, x, padding_mask=None):
        x = self.positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x, padding_mask)
        return x
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.encoder = Encoder(d_model, n_heads, hidden_dim, dropout)
        self.decoder = Decoder(d_model, n_heads, hidden_dim, dropout)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
    def forward(self, src, tgt):
        src_padding_mask = (src != 0) # 假设 0 是 padding token
        tgt_padding_mask = (tgt != 0) # 假设 0 是 padding token
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        encoder_output = self.encoder(src, src_padding_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_padding_mask, tgt_padding_mask)
        logits = self.output_linear(decoder_output)
        return logits


class WordTokenizer:
    """规则词级分词器；编号 0--3 固定给特殊标记。"""
    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
    def __init__(self, training_texts):
        self.token_to_id = {self.PAD: 0, self.UNK: 1, self.BOS: 2, self.EOS: 3}
        for text in training_texts:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)
    @staticmethod
    def tokenize(text):
        # 英文按词切分，标点单独保留；这里的中文示例预先以空格分词。
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    def encode(self, text, *, bos=False, eos=False):
        ids = [self.token_to_id.get(token, self.token_to_id[self.UNK])
               for token in self.tokenize(text)]
        if bos:
            ids.insert(0, self.token_to_id[self.BOS])
        if eos:
            ids.append(self.token_to_id[self.EOS])
        return ids
def pad_batch(sequences, pad_id):
    """把变长 token-id 序列补齐，返回 LongTensor [batch, max_length]。"""
    max_length = max(map(len, sequences))
    return torch.tensor(
        [seq + [pad_id] * (max_length - len(seq)) for seq in sequences],
        dtype=torch.long,
    )
src_texts = ["I like machine learning .", "Attention is all you need .","I love NLP ."] 
tgt_texts = ["我 喜欢 机器 学习 。", "注意力 就是 你 所 需要 的 一切 。","我 爱 NLP 。"]
# 翻译的源语言与目标语言分别建词表，因此同一个 id 在两侧没有共同语义。
src_tokenizer = WordTokenizer(src_texts)
tgt_tokenizer = WordTokenizer(tgt_texts)
src_vocab_size = len(src_tokenizer.token_to_id)
tgt_vocab_size = len(tgt_tokenizer.token_to_id)
# Encoder: <bos> source tokens <eos>
src_ids = pad_batch(
    [src_tokenizer.encode(text, bos=True, eos=True) for text in src_texts], pad_id=0
)
# Decoder teacher forcing：输入右移，标签左移一位。
tgt_input_ids = pad_batch(
    [tgt_tokenizer.encode(text, bos=True) for text in tgt_texts], pad_id=0
)
tgt_label_ids = pad_batch(
    [tgt_tokenizer.encode(text, eos=True) for text in tgt_texts], pad_id=0
)

print("src tokens:", [src_tokenizer.tokenize(text) for text in src_texts])
print("src ids:\n", src_ids)
print("tgt decoder-input ids:\n", tgt_input_ids)
print("tgt labels:\n", tgt_label_ids)

d_model = 32
n_heads = 4
hidden_dim = 64
transformer = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout=0.1,
)
epoch = 10
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)
for i in range(epoch):
    optimizer.zero_grad()
    output_logits = transformer(src_ids, tgt_input_ids)    
    loss = criterion(
        output_logits.reshape(-1, output_logits.size(-1)), tgt_label_ids.reshape(-1)
    )
    print("loss:", loss.item())
    loss.backward()
    optimizer.step()
    print(f"Epoch {i+1}/{epoch} finished. Loss: {loss.item():.4f}")
