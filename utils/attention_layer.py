import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qkv_scale=None, dropout=0., attention_dropout=0.):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim // num_heads)
        self.all_head_dim = int(self.head_dim * num_heads)
        self.qkv = nn.Linear(embed_dim, self.all_head_dim * 3, bias=qkv_bias)  # q, k, v
        self.scale = self.head_dim ** -0.5 if qkv_scale is None else qkv_scale  # / sqrt(d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs, seq_len, embed_dim = x.shape  # patches
        qkv = self.qkv(x)  # (bs, seq_len, 3*all_head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # (bs, seq_len, all_head_dim) * 3
        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seq_len, head_dim)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seq_len, head_dim)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seq_len, head_dim)

        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale)  # (bs, num_heads, seq_len, seq_len)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous()  # (bs, seq_len, num_heads, head_dim)
        out = out.view(bs, seq_len, self.all_head_dim)  # (bs, seq_len, all_head_dim)
        out = self.proj(out)  # (bs, seq_len, embed_dim)
        return out
