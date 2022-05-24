import torch
import torch.nn as nn
from utils.attention_layer import Attention
from utils.MLP import MLP
from utils.PE import *


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, qkv_bias=False, qkv_scale=None, mlp_ratio=4.0, dropout=0., attention_dropout=0.):
        super(EncoderLayer, self).__init__()
        self.atten = Attention(embed_dim, num_heads, qkv_bias, qkv_scale, dropout, attention_dropout)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.atten_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.atten_norm(x)
        x = self.atten(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, depth, embed_dim):
        super(Encoder, self).__init__()
        layer_list = []
        for i in range(depth):
            layer_list.append(EncoderLayer(embed_dim))
        self.layer_list = layer_list
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)

        x = self.norm(x)
        return x


class Vit(nn.Module):
    def __init__(self, img_size=28,
                 in_channels=3,
                 patch_size=4,
                 embed_dim=768,
                 depth=6,
                 num_heads=4,
                 qkv_bias=False,
                 qkv_scale=None,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.,
                 num_classes=10,):
        super(Vit, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, dropout)
        self.encoder = Encoder(depth, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)[:, 0]

        return x