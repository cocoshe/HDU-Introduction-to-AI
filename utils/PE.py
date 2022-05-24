import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super(PatchEmbedding, self).__init__()
        num_patch = int(img_size // patch_size) * int(img_size // patch_size)
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim,
                                         kernel_size=patch_size, stride=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [bs, c, h, w]
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)  # [bs, 1, embed_dim]
        x = self.patch_embedding(x)  # [bs, embed_dim, n_patch_h, n_patch_w]
        x = x.flatten(2)  # [bs, embed_dim, h*w]
        x = x.transpose(1, 2)  # [bs, h*w, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [bs, h*w+1, embed_dim]

        embeddings = x + self.pos_embedding  # [bs, h*w+1, embed_dim]
        embeddings = self.dropout(embeddings)
        return embeddings

