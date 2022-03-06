"""vision transformer adopted from https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch"""

from torch import nn
import numpy as np
from einops.layers.torch import Rearrange
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from udons.models.modules import utils


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SpecTransformer(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        dim = hparams["model_dim"]
        self.hparams = hparams

        patch_modules = []
        if hparams["instance_norm"]:
            patch_modules.append(Rearrange('(b p) c f t -> b p (c f) t', p=hparams["nb_patches"]))
            patch_modules.append(torch.nn.InstanceNorm2d(hparams["nb_patches"], affine=False))
            # todo make this generic to have a conv2d encoder
            patch_modules.append(Rearrange('b p (c f) t -> (b p) c f t', c=hparams["nb_channels"]))           

        if hparams["patch_encoder"] == "linear":
            patch_modules.append(Rearrange('(b p) c f t -> b p (c f t)', p=hparams["nb_patches"]))
            for i in range(hparams["mlp_layers"]):
                patch_modules.append(nn.Linear(hparams["patch_len"] * hparams["n_mels"], hparams["patch_len"] * hparams["n_mels"]))
            patch_modules.append(nn.Linear(hparams["patch_len"] * hparams["n_mels"], dim))
        elif hparams["patch_encoder"] == "conv":
            patch_modules.append(
                nn.Sequential(
                    nn.Conv2d(hparams["nb_channels"], 64, kernel_size=7, stride=2, padding=2),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.Flatten(1),
                    nn.Linear(3840, dim),
                    Rearrange('(b p) d -> b p d', p=hparams["nb_patches"])
                )
            )

        self.patch_encoding = nn.Sequential(*patch_modules)

        self.pos_embedding = nn.Parameter(torch.randn(1, hparams["nb_patches"] + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(hparams["emb_dropout"])

        self.transformer = Transformer(dim, hparams["depth"], hparams["heads"], hparams["dim_head"], hparams["mlp_dim"], hparams["dropout"])

        self.pool = hparams["pool"]
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hparams["nb_classes"])
        )

    def forward(self, x):
        # get do not apply siamese view for this model
        x = self.patch_encoding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.hparams["pos_embed"]:
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class SpecKKTransformer(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        dim = hparams["model_dim"]
        self.hparams = hparams

        patch_modules = []
        if hparams["instance_norm"]:
            patch_modules.append(Rearrange('(b p) c f t -> b p (c f) t', p=hparams["nb_patches"]))
            patch_modules.append(torch.nn.InstanceNorm2d(hparams["nb_patches"], affine=False))
            # todo make this generic to have a conv2d encoder
            patch_modules.append(Rearrange('b p (c f) t -> (b p) c f t', c=hparams["nb_channels"]))           

        if hparams["patch_encoder"] == "linear":
            patch_modules.append(Rearrange('(b p) c f t -> b p (c f t)', p=hparams["nb_patches"]))
            for i in range(hparams["mlp_layers"]):
                patch_modules.append(nn.Linear(hparams["patch_len"] * hparams["n_mels"], hparams["patch_len"] * hparams["n_mels"]))
            patch_modules.append(nn.Linear(hparams["patch_len"] * hparams["n_mels"], dim))
        elif hparams["patch_encoder"] == "conv":
            patch_modules.append(
                nn.Sequential(
                    nn.Conv2d(hparams["nb_channels"], 64, kernel_size=7, stride=2, padding=2),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.Flatten(1),
                    nn.Linear(3840, dim),
                    Rearrange('(b p) d -> b p d', p=hparams["nb_patches"])
                )
            )

        self.patch_encoding = nn.Sequential(*patch_modules)

        self.pos_embedding = nn.Parameter(torch.randn(1, hparams["nb_patches"] + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(hparams["emb_dropout"])

        self.transformer = Transformer(dim, hparams["depth"], hparams["heads"], hparams["dim_head"], hparams["mlp_dim"], hparams["dropout"])

        self.pool = hparams["pool"]
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hparams["nb_patches"]*hparams["nb_patches"])
        )

    def forward(self, x):
        # get do not apply siamese view for this model
        x = self.patch_encoding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.hparams["pos_embed"]:
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)