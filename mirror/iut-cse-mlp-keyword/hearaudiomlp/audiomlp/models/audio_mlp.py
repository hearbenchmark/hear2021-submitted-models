
###############################################################################
# The code for the model is partly adapted from lucidrains/g-mlp-pytorch
###############################################################################
# MIT License
#
# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################


import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from random import randrange


# helpers

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, act = nn.Identity(), init_eps = 1e-3):
        super().__init__()
        dim_out = dim // 2

        self.norm = nn.LayerNorm(dim_out)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)

        self.act = act

        init_eps /= dim_seq
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x):
        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        gate = F.conv1d(gate, weight, bias)

        return self.act(gate) * res


class gMLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_ff,
        seq_len,
        act = nn.Identity(),
        **kwargs
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, act)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.sgu(x)
        x = self.proj_out(x)
        return x


class gMLP_Encoder(nn.Module):
    def __init__(
        self,
        input_res = [40, 98],
        patch_res = [40, 1],
        dim = 64,
        embed_dim = 4,
        embed_drop = 0.0,
        depth = 6,
        ff_mult = 4,
        prob_survival = 0.9,
        pre_norm = False,
        **kwargs
    ):
        super().__init__()

        spec_f, spec_t = input_res
        patch_f, patch_t = patch_res

        assert (spec_f % patch_f) == 0 and (spec_t % patch_t) == 0, 'Spec height and width must be divisible by patch size'
        num_patches = (spec_f // patch_f) * (spec_t // patch_t)

        P_Norm = PreNorm if pre_norm else PostNorm  
        dim_ff = ff_mult * dim

        self.to_patch_embed = nn.Sequential(
            Rearrange('b (f p1) (t p2) -> b (f t) (p1 p2)', p1=patch_f, p2=patch_t),
            nn.Linear(patch_f * patch_t, dim)
        )

        self.prob_survival = prob_survival
        self.layers = nn.ModuleList(
            [Residual(P_Norm(dim, gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=num_patches))) for i in range(depth)]
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Dropout(embed_drop) if embed_drop!=0 else nn.Identity()
        )

    def forward(self, x):
        x = self.to_patch_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        x = self.bottleneck(x)
        return x


class gMLP_Decoder(nn.Module):
    def __init__(
        self,
        input_res = [40, 98],
        patch_res = [40, 1],
        dim = 64,
        embed_dim = 4,
        depth = 6,
        ff_mult = 4,
        prob_survival = 0.9,
        pre_norm = False,
        scale_in = True,
        **kwargs
    ):
        super().__init__()

        spec_f, spec_t = input_res
        patch_f, patch_t = patch_res

        assert (spec_f % patch_f) == 0 and (spec_t % patch_t) == 0, 'Spec height and width must be divisible by patch size'
        num_patches = (spec_f // patch_f) * (spec_t // patch_t)

        P_Norm = PreNorm if pre_norm else PostNorm  
        dim_ff = ff_mult * dim

        self.project = nn.Linear(embed_dim, dim)

        self.prob_survival = prob_survival
        self.layers = nn.ModuleList(
            [Residual(P_Norm(dim, gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=num_patches))) for i in range(depth)]
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_f * patch_t),
            Rearrange("b t f -> b f t"),
            nn.Tanh() if scale_in else nn.Identity()
        )

    def forward(self, x):
        x = self.project(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        x = self.gate(x)
        return x



class AudioMAE(nn.Module):
    """Audio MLP Autoencoder"""
    
    def __init__(
        self,
        input_res = [40, 98],
        patch_res = [40, 1],
        dim = 64,
        embed_dim = 8,
        embed_drop = 0.0,
        encoder_depth = 6,
        decoder_depth = 6,
        ff_mult = 4,
        prob_survival = 0.9,
        pre_norm = False,
        audio_processor = nn.Identity(),
        scale_in = True,
        **kwargs
    ):
        super().__init__()

        self.audio_processor = audio_processor

        self.encoder = gMLP_Encoder(
            input_res = input_res,
            patch_res = patch_res,
            dim = dim,
            embed_dim = embed_dim,
            embed_drop = embed_drop,
            depth = encoder_depth,
            ff_mult = ff_mult,
            prob_survival = prob_survival,
            pre_norm = pre_norm
        )

        self.decoder = gMLP_Decoder(
            input_res = input_res,
            patch_res = patch_res,
            dim = dim,
            embed_dim = embed_dim,
            depth = decoder_depth,
            ff_mult = ff_mult,
            prob_survival = prob_survival,
            pre_norm = pre_norm,
            scale_in = scale_in
        )
        

    def forward(self, x):
        x = self.audio_processor(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class KW_MLP(nn.Module):
    """Keyword-MLP."""
    
    def __init__(
        self,
        input_res = [40, 98],
        patch_res = [40, 1],
        num_classes = 35,
        dim = 64,
        depth = 12,
        ff_mult = 4,
        channels = 1,
        prob_survival = 0.9,
        pre_norm = False,
        **kwargs
    ):
        super().__init__()
        image_height, image_width = input_res
        patch_height, patch_width = patch_res
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        P_Norm = PreNorm if pre_norm else PostNorm
        
        dim_ff = dim * ff_mult

        self.to_patch_embed = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(channels * patch_height * patch_width, dim)
        )

        self.prob_survival = prob_survival

        self.layers = nn.ModuleList(
            [Residual(P_Norm(dim, gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=num_patches))) for i in range(depth)]
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        return self.to_logits(x)