import math
import random
from typing import List
import logging

import torch
from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F

from .config import DBERTConfig, KMGeneratorConfig, DBERT2HeadConfig
from .kmeans import *

def get_mask_from_lengths(lengths):
    """ 
    Create non-causal masks from `lengths`. 
    A mask of shape (lengths.shape[0], max len in lengths)
    is made where each entry is False if that position is
    a padding entry -- i.e. if that point is outside the length 
    specified in `lengths`. It is purely masking out regions where
    the input utterances have been padded. This is not a causal mask. 
    """
    max_len = torch.max(lengths)
    ids = torch.arange(0, max_len, device=max_len.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

class SamePad(nn.Module):
    """ Obtained from fairseq """
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x

class RelativePositionalEncoder(nn.Module):

    def __init__(self, dim, conv_pos, conv_pos_groups):
        super().__init__()
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.embedding_dim = dim
        
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())

    def forward(self, x):
        """ `x` of shape (bs, seq_len, dim) """
        x_conv = self.pos_conv(x.permute(0, 2, 1)) # (bs, seq_len, dim) -> (bs, dim, seq_len)
        x_conv = x_conv.permute(0, 2, 1) # (bs, dim, seq_len) -> (bs, seq_len, dim)
        return x + x_conv

class MaskEmb(nn.Module):

    def __init__(self, emb_dim, mtype='append'):
        """ `mtype` can be 'append' or 'replace', where 'replace' replaces each masked 
        position with a learned embedding, while 'append' appens a learned embedding to masked 
        and unmasked positions. 
        """
        super().__init__()
        self.mtype = mtype
        if self.mtype == 'append': self.embedding = nn.Embedding(2, emb_dim)
        else: self.embedding = nn.Embedding(1, emb_dim) # self.embedding.weight is (1, emb_dim)
        nn.init.normal_(self.embedding.weight, 0, 0.1)

    def forward(self, seq, mask):
        """ 
        Generates masks where each seq element may be the start of a mask span with
        probability `p`, and the mask span length is sampled between `max_span`
        and `min_span` identically for the entire batch. 
        `seq` : (bs, T=seq_len, dim)
        `mask`: (bs, seq_len)
        """
        if self.mtype == 'replace':
            seq[mask] = self.embedding.weight[0]
        elif self.mtype == 'append':
            embs = self.embedding(mask.long()) # (bs, seq len, emb_dim)
            seq.masked_fill_(mask[..., None], 0.0)
            seq = torch.cat((seq, embs), dim=-1)
        return seq

class SummaryEmb(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(1, emb_dim)
        nn.init.normal_(self.embedding.weight, 0, 0.1)

    def forward(self, seq: Tensor, mask: Tensor):
        """ 
        Creates an additional vector at the start of each utterance
        corresponding to a `summary` vector. 
        `seq`: (bs, seq_len, dim)
        `mask`: (bs, seq_len)
        """
        bs = seq.shape[0]
        # preappend summary vec
        w = self.embedding.weight[None].repeat(bs, 1, 1) # (bs, 1, dim)
        seq = torch.cat((w, seq), dim=1)
        # update mask to reflect this
        mask = F.pad(mask, (1, 0), value=False)
        return seq, mask

class KMGenerator(nn.Module):
    def __init__(self, cfg: KMGeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.km_modules = nn.ModuleList([KMean(k, cfg.dim, dist=cfg.dist, 
                                reassign_empty=cfg.reassign_empty, add_forcing=cfg.add_forcing,
                                forcing_coeff=cfg.forcing_coeff) for k in cfg.K])
        print(f"[KMGenerator] Model has {sum([p.numel() for p in self.parameters()]):,d} parameters")

    @torch.no_grad()
    def init_clusters(self, vs: List[Tensor]) -> None:
        for km, v in zip(self.km_modules, vs):
            km.init_clusters(v)
        print("[KMGenerator] Clusters initialized.")

    def forward(self, vs: List[Tensor]) -> Tensor:
        losses = []
        assignments = []
        for i, km in enumerate(self.km_modules):
            lo, assignment = km(vs[i])
            losses.append(lo)
            assignments.append(assignment)
        return torch.stack(losses, dim=0).mean(), assignments

class KMean(nn.Module):
    """
    The idea is that the main model is the discriminator -- it attempts to 
    discriminate its own output data from the kmean centroids, while the k-mean model
    attempts to generate centroids that best fit the data.  
    """

    def __init__(self, K, dim, dist='euclid', reassign_empty=False, 
                 add_forcing=False, forcing_coeff=None):
        super().__init__()
        if dist == 'euclid': self.dist_func = smart_euclid_dist
        elif dist == 'cosine': self.dist_func = smart_cosine_dist
        else: raise ValueError("Unrecognized distance func. Only `euclid` or `cosine` supported.")
        self.centroids = nn.parameter.Parameter(0.2*torch.randn(K, dim, dtype=torch.float))
        self.K = K
        self.reassign_empty = reassign_empty
        self.add_forcing = add_forcing
        self.forcing_coeff = forcing_coeff
        
    @torch.no_grad()
    def init_clusters(self, v: Tensor) -> Tensor:
        """ `v` : (bs, seq_len, dim) """
        dim = v.shape[-1]
        vflat = v.reshape(-1, dim)
        if self.dist_func == smart_cosine_dist: _tmp = cosine_dist
        else: _tmp = euclid_dist
        init_clusters = kmeans_pp_init(vflat.T, self.K, _tmp)
        self.centroids.data = init_clusters.T
        self._arange_set = set(range(0, self.K))

    def forward(self, v: Tensor) -> Tensor:
        """ `v` : (bs, seq_len, dim) """
        # Find distances
        dists = self.dist_func(v, self.centroids) # (bs, seq_len, K)
        assigned_classes = dists.argmin(dim=-1) # (bs, seq_len)

        if self.reassign_empty:
            bs, seq_len, dim = v.shape
            unique_set = set(assigned_classes.unique().tolist())
            if len(unique_set) < len(self._arange_set): 
                logging.warning(f"K-mean has centroids with no data samples; {self._arange_set - unique_set} empty clusters!")
            for i in self._arange_set - unique_set:
                # For each centroid with no points assigned, re-assign it to random point
                self.centroids.data[i] = v[random.randint(0, bs-1), random.randint(0, seq_len-1)].detach()

        t_jn = F.one_hot(assigned_classes, num_classes=self.K).bool()
        loss = dists[t_jn].mean() if self.dist_func == smart_euclid_dist else dists[t_jn].sum()

        if self.add_forcing:
            closest_dists = torch.amin(dists, dim=(0, 1)) if dists.dim() == 3 else torch.amin(dists, dim=(0))
            loss += self.forcing_coeff*(closest_dists.mean() if self.dist_func == smart_euclid_dist else closest_dists.sum())
        return loss, assigned_classes

# --------------------------------------------------------------------------------------
# ------------------------------- REVISED MODELS ---------------------------------------
# --------------------------------------------------------------------------------------

class ClsHead2(nn.Module):
    def __init__(self, K, dim, summary_dim):
        super().__init__()
        self.proj = nn.Linear(dim + summary_dim, dim, bias=False)
        self.codebook = nn.parameter.Parameter(torch.empty(K, dim, dtype=torch.float))
        self.codebook.requires_grad_(True)
        nn.init.normal_(self.codebook, std=0.2)
        
    def forward(self, c: Tensor, summary: Tensor):
        """ `(bs, seq len, dim)`, returns `(bs, seq_len, K)` cosine similarities """
        if summary is None: joint = c
        else:
            seq_len = c.shape[1]
            s = summary.unsqueeze(1).repeat(1, seq_len, 1) # (bs, seq_len, dim)
            joint = torch.cat((s, c), dim=-1)
    
        sims = smart_cosine_sim(self.proj(joint), self.codebook) # (bs, seq_len, K)
        return sims

class DBERT2(nn.Module):

    def __init__(self, cfg: DBERTConfig):
        super().__init__()
        self.conv_extractor1 = ConvFeatureExtractionModel(cfg.conv1_cfg, dropout=0.0, 
                                                           mode='default', conv_bias=False)
        # map conv dim to transformer dim
        self.post_extract_proj = (
            nn.Linear(cfg.dim, cfg.transformer_dim)
            if cfg.dim != cfg.transformer_dim
            else nn.Identity()
        )
        # summary stuffs
        self.layernorm = nn.LayerNorm(cfg.transformer_dim)
        self.masker = MaskEmb(emb_dim=cfg.transformer_dim, mtype='replace')
        self.summary_embedder = SummaryEmb(cfg.transformer_dim) if cfg.use_summary_vec else nn.Identity()

        # positional embeddings
        self.positional_embedding = RelativePositionalEncoder(cfg.transformer_dim, cfg.conv_pos, cfg.conv_pos_groups)
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(cfg.transformer_dim, cfg.n_heads, 
                                        cfg.transformer_dim*cfg.d_ff_mult, batch_first=True, 
                                        dropout=cfg.transformer_dp) for _ in range(cfg.layers)])
        
        self.final_proj = nn.Linear(cfg.transformer_dim, cfg.dim) if cfg.transformer_dim != cfg.dim else nn.Identity()
        self.summary_proj = nn.Linear(cfg.transformer_dim, cfg.summary_dim) if cfg.use_summary_vec else nn.Identity()        

        # Define projection heads:
        self.head_cfg: DBERT2HeadConfig = cfg.db2
        
        if cfg.use_cls_head == False: raise AssertionError("DBERT2 can only be used with cls heads.")

        self.s_heads = nn.ModuleList([ClsHead2(k, cfg.summary_dim, 0) for k in self.head_cfg.summary_K])
        self.c_heads = nn.ModuleList([ClsHead2(k, cfg.dim, cfg.summary_dim) for k in self.head_cfg.conv_K])
        self.o_heads = nn.ModuleList([ClsHead2(k, cfg.dim, cfg.summary_dim) for k in self.head_cfg.output_K])

        self.cfg: DBERTConfig = cfg
        
        print(f"[DBERT] Model has {sum([p.numel() for p in self.parameters()]):,d} parameters")

    def forward(self, wv1, wv1_lengths, masks, feats_only=False, ignore_mask=False):
        c = self.conv_extractor1(wv1)
        c = c.permute(0, 2, 1) # (bs, c_dim, T) --> (bs, T, c_dim)
        conv_feats = c.clone()
        # given the stride settings at the top, this is the factor.
        c_lengths = (1 + (wv1_lengths - self.cfg.conv_magic_offset)/self.cfg.conv_reduction_mult).floor().long() 

        c = self.post_extract_proj(c) # (..., dim) --> (..., transformer_dim)
        c = self.layernorm(c)
        
        noncausal_masks = ~get_mask_from_lengths(c_lengths) # must invert to work with pytorch attention implementation

        # Path 1: no mask
        c = c.clone()
        if not ignore_mask: c = self.masker(c, masks)
        c = self.positional_embedding(c)
        if self.cfg.use_summary_vec:
            c, noncausal_masks = self.summary_embedder(c, noncausal_masks)
        for mod in self.transformer_layers:
            c = mod(c, src_key_padding_mask=noncausal_masks)

        if self.cfg.use_summary_vec:
            summary_vec = c[:, 0, :]
            summary_vec = self.summary_proj(summary_vec)
            c = c[:, 1:, :]
            noncausal_masks = noncausal_masks[:, 1:]
        else: summary_vec = None
        c = self.final_proj(c)


        if feats_only: 
            c_feat = conv_feats.detach() # (bs, seq_len, dim)
            s_feat = summary_vec.detach() # (bs, dim)
            o_feat = c.detach() # (bs, seq_len, dim)
            return c_feat, s_feat, o_feat
        else: c_feat, s_feat, o_feat = conv_feats, summary_vec, c

        # Return 3 tuples: 1st is main outputs from network,
        # 2nd is features to train kmean model, 3rd is similarities to codebook embeddings
        c_sims = [head(c, summary_vec) for head in self.c_heads]
        s_sims = [head(summary_vec, None) for head in self.s_heads]
        o_sims = [head(c, summary_vec) for head in self.o_heads]

        return (c, summary_vec, c_lengths, masks), (c_feat, s_feat, o_feat), (c_sims, s_sims, o_sims)
        
    @torch.no_grad()
    def get_features(self, wv1, wv1_lengths, layer=None):
        c = self.conv_extractor1(wv1)
        c = c.permute(0, 2, 1) # (bs, c_dim, T) --> (bs, T, c_dim)
        # given the stride settings at the top, this is the factor.
        c_lengths = (1 + (wv1_lengths - self.cfg.conv_magic_offset)/self.cfg.conv_reduction_mult).floor().long() 

        c = self.post_extract_proj(c) # (..., dim) --> (..., transformer_dim)
        c = self.layernorm(c)
        
        noncausal_masks = ~get_mask_from_lengths(c_lengths) # must invert to work with pytorch attention implementation

        # Path 1: no mask
        if layer is None: layer = len(self.transformer_layers)

        c = c.clone()
        c = self.positional_embedding(c)
        if self.cfg.use_summary_vec:
            c, noncausal_masks = self.summary_embedder(c, noncausal_masks)
        for mod in self.transformer_layers[:layer]:
            c = mod(c, src_key_padding_mask=noncausal_masks)

        if self.cfg.use_summary_vec:
            summary_vec = c[:, 0, :]
            if layer == len(self.transformer_layers): summary_vec = self.summary_proj(summary_vec)
            c = c[:, 1:, :]
            noncausal_masks = noncausal_masks[:, 1:]
        else: summary_vec = None
        if layer == len(self.transformer_layers): c = self.final_proj(c)

        if summary_vec is not None:
            seq_len = c.shape[1]
            s = summary_vec.unsqueeze(1).repeat(1, seq_len, 1) # (bs, seq_len, dim)
            v = torch.cat((s, c), dim=-1)
        else: v = c

        return c, v, summary_vec

    def remove_weight_norm(self):
        print('[DBERT] Removing weight norm...')
        torch.nn.utils.remove_weight_norm(self.positional_embedding.pos_conv[0])


