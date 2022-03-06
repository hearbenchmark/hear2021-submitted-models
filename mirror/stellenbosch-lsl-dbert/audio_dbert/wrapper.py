from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor
from torch.functional import Tensor
from torch.nn import functional as F

from .model import DBERT2
import omegaconf

# Note to self: on my machine, 
# generating HEAR embeddings takes about 15min
# training the downstream models and evaluating them
# for all the PUBLIC tasks takes 3h40m

def load_model(model_file_path: str, layer=16) -> nn.Module:
    if layer == -1: layer = None
    ckpt = torch.load(model_file_path, map_location='cpu')
    model = DBERTWrapped(ckpt, layer=layer)
    model.DBERT.remove_weight_norm()
    model.eval()
    return model

class DBERTWrapped(nn.Module):

    def __init__(self, ckpt, layer):
        super().__init__()
        self.layer = layer # which transformer layer to get embeddings from.
        print("Using layer ", self.layer)
        self.cfg = omegaconf.OmegaConf.create(ckpt['cfg_yaml_string'])
        if hasattr(self.cfg.model, 'use_db2') and self.cfg.model.use_db2:
            self.DBERT = DBERT2(self.cfg.model)
        else:
            raise NotImplementedError()
        self.DBERT.load_state_dict(ckpt['model_state_dict'])

        self.sample_rate = self.cfg.data.sample_rate
        if self.layer == None: 
            base_dim = self.cfg.model.dim
            summary_dim = self.cfg.model.summary_dim
        else: 
            base_dim = self.cfg.model.transformer_dim
            summary_dim = self.cfg.model.transformer_dim

        if self.cfg.model.use_summary_vec:
            self.scene_embedding_size = base_dim + summary_dim
            self.timestamp_embedding_size = base_dim + summary_dim
        else:
            self.scene_embedding_size = base_dim
            self.timestamp_embedding_size = base_dim

def predict(audio: Tensor, model: DBERTWrapped, fp16=False) -> Tuple[Tensor]:
    """ `audio` is (bs, seq_len) in range [-1, 1]. `m_chunk` is number of minutes to chunk audio into """

    lengths = torch.tensor([a.shape[0] for a in audio], dtype=torch.long, device=audio.device)
    model.eval()
    if fp16 and 'cuda' in str(audio.device):
        with torch.cuda.amp.autocast():
            c1, v1, summaries = model.DBERT.get_features(audio, lengths, layer=model.layer)
        c1 = c1.float()
        v1 = v1.float()
        if summaries is not None: summaries = summaries.float()
    else:
        c1, v1, summaries = model.DBERT.get_features(audio, lengths, layer=model.layer)
    # c1 and c1_proj are (bs, seq_len, dim), summary vec is (bs, dim)
    if summaries is not None:
        seq_len = c1.shape[1]
        s = summaries.unsqueeze(1).repeat(1, seq_len, 1) # (bs, seq_len, dim)
        features = torch.cat((s, c1), dim=-1)
    else: features = c1

    return c1, v1, summaries, features

def samples2seqlen(samples, sr):
    # ------------------------------------------------
    # timestamp approximations are shamelessly taken
    # from the HEAR baseline. THANK YOU HEAR TEAM!
    # ------------------------------------------------
    audio_ms = int(samples / sr * 1000)
    # samples => timestamps
    # 31439 => 97 ; 31440 => 98
    # This is weird that its 5ms, not half the hopsize of 20ms
    ntimestamps = (audio_ms - 5) // 20
    # Also 32000 => 99 ; 32080 => 100
    # I don't know if this is their exact centering, but this matches
    # their shape.
    last_center = 12.5 + (ntimestamps - 1) * 20
    timestamps = torch.arange(12.5, last_center + 20, 20)
    assert len(timestamps) == ntimestamps
    return timestamps, audio_ms

def get_timestamp_embeddings(audio: Tensor, model: DBERTWrapped, m_chunk=3, fp16=False) -> Tuple[Tensor, Tensor]:
    """ `audio` is (bs, seq_len) in range [-1, 1] """
    if audio.shape[-1] > model.sample_rate*60*m_chunk:
        print(f"Warning: audio longer than {m_chunk} minutes found. Doing inference in chunks.")
        n_chunks = 1 + (audio.shape[-1] // (model.sample_rate*60*m_chunk))
             
        chunked_audio = audio.chunk(n_chunks, dim=-1)
        chunked_features = []
        chunked_timestaps = []
        rolling_dur_ms = 0.0
        for chunk in chunked_audio:
            c1, v1, summaries, features = predict(chunk, model, fp16=fp16)
            if fp16 and 'cuda' in str(audio.device): torch.cuda.empty_cache()
            chunked_features.append(features)
            timestamps, duration_ms = samples2seqlen(chunk.shape[-1], model.sample_rate)
            timestamps = timestamps + rolling_dur_ms
            chunked_timestaps.append(timestamps)
            rolling_dur_ms += duration_ms
        features = torch.cat(chunked_features, dim=1)
        timestamps = torch.cat(chunked_timestaps, dim=0)
    else:
        c1, v1, summaries, features = predict(audio, model, fp16=fp16)
        timestamps, _ = samples2seqlen(audio.shape[1], model.sample_rate)

    timestamps = timestamps.expand((features.shape[0], timestamps.shape[0]))
    assert timestamps.shape[1] == features.shape[1]
    return features, timestamps

def get_scene_embeddings(audio: Tensor, model: DBERTWrapped, m_chunk=3, fp16=False) -> Tensor:
    if audio.shape[-1] > model.sample_rate*60*m_chunk:
        print(f"Warning: audio longer than {m_chunk} minutes found. Doing inference in chunks.")
        n_chunks = 1 + (audio.shape[-1] // (model.sample_rate*60*m_chunk))
             
        chunked_audio = audio.chunk(n_chunks, dim=-1)
        chunked_features = []
        for chunk in chunked_audio:
            c1, v1, summaries, features = predict(chunk, model, fp16=fp16)
            if fp16 and 'cuda' in str(audio.device): torch.cuda.empty_cache()
            chunked_features.append(features)
        features = torch.cat(chunked_features, dim=1)
    else:
        c1, v1, summaries, features = predict(audio, model, fp16=fp16)

    # if there is a summary vector, it will be the first summary_dim
    # dimensions of the mean, since we just taking the mean of the same
    # vector in those positions.
    return features.mean(dim=1)

def verify_mem_usage(model):
    """ Tests that model can parse 20min of audio in under 16GB of GPU memory """
    model = model.to('cuda')
    x1 = torch.randn(1, int(model.sample_rate*60*4.9)).to('cuda')
    get_timestamp_embeddings(x1, model)
    get_scene_embeddings(x1, model)
    x2 = torch.randn(1, model.sample_rate*60*20).to('cuda')
    get_timestamp_embeddings(x2, model)
    get_scene_embeddings(x2, model)
