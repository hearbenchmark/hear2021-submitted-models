import torch
from torch import nn
from torch.nn import functional as F
from .audio_mlp import KW_MLP, gMLP_Encoder
from nnAudio.Spectrogram import MelSpectrogram, MFCC
from einops import rearrange


class AudioMLP_Wrapper(nn.Module):
    """Wrapper for Audio MLP Models."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        timestamp_embedding_size: int = 64,
        scene_embedding_size: int = 1024,
        encoder_type: str = "kwmlp",
        encoder_ckpt: str = ""
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.timestamp_embedding_size = timestamp_embedding_size
        self.scene_embedding_size = scene_embedding_size

        assert encoder_type in ["kwmlp", "audiomae"], "Unsupported model."

        if encoder_type == "kwmlp":
            self.encoder = KW_MLP()
            if encoder_ckpt != "":
                ckpt = torch.load(encoder_ckpt, map_location="cpu")["model_state_dict"]
                self.encoder.load_state_dict(ckpt)
            self.audio_processor = MFCC(
                n_mfcc=40,
                sr=sample_rate,
                n_mels=40,
                n_fft=480,
                win_length=480,
                hop_length=160,
                center=False
            )
            self.encoder.to_logits = nn.LayerNorm(
                timestamp_embedding_size
            )

        elif encoder_type == "audiomae":
            self.encoder = gMLP_Encoder(
                embed_dim=timestamp_embedding_size
            )
            if encoder_ckpt != "":
                ckpt = torch.load(encoder_ckpt, map_location="cpu")["model_state_dict"]
                self.encoder.load_state_dict(ckpt)
            self.audio_processor = MelSpectrogram(
                sr=sample_rate,
                n_mels=40,
                n_fft=480,
                win_length=480,
                hop_length=160,
                center=False
            )

    def forward(self, x: torch.Tensor):
        b, num_samples = x.shape
        
        # model input must be a multiple of sr
        if num_samples < self.sample_rate:
            x = F.pad(x, (0, self.sample_rate - num_samples), "constant", 0)
        elif num_samples %  self.sample_rate:
            x = F.pad(x, (0, self.sample_rate - num_samples %  self.sample_rate), "constant", 0)
        
        x = rearrange(x, "b (t sr) -> (b t) sr", sr=self.sample_rate)
        x = self.audio_processor(x)
        x = self.encoder(x)

        # compensate for chunking
        x = rearrange(x, "b d f -> b f d") # replicate cannot pad arbitrary axis
        x = F.pad(x, (1, 1), mode="replicate")
        x = rearrange(x, "(b t) f d -> b (t d) f", b=b)
        x = x[:, 1:-1, :]            
        return x



