"""EfficientNet."""
import logging
import torch
import torch.nn as nn
import torchaudio.transforms as T
from efficientnet_pytorch import EfficientNet


class EfficientNet_b0(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        n_embedding=512,
        n_aug=None,
    ):

        super(self.__class__, self).__init__()
        self.spectrogram_extracter = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            power=2.0,
            n_mels=n_mels,
        )
        self.efficientnet = EfficientNet.from_name(
            model_name="efficientnet-b0", in_channels=1
        )
        self.fc1 = nn.Linear(1280, n_embedding, bias=True)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=n_embedding)
        self.fc2 = torch.nn.Linear(n_embedding, n_embedding, bias=False)
        self.sample_rate = sample_rate
        self.scene_embedding_size = n_embedding
        self.timestamp_embedding_size = n_embedding
        self.n_timestamp = None
        self.n_aug = n_aug
        if n_aug is not None:
            self.aug_fc = nn.Linear(n_embedding, n_aug, bias=True)

    def forward(self, X):
        """X: (batch_size, T', mels)"""
        # logging.info(f"X:{X.shape}")
        if len(X.shape) == 2:
            # X: (batch_size, wave_length)->(batch_size, T', mels)
            X = self.spectrogram_extracter(X).transpose(1, 2)
        x = X.unsqueeze(1)  # (B, 1, T', mels)
        x = self.efficientnet.extract_features(x)
        # logging.info(f"x:{x.shape}")
        x = x.max(dim=3)[0]
        # logging.info(f"x:{x.shape}")
        embedding_h = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        self.n_timestamp = embedding_h.shape[2]
        # logging.info(f"embedding_h: {embedding_h.shape}")
        embed = torch.tanh(self.layer_norm(embedding_h.max(dim=2)[0]))
        embedding_z = self.fc2(embed)
        output_dict = {
            # (B, T', timestamp_embedding_size)
            "framewise_embedding": embedding_h.transpose(1, 2),
            # (B, scene_embedding_size)
            "clipwise_embedding": embedding_h.max(dim=2)[0],
            "embedding_z": embedding_z,  # (B, n_embedding)
        }
        if self.n_aug is not None:
            output_dict["aug_output"] = self.aug_fc(embed)

        return output_dict
