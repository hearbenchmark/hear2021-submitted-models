from torch import Tensor
from torch.nn import functional as F


def initial_padding(audio: Tensor, sr=16000, hop_ms=10, window_ms=30) -> Tensor:
    """Do some initial padding in order to get embeddings at the start/end of audio.

    Args:
        audio (Tensor): n_sounds x n_samples of mono audio.
        sr (int, optional): Sample rate. Defaults to 16000.
        hop_ms (int, optional): Hop length in ms. Defaults to 10.
        window_ms (int, optional): Window length in ms. Defaults to 30.

    Returns:
        Tensor: n_sounds x n_samples_padded.
    """
    init_pad = int((window_ms // 2 - hop_ms) / 1000 * sr) if window_ms // 2 > hop_ms else 0
    end_pad = int((window_ms // 2 ) / 1000 * sr)
    return F.pad(audio, (init_pad, end_pad), "constant", 0)