"""
Baseline model for HEAR 2021 NeurIPS competition.

This is simply a mel spectrogram followed by random projection.
"""


from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from hearline.models import Conformer
from hearline.models import EfficientNet_b0

MODEL_NAME = "Conformer"
BATCH_SIZE = 64


def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for a frame of audio.

    Args:
        model_file_path: Load model checkpoint from this file path. For this baseline,
            if no path is provided then the default random init weights for the
            linear projection layer will be used.
    Returns:
        Model: torch.nn.Module loaded on the specified device.
    """
    if MODEL_NAME == "EfficientNet_b0":
        model = EfficientNet_b0()
    elif MODEL_NAME == "Conformer":
        model = Conformer()
    model.load_state_dict(torch.load(model_file_path)["model"])
    return model


def inference(audio, model, key="clipwise_output"):
    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(audio)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    model.eval()
    embeddings_list = []
    with torch.no_grad():
        for batch in loader:
            # print(len(batch), batch, batch[0].shape)
            embeddings_list.append(model(batch[0])[key])
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings


def get_timestamp(x, sample_rate, n_frame):
    t_end = x.size(1) / sample_rate * 1000  # mili sec
    timestamp = torch.from_numpy(
        np.linspace(0, t_end, n_frame).astype(np.float32)
    ).clone()
    timestamp = timestamp.expand(x.shape[0], -1)
    return timestamp


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1].
        model: Loaded model.

    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )
    model = model.to(audio.device)
    embeddings = inference(audio, model, key="framewise_embedding")
    timestamps = get_timestamp(
        audio, sample_rate=model.sample_rate, n_frame=model.n_timestamp
    )
    return embeddings, timestamps


def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using torch.mean().

    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.

    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )
    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)
    embeddings = inference(audio, model, key="clipwise_embedding")
    return embeddings
