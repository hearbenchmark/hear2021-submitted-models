import librosa
import numpy as np
import torch
import wav2clip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_file_path, frame_length=16000):
    model = wav2clip.get_model(device=device, pretrained=True)
    model.sample_rate = 16000
    model.scene_embedding_size = 512
    model.timestamp_embedding_size = 512
    if frame_length:
        model.frame_length = frame_length
        model.hop_length = int(model.sample_rate * 0.05)
    return model


def get_timestamp_embeddings(
    audio,
    model,
):
    assert len(audio.shape) == 2

    embeddings = wav2clip.embed_audio(audio.to("cpu").numpy(), model)
    embeddings = np.swapaxes(embeddings, 1, 2)

    timestamp = librosa.frames_to_time(
        range(embeddings[0].shape[0]), sr=model.sample_rate, hop_length=model.hop_length
    )
    timestamps = np.array([timestamp] * embeddings.shape[0])

    assert len(timestamps.shape) == 2
    assert embeddings.shape[0] == audio.shape[0]
    assert len(embeddings.shape) == 3
    assert timestamps.shape[0] == audio.shape[0]
    assert embeddings.shape[1] == timestamps.shape[1]
    assert embeddings.shape[2] == model.timestamp_embedding_size
    return torch.from_numpy(embeddings).to(device), torch.from_numpy(timestamps).to(
        device
    )


def get_scene_embeddings(
    audio,
    model,
):
    model.frame_length = None
    model.hop_length = None
    embeddings = wav2clip.embed_audio(audio.to("cpu").numpy(), model)

    assert len(embeddings.shape) == 2
    assert embeddings.shape[0] == audio.shape[0]
    assert embeddings.shape[1] == model.scene_embedding_size
    return torch.from_numpy(embeddings).to(device)
