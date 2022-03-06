from udons.models.jigsaw_transformer_model import JigsawTransformerModel
from udons.hearutils import frame_audio
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

import torch
import torchaudio


# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 32  # = patch size

class JigSawEmbedder(torch.nn.Module):
    """
    JigSaw Embedder class
    """
    def __init__(self, model: torch.nn.Module):
        """
        Initialize JigSaw Embedder
        """
        super().__init__()
        self.sample_rate = 16000
        self.pos_embed = True
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=100,
            f_min=27.5,
            f_max=16000,
            n_mels=128,
        )
        
        modules = [Rearrange('b c f t -> b (c f t)')]
        modules.extend(list(model.patch_encoding.children())[1:])
        # self.model = torch.nn.Sequential(*modules)
        self.model = model

    def forward(self, x):
        """
        Forward pass
        """

        x = self.transform(x)
        # get do not apply siamese view for this model
        # this assumes siamesed audio and applies Rearrange('(b p) c f t -> b p (c f t)
        x = self.model.patch_encoding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.model.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed:
            x += self.model.pos_embedding[:, :(n + 1)]
        x = self.model.dropout(x)

        x = self.model.transformer(x)
        return x[:, 1:] # return all tokens except CLS token

def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.
    Args:
        model_file_path: Load model checkpoint from this file path. For this baseline,
            if no path is provided then the default random init weights for the
            linear projection layer will be used.
    Returns:
        Model
    """
    lightning_model = JigsawTransformerModel.load_from_checkpoint(checkpoint_path=model_file_path)
    model = JigSawEmbedder(lightning_model.model)
    model.eval()
    model.sample_rate = 16000  # sample rate
    model.embedding_size = lightning_model.hparams["model_dim"]  # model_dim  TODO: get from configs
    model.scene_embedding_size = model.embedding_size
    model.timestamp_embedding_size = model.embedding_size
    return model


def get_timestamp_embeddings(
    audio: torch.Tensor,
    model: torch.nn.Module,
):
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

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed.

    # in fact we are using this to create already the patches
    frames, timestamps = frame_audio(
        audio,
        frame_size=1599,
        hop_size=50,
        sample_rate=16000,
    )
    patch_len = 5
    audio_batches, num_frames, frame_size = frames.shape

    frames = rearrange(frames, 'b p t -> (b p) t')

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(frames)
    # trick here: create patches with batch_size = 3
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE * patch_len, shuffle=False, drop_last=False
    )


    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    embeddings_list = []
    remainder = 0
    with torch.no_grad():
        for batch in loader:
            model_input = batch[0][:, None]
            if model_input.shape[0] != BATCH_SIZE * patch_len:
                remainder = BATCH_SIZE * patch_len - model_input.shape[0]
                model_input = torch.cat((model_input, torch.zeros(remainder, 1, model_input.shape[2]).to(audio.device)), dim=0)
            out = model(model_input)
            embeddings_list.append(out)

    # Concatenate mini-batches back together and unflatten the frames
    # to reconstruct the audio batches
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = rearrange(embeddings, 'b p f -> (b p) f')
    embeddings = embeddings[:-remainder].unflatten(0, (audio_batches, num_frames))

    return embeddings, timestamps


def get_scene_embeddings(
    audio: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
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
    embeddings, _ = get_timestamp_embeddings(audio, model)
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="model directory")
    args = parser.parse_args()
    model = load_model(args.model_path)
    x, _ = get_timestamp_embeddings(torch.rand(21, 4041000), model)
    print(x.shape)
