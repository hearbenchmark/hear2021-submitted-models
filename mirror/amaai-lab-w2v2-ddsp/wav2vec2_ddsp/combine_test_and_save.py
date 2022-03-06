from typing import Tuple
import torch
import fairseq
from torch import Tensor
# from ddsp import DDSP
from ddsp import DDSP,extract_loudness_torch, extract_pitch_torch
import torchcrepe
import os

# class CombineModel(torch.nn.Module):
#     def __init__(self, modelA, modelB, scene_embedding_size = None):
#         super(CombineModel, self).__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.scene_embedding_size = scene_embedding_size
       
#     def forward(self, x):
#         x1 = self.modelA(x)
#         x2 = self.modelB(x)
#         trimmed_len = torch.min(torch.tensor([x1, x2]))
#         x1 = x1[:, :trimmed_len, :]
#         x2 = x2[:, :trimmed_len, :]
#         x = torch.cat((x1, x2), dim=-1)
#         return x

class CombineModel(torch.nn.Module):
    def __init__(self, modelA, modelB, scene_embedding_size = None):
        super(CombineModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.scene_embedding_size = scene_embedding_size
       
    def forward(self, x, pitch, loudness):
        x1 = self.modelA(x, mask=False, features_only=True)['x']
        _, amplitude_return, total_amp_return, pitch_return, harmonic_return = self.modelB(pitch, loudness)
        trimmed_len = torch.min(torch.tensor([x1.size(dim=1), amplitude_return[1].size(dim=1)]))
        x1 = x1[:, :trimmed_len, :]
        amplitude_return = amplitude_return[:, :trimmed_len, :]
        total_amp_return = total_amp_return[:, :trimmed_len, :]
        pitch_return = pitch_return[:, :trimmed_len, :]
        harmonic_return = harmonic_return[:, :trimmed_len, :]

        x = torch.cat((x1, amplitude_return, total_amp_return,pitch_return, harmonic_return ), dim=-1)
        return x

def load_model_Save_weig(
    model_file_path: str = "", model_hub: str = "facebook/wav2vec2-large-100k-voxpopuli"
) -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
        model_hub: Which wav2vec2 model to load from hugging face.
    Returns:
        Model
    """
    # model_fairseq = FairseqWav2Vec2(model_url, save_path="pretrained/local_model.pt")
    wav2vec, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(["checkpoint_last.pt"])
    model1 = wav2vec[0]

    model2 =DDSP(hidden_size = 512, n_harmonic=100, n_bands=65, sampling_rate=16000,
                 block_size=320, n_fft=2048)
    ckpt = torch.load("state.pth")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model2.load_state_dict(new_state_dict)

    model = CombineModel(model1, model2)
    if torch.cuda.is_available():
        model.cuda()
    # sample rate and embedding sizes are required model attributes for the HEAR API
    model.sample_rate = 16000
    model.scene_embedding_size = 768+103
    # model1.timestamp_embedding_size = 768
    # model1.scene_embedding_size = ?
    # model1.timestamp_embe
    # dding_size = ?
    path = "combined_weight.pt"
    torch.save(model.state_dict(), path)
    return model


# def load_model(
#     model_file_path: str = "", model_hub: str = "facebook/wav2vec2-large-100k-voxpopuli"
# ) -> torch.nn.Module:
#     """
#     Returns a torch.nn.Module that produces embeddings for audio.

#     Args:
#         model_file_path: Ignored.
#         model_hub: Which wav2vec2 model to load from hugging face.
#     Returns:
#         Model
#     """
#     # model_fairseq = FairseqWav2Vec2(model_url, save_path="pretrained/local_model.pt")
#     wav2vec, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file_path])
#     model1 = wav2vec[0]

#     model2 =DDSP(hidden_size = 512, n_harmonic=100, n_bands=65, sampling_rate=16000,
#                  block_size=320, n_fft=2048)

#     model = CombineModel(model1, model2)
#     model.load_state_dict(torch.load("combined_weight.pt"))

#     if torch.cuda.is_available():
#         model.cuda()
#     # sample rate and embedding sizes are required model attributes for the HEAR API
#     model.sample_rate = 16000
#     model.scene_embedding_size = 768+103
#     return model



class CombineModel(torch.nn.Module):
    def __init__(self, modelA, modelB, scene_embedding_size = None):
        super(CombineModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.scene_embedding_size = scene_embedding_size
       
    def forward(self, x, pitch, loudness):
        x1 = self.modelA(x, mask=False, features_only=True)['x']
        _, amplitude_return, total_amp_return, pitch_return, harmonic_return = self.modelB(pitch, loudness)
        trimmed_len = torch.min(torch.tensor([x1.size(dim=1), amplitude_return[1].size(dim=1)]))
        x1 = x1[:, :trimmed_len, :]
        amplitude_return = amplitude_return[:, :trimmed_len, :]
        total_amp_return = total_amp_return[:, :trimmed_len, :]
        pitch_return = pitch_return[:, :trimmed_len, :]
        harmonic_return = harmonic_return[:, :trimmed_len, :]

        x = torch.cat((x1, amplitude_return, total_amp_return,pitch_return, harmonic_return ), dim=-1)
        return x
 

def load_model(
    model_file_path: str = "", model_hub: str = "facebook/wav2vec2-large-100k-voxpopuli"
) -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
        model_hub: Which wav2vec2 model to load from hugging face.
    Returns:
        Model
    """
    # model_fairseq = FairseqWav2Vec2(model_url, save_path="pretrained/local_model.pt")
    # wav2vec, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file_path])
    # model1 = wav2vec[0]

    from omegaconf import OmegaConf
    cfg_model = {'_name': 'wav2vec2', 'extractor_mode': 'default', 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'activation_fn': 'gelu', 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.05, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 256, 'layer_norm_first': False, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_bias': False, 'logit_temp': 0.1, 'quantize_targets': True, 'quantize_input': False, 'same_quantizer': False, 'target_glu': False, 'feature_grad_mult': 0.1, 'quantizer_depth': 1, 'quantizer_factor': 3, 'latent_vars': 320, 'latent_groups': 2, 'latent_dim': 0, 'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_before': False, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'num_negatives': 100, 'negatives_from_everywhere': False, 'cross_sample_negatives': 0, 'codebook_negatives': 0, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995]}
    cfg_task ={'_name': 'audio_pretraining', 'data': '/workspace/public_data/raven/wav2vec_AMT/Data_tsvs/Speech_Music', 'labels': None, 'binarized_dataset': False, 'sample_rate': 16000, 'normalize': False, 'enable_padding': False, 'max_sample_size': 250000, 'min_sample_size': 32000, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'inferred_w2v_config': None, 'tpu': False, 'text_compression_level': 'none'}
    task = fairseq.tasks.setup_task(OmegaConf.create(cfg_task))
    model1 = task.build_model(OmegaConf.create(cfg_model))

    model2 =DDSP(hidden_size = 512, n_harmonic=100, n_bands=65, sampling_rate=16000,
                 block_size=320, n_fft=2048)

    model = CombineModel(model1, model2)
    model.load_state_dict(torch.load("combined_weight.pt"))
    # PATH = ""
    # if not os.path.isfile(PATH):
    #     #download
    #     print("finished downloading")
    # model2.load_state_dict(torch.load(PATH))
    if torch.cuda.is_available():
        model.cuda()
    # sample rate and embedding sizes are required model attributes for the HEAR API
    model.sample_rate = 16000
    model.scene_embedding_size = 768+103
    # model1.timestamp_embedding_size = 768
    # model1.scene_embedding_size = ?
    # model1.timestamp_embedding_size = ?


    return model


def get_timestamp_embeddings(
    audio,model
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

    # Send the model to the same device that the audio tensor is on.
    # model = model.to(audio.device)

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    loudness = extract_loudness_torch(audio, 16000, 320)
    mean_l = torch.mean(loudness, axis=-1).unsqueeze(-1) #b, 1
    std_l = torch.std(loudness, axis=-1).unsqueeze(-1) #b, 1s
    loudness = (loudness-mean_l)/std_l #b, len_l

    pitch= extract_pitch_torch(audio, 16000, 320) #device
    pitch = pitch.unsqueeze(-1)
    loudness = loudness.unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        embeddings = model(audio, pitch, loudness) # (batch, timesteps, 768)        

    # Length of the audio in MS
    audio_ms = int(audio.shape[1] / model.sample_rate * 1000)

    # It's a bit hard to determine the timestamps, but here is a
    # decent shot at it.
    # See also: https://github.com/speechbrain/speechbrain/issues/966#issuecomment-914492048 # noqa
    # https://github.com/speechbrain/speechbrain/blob/98f90f82acc327a8180f3591135a18e278d3e0e2/speechbrain/alignment/ctc_segmentation.py#L413-L419 # noqa

    # samples => timestamps
    # 31439 => 97
    # 31440 => 98
    # This is weird that its 5ms, not half the hopsize of 20
    ntimestamps = (audio_ms - 5) // 20

    # Also
    # 32000 => 99
    # 32080 => 100

    # I don't know if this is their exact centering, but this matches
    # their shape.
    last_center = 12.5 + (ntimestamps - 1) * 20
    timestamps = torch.arange(12.5, last_center + 20, 20)
    assert len(timestamps) == ntimestamps
    timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))
    assert timestamps.shape[1] == embeddings.shape[1]

    return embeddings, timestamps


# TODO: There must be a better way to do scene embeddings,
# e.g. just truncating / padding the audio to 2 seconds
# and concatenating a subset of the embeddings.
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
    embeddings, _ = get_timestamp_embeddings(audio, model)
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings


# ckpt1 = "checkpoint_last.pt"
wav2vec, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(["checkpoint_last.pt"])
model1 = wav2vec[0]


from omegaconf import OmegaConf
cfg_model = {'_name': 'wav2vec2', 'extractor_mode': 'default', 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'activation_fn': 'gelu', 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.05, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 256, 'layer_norm_first': False, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_bias': False, 'logit_temp': 0.1, 'quantize_targets': True, 'quantize_input': False, 'same_quantizer': False, 'target_glu': False, 'feature_grad_mult': 0.1, 'quantizer_depth': 1, 'quantizer_factor': 3, 'latent_vars': 320, 'latent_groups': 2, 'latent_dim': 0, 'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_before': False, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'num_negatives': 100, 'negatives_from_everywhere': False, 'cross_sample_negatives': 0, 'codebook_negatives': 0, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995]}
cfg_task ={'_name': 'audio_pretraining', 'data': '/workspace/public_data/raven/wav2vec_AMT/Data_tsvs/Speech_Music', 'labels': None, 'binarized_dataset': False, 'sample_rate': 16000, 'normalize': False, 'enable_padding': False, 'max_sample_size': 250000, 'min_sample_size': 32000, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'inferred_w2v_config': None, 'tpu': False, 'text_compression_level': 'none'}
task = fairseq.tasks.setup_task(OmegaConf.create(cfg_task))
model2 = task.build_model(OmegaConf.create(cfg_model))
model2.load_state_dict(torch.load("checkpoint_last.pt")['model'],strict=True , model_cfg=cfg_model)

# model2.load_
print("model1, model2", type(model1),type(model2))
print("mod1", model1)
print("mod2", model2)
# audio = torch.rand((2, 16000)).cuda()
# f0 = torchcrepe.predict(
#                     audio,
#                     16000,
#                     # int(1000 * block_size / sampling_rate),
#                     int(160),
#                     fmin = 0,
#                     fmax= 2006,
#                     model = 'full',
#                     batch_size=2048,
#                     decoder = torchcrepe.decode.viterbi,
#                     device="cuda:1")
# out = model1(audio, mask=False, features_only=True)
# print(out['x'].shape)
# model2 =DDSP(hidden_size = 512, n_harmonic=100, n_bands=65, sampling_rate=16000,
#                 block_size=320, n_fft=2048)


# pitch = torch.rand((2, 50, 1))
# loudness =  torch.rand((2, 50,1))
# _,amplitude_return, total_amp_return, pitch_return, harmonic_return =  model2(pitch, loudness)
# print(amplitude_return.shape, total_amp_return.shape, pitch_return.shape, harmonic_return.shape)


# model = load_model(model_file_path = "checkpoint_last.pt")
# out, _ = get_timestamp_embeddings(audio, model)
# print(out.shape)

# wav2vec, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(["checkpoint_last.pt"])
# model1 = wav2vec[0]
# print(wav2vec, cfg, task)
#


# # print("see waht's saved", torch.load("checkpoint_last.pt")['model'])
# from fairseq.tasks.audio_pretraining import AudioPretrainingTask
# model = AudioPretrainingTask.build_model(model_cfg= cfg)
# print(model)
# load_model_Save_weig()
