from json import decoder
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa as li
# import crepe
import math
# from .crepepytorch import CREPE
import torchaudio
import torchcrepe

def safe_log(x):
    return torch.log(x + 1e-7)


@torch.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for x in dataset:
        l = x["loudness"]
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def scale_function(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    # signal = signal.cpu().detach().numpy()
    
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )

    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sampling_rate, n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    S = torch.tensor(S)
    return S


def extract_pitch(signal, sampling_rate, block_size):
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0 = f0[1].reshape(-1)[:-1]

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0


def extract_loudness_torch(signal,sr, block_size, n_fft=2048, device = "cuda:0"):
    speclayer =torchaudio.transforms.Spectrogram( n_fft = n_fft, hop_length = block_size, win_length = n_fft,center = True, power=1).to(device)
    S = speclayer(signal.to(device)) #(B, F, T)
    # print("asd",signal.shape, S.shape)

    S = 20* torch.log(torch.abs(S)+ 1e-7)

    f_torch = torch.linspace(0, 0.5*sr, steps=int(n_fft/2)+1).to(device)
    # f_torch = torch.linspace(0, 0.5*sr, steps=int(n_fft/2)+1)

    a_weight = torch_A_weighting(f_torch, device = device)

    # S = S.to(self.device) + a_weight.reshape(1, -1, 1) #(F,) ==> (1, F, 1) broadcast along batch and dim
    S = S + a_weight.reshape(1, -1, 1) #(F,) ==> (1, F, 1) broadcast along batch and dim

    S = torch.mean(S, 1)[..., :-1] #(B, T)

    return S


def torch_A_weighting(FREQUENCIES, min_db = -80.0, device = "cuda:0"):
    """
    Compute A-weighting weights in Decibel scale (codes from librosa) and 
    transform into amplitude domain (with DB-SPL equation).
    
    Argument: 
        FREQUENCIES : tensor of frequencies to return amplitude weight
        min_db : mininum decibel weight. appropriate min_db value is important, as 
            exp/log calculation might raise numeric error with float32 type. 
    
    Returns:
        weights : tensor of amplitude attenuation weights corresponding to the FREQUENCIES tensor.
    """
    
    # Calculate A-weighting in Decibel scale.
    FREQUENCY_SQUARED = FREQUENCIES ** 2 
    const = torch.tensor([12200, 20.6, 107.7, 737.9]) ** 2.0
    WEIGHTS_IN_DB = 2.0 + 20.0 * (torch.log10(const[0]) + 4 * torch.log10(FREQUENCIES)
                            - torch.log10(FREQUENCY_SQUARED + const[0])
                            - torch.log10(FREQUENCY_SQUARED + const[1])
                            - 0.5 * torch.log10(FREQUENCY_SQUARED + const[2])
                            - 0.5 * torch.log10(FREQUENCY_SQUARED + const[3]))
    # Set minimum Decibel weight.
    if min_db is not None:
        WEIGHTS_IN_DB = torch.max(WEIGHTS_IN_DB.to(device), torch.tensor([min_db], dtype = torch.float32).to(device))
    
    # Transform Decibel scale weight to amplitude scale weight.
    weights = torch.exp(torch.log(torch.tensor([10.], dtype = torch.float32).to(device)) * WEIGHTS_IN_DB / 10) 

    # Transform Decibel scale weight to amplitude scale weight.
    # weights = torch.exp(torch.log(torch.tensor([10.], dtype = torch.float32).to(self.device)) * WEIGHTS_IN_DB / 10) 
    
    return WEIGHTS_IN_DB


def extract_pitch_torch(signal, sr, block_size, device = "cuda:0"):

    # net = CREPE().to(device)
    # time, f0, confidence, activation = net.predict(
    #     signal.to(device),
    #     sr=sr,
    #     viterbi=True,
    #     step_size=int(1000 * block_size /sr), #int(self.config.frame_resolution * 1000),
    #     batch_size=32,
    # )
    # f0 = f0[:, :-1]
    # print("signal dafuq", signal.shape, sr, block_size)

    f0 = torchcrepe.predict(
                        signal,
                        sr,
                        # int(1000 * block_size / sampling_rate),
                        int(sr*block_size / sr),
                        fmin = 0,
                        fmax= 2006,
                        model = 'full',
                        batch_size=2048,
                        decoder = torchcrepe.decode.viterbi,
                        device=device)

    f0 = f0[:, :-1]
    # print("raw input length", f0,"step size", int(1000 * block_size / sampling_rate))

    # if f0.shape[-1] != length:
    #     f0 = np.interp(
    #         np.linspace(0, 1, length, endpoint=False),
    #         np.linspace(0, 1, f0.shape[-1], endpoint=False),
    #         f0,
    #     )
    return f0


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(pitch, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output