import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
# from core import mlp, gru, scale_function, remove_above_nyquist, upsample
# from core import harmonic_synth, amp_to_impulse_response, fft_convolve
# from core import resample
import math
# import torchcrepe
import librosa as li
import numpy as np
# import glob
import torchaudio
# import nnAudio
# from nnAudio import Spectrogram
# from .crepepytorch import CREPE
class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size, n_fft=2048 , device = 'cuda:0'):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))


        #sam
        self.sr = torch.tensor(sampling_rate)
        self.block_size =torch.tensor(block_size)
        self.device = device
        self.n_fft = n_fft
        # self.spec_layer = Spectrogram.STFT(n_fft=2048, freq_bins=None, hop_length=self.block_size,
        #                             window='hann', freq_scale='linear', center=True, pad_mode='reflect',
        #                             fmin=50,fmax=6000, sr=sampling_rate) # Initializing the model
        self.speclayer =torchaudio.transforms.Spectrogram( n_fft = n_fft, hop_length = block_size, win_length = n_fft,center = True, power=1)

    
    def torch_A_weighting(self, FREQUENCIES, min_db = -80.0):
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
            WEIGHTS_IN_DB = torch.max(WEIGHTS_IN_DB.to(self.device), torch.tensor([min_db], dtype = torch.float32).to(self.device))
        
        # Transform Decibel scale weight to amplitude scale weight.
        weights = torch.exp(torch.log(torch.tensor([10.], dtype = torch.float32).to(self.device)) * WEIGHTS_IN_DB / 10) 
        
        return WEIGHTS_IN_DB


    def extract_loudness_torch(self, signal):

        S = self.speclayer(signal) #(B, F, T)
        S = 20* torch.log(torch.abs(S))

        f_torch = torch.linspace(0, 0.5*self.sr, steps=int(self.n_fft/2)+1).to(self.device)

        a_weight = self.torch_A_weighting(f_torch)

        S = S.to(self.device) + a_weight.reshape(1, -1, 1) #(F,) ==> (1, F, 1) broadcast along batch and dim
        S = torch.mean(S, 1)[..., :-1] #(B, T)

        return S



    def extract_loudness(self, signal):
        S = li.stft(signal,n_fft=self.n_fft,hop_length=self.block_size,win_length=self.n_fft,center=True,)

        S = 20* np.log(abs(S) + 1e-7) #(F, T)

        f = li.fft_frequencies(self.sampling_rate, self.n_fft)
        a_weight = li.A_weighting(f) #(F, )

        S = S + a_weight.reshape(-1, 1) #broadcast along T dim
        S = np.mean(S, 0)[..., :-1]

        S = torch.tensor(S)
        return S
    
    def extract_pitch_torch(self, signal):

        net = CREPE()
        time, f0, confidence, activation = net.predict(
            signal,
            sr=self.sr,
            viterbi=True,
            step_size=int(1000 * self.block_size / self.sr), #int(self.config.frame_resolution * 1000),
            batch_size=32,
        )
        f0 = f0[:, :-1]
        # print("raw input length", f0,"step size", int(1000 * block_size / sampling_rate))

        # if f0.shape[-1] != length:
        #     f0 = np.interp(
        #         np.linspace(0, 1, length, endpoint=False),
        #         np.linspace(0, 1, f0.shape[-1], endpoint=False),
        #         f0,
        #     )
        return f0


        # return 

    def forward(self, pitch, loudness):
        # retrieve pitch and loudness
        # pitch = 
        # pitch = self.extract_pitch_torch(signal)
        # loudness = self.extract_loudness_torch(signal)
        pitch = pitch.float()
        loudness = loudness.float()
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitude_return = amplitudes
        total_amp_return = total_amp
        pitch_return = pitch
        harmonic_return = harmonic_synth(pitch, amplitudes, self.sampling_rate)
        amplitudes *= total_amp
        # print("check,", total_amp.shape, amplitudes.shape,pitch.shape,loudness.shape,  )

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal, amplitude_return, total_amp_return, pitch_return, harmonic_return

    def realtime_forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal

