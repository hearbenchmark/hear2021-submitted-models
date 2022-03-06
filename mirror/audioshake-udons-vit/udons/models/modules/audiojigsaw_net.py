import pathlib
from torch import nn
import torch
from einops import rearrange
from udons.models.modules import utils


class AlexNetJigsaw(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super(AlexNetJigsaw, self).__init__()
        self.patch_model = nn.Sequential(
            nn.Conv2d(hparams["nb_channels"], 64, kernel_size=7, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Flatten(1),
            nn.Dropout(),
            nn.Linear(7936, 512),
        )
        self.unpatch = utils.SiameseConcatView(hparams["nb_patches"])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * hparams["nb_patches"], 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, hparams["nb_classes"]),
        )

    def forward(self, patched_x: torch.Tensor) -> torch.Tensor:
        """Forward pass of sigsaw model

        input args: (N * num_towers) x C x H x W
        output: (N x num_casses)
        """
        patched_x = self.patch_model(patched_x)
        x = self.unpatch(patched_x)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear


class LSTMJigsaw(nn.Module):
    def __init__(self, hparams: dict):
        super(LSTMJigsaw, self).__init__()
        print(hparams["dims"])
        self.hidden_size = hparams['hidden_size']
        self.fc1 = Linear(hparams['n_mels'], self.hidden_size)

        self.bn1 = BatchNorm1d(self.hidden_size)
        lstm_hidden_size = self.hidden_size // 2

        self.lstm = LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=hparams['nb_layers'],
            bidirectional=True,
            batch_first=True
        )


        self.fc2 = Linear(
            in_features=lstm_hidden_size, out_features=self.hidden_size
        )

        self.bn2 = BatchNorm1d(self.hidden_size)

        self.unpatch = utils.SiameseConcatView(hparams["nb_patches"])

        self.fc3 = Linear(
            in_features=self.hidden_size * hparams["nb_patches"],
            out_features=hparams["nb_classes"],
        )


    def forward(self, x: Tensor) -> Tensor:
        # get current spectrogram shape
        # b=batch_size, nb_channels=1, f=features, t=time steps
        nb_b, nb_c, nb_f, nb_t  = x.data.shape
        
        x = torch.mean(x, dim=1)
        x = self.fc1(rearrange(x, 'b f t -> (b t) f'))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = rearrange(x, '(b t) h -> b t h', b=nb_b)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out, _ = self.lstm(x)

        # sum state vector of last/first hidden state from reverse/forward direction
        last_state = lstm_out[:, -1, :self.hidden_size // 2] + lstm_out[:, -1, self.hidden_size // 2:]

        x = self.fc2(last_state)
        x = self.bn2(x)

        x = F.relu(x)
        x = rearrange(x, '(b t) h -> b (t h)', b=nb_b)

        x = self.unpatch(x)

        x = self.fc3(x)
        return x

if __name__ == "__main__":
    from torch.autograd import Variable
    from torchvision import datasets, transforms

    # test
    hparams = {}
    hparams["nb_patches"] = 5
    hparams["nb_classes"] = 1000
    hparams["hidden_size"] = 128
    hparams["n_mels"] = 128
    hparams["nb_channels"] = 1
    hparams["nb_layers"] = 1
    model = LSTMJigsaw(hparams)
    # patch size divisionable by 32!
    x = torch.rand(5 * 16, 128, 32)
    y = model(x)
    print(y.shape)
