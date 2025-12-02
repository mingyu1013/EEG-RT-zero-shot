# models_eegnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetV4Encoder(nn.Module):
    """
    EEGNet v4 encoder.
    입력: (B, C, T) 또는 (B, 1, C, T)
    출력: (B, F)
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times

        # 1) temporal conv
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # 2) depthwise spatial conv
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_chans, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # 3) separable conv
        self.sep_conv1 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 16),
            padding=(0, 8),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2))
        self.drop2 = nn.Dropout(dropout)

        # flatten dimension 계산
        with torch.no_grad():
            x = torch.zeros(1, 1, n_chans, n_times)
            feats = self._forward_features(x)
            self.flat_size = feats.shape[1]

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_conv1(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        return self._forward_features(x)


class EEGNetRegressorMLP(nn.Module):
    """
    EEGNet encoder 위에 MLP head를 붙인 RT 회귀 모델.
    """

    def __init__(self, encoder: EEGNetV4Encoder, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(encoder.flat_size, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        out = self.mlp(feats)
        return out.squeeze(-1)
