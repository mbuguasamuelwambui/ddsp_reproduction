"""
Multi-Scale Spectral Loss for DDSP
Computes multi-resolution STFT magnitude and log-magnitude distances across multiple FFT window sizes.
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleSpectralLoss(nn.Module):
    """
    Computes linear magnitude L1 distance and log-magnitude L1 distance
    for a single STFT configuration (n_fft, hop_length, win_length).
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        log_epsilon: float = 1e-7,
        alpha_log: float = 1.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.log_epsilon = log_epsilon
        self.alpha_log = alpha_log
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, target_audio: torch.Tensor, pred_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target_audio: [Batch, T_samples]
            pred_audio: [Batch, T_samples]
        Returns:
            loss_lin, loss_log
        """
        # Ensure audio has identical length
        min_len = min(target_audio.shape[-1], pred_audio.shape[-1])
        target_audio = target_audio[..., :min_len]
        pred_audio = pred_audio[..., :min_len]

        # Ensure window is on the same device and dtype as the audio
        window = self.window.to(device=target_audio.device, dtype=target_audio.dtype)

        # Compute STFT magnitudes
        target_stft = torch.stft(
            target_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        pred_stft = torch.stft(
            pred_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        target_mag = torch.abs(target_stft)
        pred_mag = torch.abs(pred_stft)

        # 1. Linear Magnitude L1 Loss
        loss_lin = F.l1_loss(pred_mag, target_mag)

        # 2. Log-Magnitude L1 Loss
        target_log = torch.log(target_mag + self.log_epsilon)
        pred_log = torch.log(pred_mag + self.log_epsilon)
        loss_log = F.l1_loss(pred_log, target_log)

        return loss_lin, self.alpha_log * loss_log


class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-Scale Spectral Loss combining multiple STFT scales.
    Default FFT sizes: [2048, 1024, 512, 256, 128, 64].
    """

    def __init__(
        self,
        fft_sizes: List[int] = (2048, 1024, 512, 256, 128, 64),
        alpha_log: float = 1.0,
    ):
        super().__init__()
        self.scales = nn.ModuleList([
            SingleScaleSpectralLoss(
                n_fft=fft_size,
                hop_length=fft_size // 4,
                win_length=fft_size,
                alpha_log=alpha_log,
            )
            for fft_size in fft_sizes
        ])

    def forward(self, target_audio: torch.Tensor, pred_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_audio: [Batch, T_samples] Target ground-truth waveform
            pred_audio: [Batch, T_samples] Synthesized waveform
        Returns:
            total_loss: Scalar multi-scale spectral loss
        """
        total_loss = 0.0
        for scale in self.scales:
            loss_lin, loss_log = scale(target_audio, pred_audio)
            total_loss = total_loss + loss_lin + loss_log
        return total_loss
