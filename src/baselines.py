"""
Baseline and Ablation Models for DDSP Comparative Benchmark.
Based on Engel et al. (ICLR 2020) Section 5:
1. Classical Source-Filter DSP Vocoder (LPC-style filter + pulse/noise excitation)
2. Harmonic-Only DDSP (Ablation without Filtered Noise)
3. Noise-Only DDSP (Ablation without Harmonic Oscillator)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

from .dsp import HarmonicOscillator, FilteredNoiseSynthesizer, DifferentiableReverb
from .model import modified_sigmoid, MLP


class ClassicalSourceFilterVocoder:
    """
    Standard Classical DSP Baseline: Source-Filter Vocoder (without neural network).
    Generates excitation (glottal/harmonic pulse train for voiced, white noise for unvoiced)
    filtered by LPC-estimated spectral envelope.
    """

    def __init__(self, sample_rate: int = 16000, lpc_order: int = 16):
        self.sample_rate = sample_rate
        self.lpc_order = lpc_order

    def synthesize(self, audio_target: np.ndarray, f0_hz: np.ndarray) -> np.ndarray:
        """Synthesizes audio using classical LPC analysis and pulse/noise excitation."""
        audio = audio_target.astype(np.float64)
        f0 = np.squeeze(f0_hz)
        hop_length = 64
        num_frames = len(f0)
        target_len = len(audio)

        # 1. Estimate LPC coefficients across frames
        lpc_coeffs = librosa.lpc(audio, order=self.lpc_order)

        # 2. Construct excitation signal
        excitation = np.zeros(target_len)
        phase = 0.0
        for i in range(target_len):
            frame_idx = min(i // hop_length, num_frames - 1)
            pitch = f0[frame_idx]
            if pitch > 10.0:
                # Periodic impulse train
                phase += pitch / self.sample_rate
                if phase >= 1.0:
                    phase -= 1.0
                    excitation[i] = 1.0
            else:
                # White noise for unvoiced / silence
                excitation[i] = np.random.randn() * 0.1

        # 3. Filter excitation with LPC inverse-filter
        # y[n] = e[n] - sum_{k=1}^P a_k y[n-k]
        synth_audio = np.zeros(target_len)
        a = lpc_coeffs
        for n in range(target_len):
            val = excitation[n]
            for k in range(1, min(len(a), n + 1)):
                val -= a[k] * synth_audio[n - k]
            synth_audio[n] = val

        # Normalize output
        peak = np.max(np.abs(synth_audio) + 1e-7)
        if peak > 0:
            synth_audio = (synth_audio / peak) * np.max(np.abs(audio) + 1e-7)

        return synth_audio.astype(np.float32)


class HarmonicOnlyDDSP(nn.Module):
    """
    DDSP Ablation 1: Harmonic Oscillator Only (No Filtered Noise).
    Demonstrates synthesis degradation when transient and friction noise are omitted.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_harmonics: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics

        self.f0_mlp = MLP(in_dim=1, hidden_dim=128, out_dim=128, num_layers=2)
        self.loudness_mlp = MLP(in_dim=1, hidden_dim=128, out_dim=128, num_layers=2)

        self.gru = nn.GRU(input_size=256, hidden_size=hidden_dim, batch_first=True)
        self.harmonic_amp_head = nn.Linear(hidden_dim, 1)
        self.harmonic_dist_head = nn.Linear(hidden_dim, n_harmonics)

        self.harmonic_synth = HarmonicOscillator(
            sample_rate=sample_rate, n_harmonics=n_harmonics
        )
        self.reverb = DifferentiableReverb()

    def normalize_f0(self, f0_hz: torch.Tensor) -> torch.Tensor:
        f0_safe = torch.clamp(f0_hz, min=1.0)
        midi = 69.0 + 12.0 * torch.log2(f0_safe / 440.0)
        norm_pitch = torch.clamp(midi / 127.0, min=0.0, max=1.0)
        return norm_pitch * (f0_hz > 10.0).float()

    def normalize_loudness(self, loudness_db: torch.Tensor) -> torch.Tensor:
        return torch.clamp((loudness_db + 120.0) / 120.0, min=0.0, max=1.0)

    def forward(self, f0_hz: torch.Tensor, loudness_db: torch.Tensor, target_samples: int) -> dict:
        f0_norm = self.normalize_f0(f0_hz)
        ld_norm = self.normalize_loudness(loudness_db)
        cond = torch.cat([self.f0_mlp(f0_norm), self.loudness_mlp(ld_norm)], dim=-1)

        gru_out, _ = self.gru(cond)
        total_amp = modified_sigmoid(self.harmonic_amp_head(gru_out)) * (f0_hz > 10.0).float()
        harm_dist = F.softmax(self.harmonic_dist_head(gru_out), dim=-1)
        harmonic_amplitudes = total_amp * harm_dist

        dry_audio = self.harmonic_synth(f0_hz, harmonic_amplitudes, target_samples)
        wet_audio = self.reverb(dry_audio)

        return {"audio": wet_audio, "dry_audio": dry_audio}


class NoiseOnlyDDSP(nn.Module):
    """
    DDSP Ablation 2: Filtered Noise Only (No Harmonic Oscillator).
    Demonstrates synthesis degradation when harmonic tonal components are omitted.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter_banks: int = 65,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filter_banks = n_filter_banks

        self.f0_mlp = MLP(in_dim=1, hidden_dim=128, out_dim=128, num_layers=2)
        self.loudness_mlp = MLP(in_dim=1, hidden_dim=128, out_dim=128, num_layers=2)

        self.gru = nn.GRU(input_size=256, hidden_size=hidden_dim, batch_first=True)
        self.noise_head = nn.Linear(hidden_dim, n_filter_banks)

        self.noise_synth = FilteredNoiseSynthesizer(
            sample_rate=sample_rate, n_filter_banks=n_filter_banks
        )
        self.reverb = DifferentiableReverb()

    def normalize_f0(self, f0_hz: torch.Tensor) -> torch.Tensor:
        f0_safe = torch.clamp(f0_hz, min=1.0)
        midi = 69.0 + 12.0 * torch.log2(f0_safe / 440.0)
        return torch.clamp(midi / 127.0, min=0.0, max=1.0) * (f0_hz > 10.0).float()

    def normalize_loudness(self, loudness_db: torch.Tensor) -> torch.Tensor:
        return torch.clamp((loudness_db + 120.0) / 120.0, min=0.0, max=1.0)

    def forward(self, f0_hz: torch.Tensor, loudness_db: torch.Tensor, target_samples: int) -> dict:
        f0_norm = self.normalize_f0(f0_hz)
        ld_norm = self.normalize_loudness(loudness_db)
        cond = torch.cat([self.f0_mlp(f0_norm), self.loudness_mlp(ld_norm)], dim=-1)

        gru_out, _ = self.gru(cond)
        noise_mags = modified_sigmoid(self.noise_head(gru_out))

        dry_audio = self.noise_synth(noise_mags, target_samples)
        wet_audio = self.reverb(dry_audio)

        return {"audio": wet_audio, "dry_audio": dry_audio}
