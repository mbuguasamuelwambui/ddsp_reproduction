"""
Differentiable Digital Signal Processing (DDSP) Primitives in PyTorch
Includes:
- Harmonic Additive Synthesizer (sinusoidal bank with anti-aliasing)
- Filtered Noise Synthesizer (time-varying FIR filter over white noise)
- Differentiable Reverb (trainable FIR room impulse response)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_frames(frame_tensor: torch.Tensor, target_samples: int) -> torch.Tensor:
    """
    Linearly interpolates frame-rate control signals [B, T_frames, C] 
    up to audio sample-rate [B, T_samples, C].
    """
    # Permute to [B, C, T_frames] for 1D linear interpolation
    x = frame_tensor.transpose(1, 2)
    x_upsampled = F.interpolate(x, size=target_samples, mode="linear", align_corners=True)
    return x_upsampled.transpose(1, 2)


class HarmonicOscillator(nn.Module):
    """
    Differentiable Harmonic Additive Synthesizer.
    Generates audio by summing harmonically-related sinusoids:
        y(t) = sum_{k=1}^K A_k(t) * sin(phi_k(t))
    where phi_k(t) = 2 * pi * cumsum(k * f0(t) / sr).
    Includes anti-aliasing masking to zero out harmonics above the Nyquist limit.
    """

    def __init__(self, sample_rate: int = 16000, n_harmonics: int = 64):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics

    def forward(
        self,
        f0_hz: torch.Tensor,
        harmonic_amplitudes: torch.Tensor,
        target_samples: int,
    ) -> torch.Tensor:
        """
        Args:
            f0_hz: [Batch, T_frames, 1] Fundamental frequency in Hz
            harmonic_amplitudes: [Batch, T_frames, n_harmonics] Harmonic amplitudes
            target_samples: Total number of audio samples in the output waveform

        Returns:
            audio: [Batch, target_samples] Synthesized harmonic waveform
        """
        batch_size = f0_hz.shape[0]

        # 1. Upsample frame-rate control signals to audio sample-rate
        f0_up = upsample_frames(f0_hz, target_samples)  # [B, T_samples, 1]
        amps_up = upsample_frames(harmonic_amplitudes, target_samples)  # [B, T_samples, K]

        # 2. Compute harmonic frequencies f_k(t) = k * f0(t)
        # Harmonic indices: [1, 2, ..., K]
        harmonics = torch.arange(
            1, self.n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype
        ).view(1, 1, -1)  # [1, 1, K]

        harmonic_freqs = f0_up * harmonics  # [B, T_samples, K]

        # 3. Anti-Aliasing: Zero out harmonics above Nyquist frequency (sample_rate / 2)
        nyquist = self.sample_rate / 2.0
        anti_aliasing_mask = (harmonic_freqs < nyquist).float()
        active_amps = amps_up * anti_aliasing_mask  # [B, T_samples, K]

        # 4. Integrate frequency to compute instantaneous phase phi(t)
        # phi_k(t) = 2 * pi * integral(f_k(tau) d tau)
        phase_increment = 2.0 * math.pi * (harmonic_freqs / float(self.sample_rate))
        phase = torch.cumsum(phase_increment, dim=1) % (2.0 * math.pi)

        # 5. Sinusoidal Summation
        # y(t) = sum_k A_k(t) * sin(phi_k(t))
        sinusoids = torch.sin(phase)
        harmonic_audio = torch.sum(active_amps * sinusoids, dim=-1)  # [B, T_samples]

        return harmonic_audio


class FilteredNoiseSynthesizer(nn.Module):
    """
    Differentiable Filtered Noise Synthesizer.
    Generates time-varying colored/filtered noise by convolving white noise
    with dynamic transfer function frequency responses.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter_banks: int = 65,
        hop_length: int = 64,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filter_banks = n_filter_banks
        self.hop_length = hop_length
        self.n_fft = (n_filter_banks - 1) * 2  # e.g., (65 - 1) * 2 = 128
        self.win_length = self.n_fft
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(
        self,
        noise_magnitudes: torch.Tensor,
        target_samples: int,
    ) -> torch.Tensor:
        """
        Args:
            noise_magnitudes: [Batch, T_frames, n_filter_banks] Linear transfer function magnitudes
            target_samples: Total number of audio samples

        Returns:
            audio: [Batch, target_samples] Synthesized noise waveform
        """
        batch_size = noise_magnitudes.shape[0]

        # 1. Generate uniform / gaussian white noise [-1, 1]
        noise = (
            torch.rand(
                batch_size,
                target_samples,
                device=noise_magnitudes.device,
                dtype=noise_magnitudes.dtype,
            )
            * 2.0
            - 1.0
        )

        # 2. Compute STFT of the white noise
        window = self.window.to(device=noise.device, dtype=noise.dtype)
        noise_stft = torch.stft(
            noise,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )  # [B, n_filter_banks, T_stft]

        # 3. Align predicted filter magnitudes with STFT time dimension
        h_mags = noise_magnitudes.transpose(1, 2)
        target_stft_frames = noise_stft.shape[-1]

        h_mags_interp = F.interpolate(
            h_mags, size=target_stft_frames, mode="linear", align_corners=True
        )  # [B, n_filter_banks, T_stft]

        # 4. Filter in frequency domain
        filtered_stft = noise_stft * h_mags_interp

        # 5. Inverse STFT
        filtered_audio = torch.istft(
            filtered_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=target_samples,
        )

        return filtered_audio


class DifferentiableReverb(nn.Module):
    """
    Trainable FIR Reverb Module.
    Convolves dry synthetic audio with a learnable impulse response.
    """

    def __init__(self, reverb_length: int = 4000):
        super().__init__()
        self.reverb_length = reverb_length
        # Initialize with exponentially decaying white noise
        decay = torch.exp(-torch.linspace(0, 5, reverb_length))
        ir = (torch.rand(reverb_length) * 2.0 - 1.0) * decay
        self.ir = nn.Parameter(ir)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [Batch, T_samples] Dry audio signal
        Returns:
            reverberant_audio: [Batch, T_samples] Wet reverberant audio
        """
        batch_size, num_samples = audio.shape
        # Pad for valid/causal convolution
        ir_norm = self.ir / (torch.norm(self.ir) + 1e-7)
        ir_kernel = ir_norm.view(1, 1, -1)  # [1, 1, L_ir]

        audio_padded = F.pad(audio.unsqueeze(1), (self.reverb_length - 1, 0), mode="constant")
        wet = F.conv1d(audio_padded, ir_kernel).squeeze(1)

        # Mix dry + wet
        return 0.7 * audio + 0.3 * wet[:, :num_samples]
