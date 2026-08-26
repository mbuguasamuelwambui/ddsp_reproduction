"""
DDSP Neural Decoder Architecture in PyTorch
Converts pitch (F0) and loudness contours into synthesis parameters for the DSP engines.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsp import HarmonicOscillator, FilteredNoiseSynthesizer, DifferentiableReverb


def modified_sigmoid(x: torch.Tensor, exponent: float = 10.0, max_val: float = 2.0) -> torch.Tensor:
    """
    Stabilized modified sigmoid activation: max_val * sigmoid(x)**(log(10))
    Used by DDSP to produce smooth positive amplitudes and filter gains.
    """
    return max_val * (torch.sigmoid(x) ** math.log(exponent))


class MLP(nn.Module):
    """Multi-Layer Perceptron block with LayerNorm and ReLU/ELU."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DDSPAutoencoder(nn.Module):
    """
    DDSP Neural Synthesizer (Decoder).
    
    Inputs:
        f0_hz: Fundamental frequency contour [Batch, T_frames, 1]
        loudness_db: A-weighted loudness contour [Batch, T_frames, 1]
        
    Outputs:
        audio: Synthesized audio waveform [Batch, T_samples]
        components: Dictionary containing harmonic, noise, and reverb components.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_rate: int = 250,  # e.g., 64 samples per frame at 16kHz
        n_harmonics: int = 64,
        n_filter_banks: int = 65,
        hidden_dim: int = 256,
        gru_layers: int = 1,
        reverb_length: int = 4000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.n_harmonics = n_harmonics
        self.n_filter_banks = n_filter_banks

        # 1. Conditioning Encoders
        # F0 is mapped from Hz to normalized scale
        self.f0_mlp = MLP(in_dim=1, hidden_dim=128, out_dim=128, num_layers=2)
        # Loudness is mapped from normalized dB
        self.loudness_mlp = MLP(in_dim=1, hidden_dim=128, out_dim=128, num_layers=2)

        # 2. Sequence Processor (GRU)
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        # 3. Output Projection Heads
        # Harmonic Controls: overall amplitude (1) + harmonic distribution (n_harmonics)
        self.harmonic_amp_head = nn.Linear(hidden_dim, 1)
        self.harmonic_dist_head = nn.Linear(hidden_dim, n_harmonics)

        # Filtered Noise Controls: filter bank transfer function magnitudes
        self.noise_head = nn.Linear(hidden_dim, n_filter_banks)

        # 4. Differentiable DSP Synthesizer Modules
        self.harmonic_synth = HarmonicOscillator(
            sample_rate=sample_rate, n_harmonics=n_harmonics
        )
        self.noise_synth = FilteredNoiseSynthesizer(
            sample_rate=sample_rate, n_filter_banks=n_filter_banks
        )
        self.reverb = DifferentiableReverb(reverb_length=reverb_length)

    def normalize_f0(self, f0_hz: torch.Tensor) -> torch.Tensor:
        """Converts Hz to normalized continuous scale [0, 1]."""
        # Clamp minimum to 0 Hz, use MIDI scale: 69 + 12*log2(f0/440)
        f0_safe = torch.clamp(f0_hz, min=1.0)
        midi = 69.0 + 12.0 * torch.log2(f0_safe / 440.0)
        # Map MIDI pitch range [0, 127] to [0, 1]
        norm_pitch = torch.clamp(midi / 127.0, min=0.0, max=1.0)
        # Zero out unvoiced sections where f0 == 0
        voiced_mask = (f0_hz > 10.0).float()
        return norm_pitch * voiced_mask

    def normalize_loudness(self, loudness_db: torch.Tensor) -> torch.Tensor:
        """Maps loudness in dB [-120, 0] to [0, 1]."""
        return torch.clamp((loudness_db + 120.0) / 120.0, min=0.0, max=1.0)

    def forward(
        self,
        f0_hz: torch.Tensor,
        loudness_db: torch.Tensor,
        target_samples: int,
    ) -> dict:
        """
        Args:
            f0_hz: [Batch, T_frames, 1] in Hz
            loudness_db: [Batch, T_frames, 1] in dB
            target_samples: Number of audio samples to generate

        Returns:
            Dictionary with keys:
                'audio': Final reverberant synthesized audio [Batch, target_samples]
                'harmonic_audio': Dry harmonic audio [Batch, target_samples]
                'noise_audio': Dry filtered noise audio [Batch, target_samples]
                'dry_audio': Dry combined audio [Batch, target_samples]
                'harmonic_amplitudes': Predicted harmonic amplitudes [B, T_frames, K]
                'noise_magnitudes': Predicted noise filter response [B, T_frames, n_filter_banks]
        """
        # 1. Normalize features
        f0_norm = self.normalize_f0(f0_hz)
        loudness_norm = self.normalize_loudness(loudness_db)

        # 2. Encode conditioning features
        f0_emb = self.f0_mlp(f0_norm)
        ld_emb = self.loudness_mlp(loudness_norm)
        cond = torch.cat([f0_emb, ld_emb], dim=-1)  # [B, T_frames, 256]

        # 3. Temporal Processing
        gru_out, _ = self.gru(cond)  # [B, T_frames, hidden_dim]

        # 4. Predict DSP Parameters
        # Overall Harmonic Amplitude A(t)
        total_amp = modified_sigmoid(self.harmonic_amp_head(gru_out))  # [B, T_frames, 1]
        
        # Voicing gate: zero out harmonic amp if f0 <= 10 Hz
        voiced_gate = (f0_hz > 10.0).float()
        total_amp = total_amp * voiced_gate

        # Harmonic distribution c_k(t) with softmax
        harm_dist = F.softmax(self.harmonic_dist_head(gru_out), dim=-1)  # [B, T_frames, K]
        harmonic_amplitudes = total_amp * harm_dist  # [B, T_frames, K]

        # Noise filter frequency magnitudes H(f)
        noise_magnitudes = modified_sigmoid(self.noise_head(gru_out))  # [B, T_frames, filter_banks]

        # 5. Differentiable Synthesis
        harmonic_audio = self.harmonic_synth(f0_hz, harmonic_amplitudes, target_samples)
        noise_audio = self.noise_synth(noise_magnitudes, target_samples)

        # Combine dry audio
        dry_audio = harmonic_audio + noise_audio

        # Apply differentiable reverb
        reverb_audio = self.reverb(dry_audio)

        return {
            "audio": reverb_audio,
            "dry_audio": dry_audio,
            "harmonic_audio": harmonic_audio,
            "noise_audio": noise_audio,
            "harmonic_amplitudes": harmonic_amplitudes,
            "noise_magnitudes": noise_magnitudes,
        }
