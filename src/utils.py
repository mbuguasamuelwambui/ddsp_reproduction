"""
Plotting, Audio Export, and Linear-Frequency Spectrogram Visualizations for DDSP.
Follows the exact visual styling of Engel et al. (ICLR 2020) and the online supplement:
https://storage.googleapis.com/ddsp/index.html
"""

import os
from typing import Optional, Dict, List, Tuple
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display


def save_audio(file_path: str, audio: torch.Tensor, sample_rate: int = 16000):
    """Saves a 1D or 2D audio tensor to a 16-bit WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().squeeze().numpy()
    else:
        audio_np = np.squeeze(audio)

    # Normalize audio to prevent clipping
    peak = np.max(np.abs(audio_np))
    if peak > 1.0:
        audio_np = audio_np / peak * 0.99

    sf.write(file_path, audio_np, sample_rate)


def compute_linear_log_mag_spectrogram(audio: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Computes linear-frequency log-magnitude spectrogram in dB (matching the DDSP paper)."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann")
    mag = np.abs(stft)
    spec_db = librosa.amplitude_to_db(mag, ref=np.max, top_db=80.0)
    return spec_db


def plot_spectrogram_comparison(
    target_audio: torch.Tensor,
    synth_audio: torch.Tensor,
    sample_rate: int = 16000,
    sr: Optional[int] = None,
    title: str = "DDSP Audio Comparison",
    title_a: Optional[str] = None,
    title_b: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Generates side-by-side linear-frequency log-magnitude spectrogram comparisons."""
    if sr is not None:
        sample_rate = sr

    if isinstance(target_audio, torch.Tensor):
        tgt = target_audio.detach().cpu().squeeze().numpy()
    else:
        tgt = np.squeeze(target_audio)

    if isinstance(synth_audio, torch.Tensor):
        syn = synth_audio.detach().cpu().squeeze().numpy()
    else:
        syn = np.squeeze(synth_audio)

    tgt_db = compute_linear_log_mag_spectrogram(tgt)
    syn_db = compute_linear_log_mag_spectrogram(syn)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    img1 = librosa.display.specshow(
        tgt_db, sr=sample_rate, hop_length=256, x_axis="time", y_axis="linear", ax=axes[0], cmap="magma"
    )
    axes[0].set_title(title_a or f"Target Ground Truth Audio: {title}", fontweight="bold")
    axes[0].set_ylabel("Frequency (Hz)", fontweight="bold")
    axes[0].set_ylim(0, 8000)
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    img2 = librosa.display.specshow(
        syn_db, sr=sample_rate, hop_length=256, x_axis="time", y_axis="linear", ax=axes[1], cmap="magma"
    )
    axes[1].set_title(title_b or "DDSP Synthesized Audio", fontweight="bold")
    axes[1].set_ylabel("Frequency (Hz)", fontweight="bold")
    axes[1].set_xlabel("Time (seconds)", fontweight="bold")
    axes[1].set_ylim(0, 8000)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig


def plot_modular_decomposition_grid(
    decomp_dict: Optional[dict] = None,
    original_audio: Optional[np.ndarray] = None,
    full_synth: Optional[np.ndarray] = None,
    harmonic_synth: Optional[np.ndarray] = None,
    noise_synth: Optional[np.ndarray] = None,
    dry_synth: Optional[np.ndarray] = None,
    f0: Optional[np.ndarray] = None,
    loudness: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    title: str = "DDSP Modular Decomposition of Audio",
    save_path: Optional[str] = None,
):
    """
    Plots the 5-panel Modular Audio Decomposition spectrograms matching:
    https://storage.googleapis.com/ddsp/index.html#modular
    (Original, Resynthesis, Anechoic Dry, Harmonic Only, Filtered Noise Only)
    """
    # Build dictionary from args if not provided directly
    if decomp_dict is None:
        decomp_dict = {}
    if original_audio is not None:
        decomp_dict["original"] = original_audio
    if full_synth is not None:
        decomp_dict["full_synth"] = full_synth
    if dry_synth is not None:
        decomp_dict["dry_synth"] = dry_synth
    elif "dry_synth" not in decomp_dict and harmonic_synth is not None and noise_synth is not None:
        decomp_dict["dry_synth"] = harmonic_synth + noise_synth
    if harmonic_synth is not None:
        decomp_dict["harmonic_synth"] = harmonic_synth
    if noise_synth is not None:
        decomp_dict["noise_synth"] = noise_synth

    keys = [
        ("original", "Target Audio Recording"),
        ("full_synth", "Full DDSP Resynthesis (Harmonic + Noise + Reverb)"),
        ("dry_synth", "Anechoic Dry Audio (Reverb Bypassed)"),
        ("harmonic_synth", "Harmonic Additive Synthesizer (Sinusoidal Bank)"),
        ("noise_synth", "Filtered Noise Synthesizer (Friction / Transients)"),
    ]

    # Filter keys that exist in decomp_dict
    active_keys = [k for k in keys if k[0] in decomp_dict and decomp_dict[k[0]] is not None]
    if not active_keys:
        print("Warning: No audio streams found to plot in plot_modular_decomposition_grid.")
        return None

    fig, axes = plt.subplots(len(active_keys), 1, figsize=(12, 2.3 * len(active_keys)), sharex=True)
    if len(active_keys) == 1:
        axes = [axes]

    for i, (k, label) in enumerate(active_keys):
        audio = decomp_dict[k]
        spec_db = compute_linear_log_mag_spectrogram(audio)
        img = librosa.display.specshow(
            spec_db, sr=sample_rate, hop_length=256, x_axis="time", y_axis="linear", ax=axes[i], cmap="magma"
        )
        axes[i].set_title(label, fontweight="bold", fontsize=11)
        axes[i].set_ylabel("Freq (Hz)", fontweight="bold", fontsize=9)
        axes[i].set_ylim(0, 8000)
        fig.colorbar(img, ax=axes[i], format="%+2.0f dB")

    axes[-1].set_xlabel("Time (seconds)", fontweight="bold", fontsize=10)
    plt.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig
