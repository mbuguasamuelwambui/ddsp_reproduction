"""
Audio Feature Extraction and Dataset Utilities for DDSP.
Extracts fundamental frequency (F0) and A-weighted loudness contours
from audio recordings.
"""

import os
import math
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader


def load_audio_file(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file from disk, converts to mono, resamples to the target
    sample rate, and normalizes peak amplitude to -1.0 dBFS.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    peak = np.max(np.abs(audio)) + 1e-7
    if peak > 0:
        audio = (audio / peak) * 0.9
    return audio.astype(np.float32), sr


class AudioFeatureExtractor:
    """
    Extracts frame-level F0 (pitch) in Hz and A-weighted Loudness in dB
    from raw 16 kHz audio waveforms.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_rate: int = 250,  # 250 frames/sec (hop_size = 64 samples at 16kHz)
        f0_min: float = 50.0,
        f0_max: float = 1000.0,
    ):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.hop_length = sample_rate // frame_rate  # 64 samples
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts F0 contour using librosa.pyin (Probabilistic YIN).
        Returns array of shape [T_frames, 1] with unvoiced frames set to 0.0 Hz.
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fill_na=0.0,
        )
        # Clean any remaining NaNs
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0.astype(np.float32).reshape(-1, 1)

    def extract_loudness(self, audio: np.ndarray, n_fft: int = 2048) -> np.ndarray:
        """
        Computes A-weighted power in dB across frames.
        Returns array of shape [T_frames, 1].
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
        )
        power_spec = np.abs(stft) ** 2

        # Compute A-weighting curve across STFT frequency bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
        f_sq = freqs ** 2
        c1 = 12194.217 ** 2
        c2 = 20.598997 ** 2
        c3 = 107.65265 ** 2
        c4 = 737.86223 ** 2

        num = c1 * (f_sq ** 2)
        den = (f_sq + c2) * np.sqrt((f_sq + c3) * (f_sq + c4)) * (f_sq + c1)
        a_weight = 1.2588966 * (num / (den + 1e-10))
        a_weight = a_weight.reshape(-1, 1)  # [n_bins, 1]

        # Apply weighting and sum across frequency bins
        weighted_power = np.sum(power_spec * a_weight, axis=0) + 1e-10

        # Convert to dB relative to full scale
        loudness_db = 10.0 * np.log10(weighted_power) - 20.0
        loudness_db = np.clip(loudness_db, a_min=-120.0, a_max=0.0)
        return loudness_db.astype(np.float32).reshape(-1, 1)

    def process_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full feature extraction pipeline for an audio array.
        Returns: (audio_cropped, f0, loudness)
        """
        f0 = self.extract_f0(audio)
        loudness = self.extract_loudness(audio)

        # Align lengths
        min_frames = min(len(f0), len(loudness))
        f0 = f0[:min_frames]
        loudness = loudness[:min_frames]

        target_samples = min_frames * self.hop_length
        audio_cropped = audio[:target_samples]

        return audio_cropped.astype(np.float32), f0, loudness


class MonophonicAudioDataset(Dataset):
    """
    PyTorch Dataset for DDSP model training.
    Loads audio files, splits into fixed chunks, and pre-extracts
    F0 and Loudness contours.
    """

    def __init__(
        self,
        audio_files: List[str],
        sample_rate: int = 16000,
        chunk_duration_sec: float = 4.0,
        frame_rate: int = 250,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration_sec)
        self.extractor = AudioFeatureExtractor(sample_rate=sample_rate, frame_rate=frame_rate)
        self.samples = []

        print(f"Loading and processing {len(audio_files)} audio file(s)...")

        for fpath in audio_files:
            try:
                audio, sr = load_audio_file(fpath, sample_rate=sample_rate)
            except Exception as e:
                print(f"Warning: Could not load {fpath}: {e}")
                continue

            total_samples = len(audio)
            step = self.chunk_samples

            if total_samples < step:
                # Pad short audio up to 4s
                pad_len = step - total_samples
                audio = np.pad(audio, (0, pad_len), mode="constant")
                total_samples = len(audio)

            for start in range(0, total_samples - step + 1, step):
                chunk = audio[start : start + step]
                cropped_audio, f0, loudness = self.extractor.process_audio(chunk)
                self.samples.append({
                    "audio": torch.from_numpy(cropped_audio),
                    "f0": torch.from_numpy(f0),
                    "loudness": torch.from_numpy(loudness),
                })

        print(f"Dataset ready: {len(self.samples)} total {chunk_duration_sec}s chunks.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def create_dataloader(
    audio_files: List[str],
    batch_size: int = 4,
    shuffle: bool = True,
    sample_rate: int = 16000,
    chunk_duration_sec: float = 4.0,
) -> DataLoader:
    """Creates a PyTorch DataLoader from real audio files."""
    dataset = MonophonicAudioDataset(
        audio_files=audio_files,
        sample_rate=sample_rate,
        chunk_duration_sec=chunk_duration_sec,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
