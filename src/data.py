"""
Audio Feature Extraction and Dataset Utilities for DDSP.
Extracts fundamental frequency (F0) and A-weighted loudness contours.
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


class AudioFeatureExtractor:
    """
    Extracts frame-level F0 (pitch) in Hz and A-weighted Loudness in dB
    from raw 16 kHz audio waveforms.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_rate: int = 250,  # 250 frames/sec (hop_size = 64 samples)
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
        # Avoid log(0)
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
        # Clip minimum floor
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
        audio_cropped = audio[:target_samples].astype(np.float32)

        return audio_cropped, f0, loudness


class MonophonicAudioDataset(Dataset):
    """
    PyTorch Dataset for DDSP training.
    Loads audio files, extracts or caches (F0, Loudness, Audio) chunks.
    """

    def __init__(
        self,
        audio_files: List[str],
        sample_rate: int = 16000,
        frame_rate: int = 250,
        chunk_duration_sec: float = 4.0,
    ):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.chunk_samples = int(chunk_duration_sec * sample_rate)
        self.extractor = AudioFeatureExtractor(
            sample_rate=sample_rate, frame_rate=frame_rate
        )

        self.samples = []
        self._load_and_segment(audio_files)

    def _load_and_segment(self, audio_files: List[str]):
        """Loads and segments long recordings into fixed-duration chunks."""
        for path in audio_files:
            if not os.path.exists(path):
                continue
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            # Normalize peak
            peak = np.max(np.abs(audio))
            if peak > 1e-4:
                audio = audio / peak

            total_samples = len(audio)
            step = self.chunk_samples

            for start in range(0, total_samples - step + 1, step):
                chunk = audio[start : start + step]
                cropped_audio, f0, loudness = self.extractor.process_audio(chunk)
                self.samples.append({
                    "audio": torch.from_numpy(cropped_audio),
                    "f0": torch.from_numpy(f0),
                    "loudness": torch.from_numpy(loudness),
                })

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
    dataset = MonophonicAudioDataset(
        audio_files=audio_files,
        sample_rate=sample_rate,
        chunk_duration_sec=chunk_duration_sec,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
