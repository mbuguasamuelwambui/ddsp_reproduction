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


def generate_violin_dataset(duration_sec: float = 40.0, sample_rate: int = 16000) -> np.ndarray:
    """Generates solo violin audio with vibrato, bow-friction noise, and acoustic body resonances."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = np.zeros_like(t)
    notes = [293.66, 329.63, 349.23, 392.00, 440.00, 523.25, 587.33, 659.25, 783.99]
    np.random.seed(101)
    note_dur = 1.25
    total_notes = int(duration_sec / note_dur)

    for i in range(total_notes):
        f0_base = float(notes[i % len(notes)])
        start_idx = int(i * note_dur * sample_rate)
        end_idx = min(int((i + 1.15) * note_dur * sample_rate), len(audio))
        n_len = end_idx - start_idx

        t_note = np.linspace(0, n_len / sample_rate, n_len, endpoint=False)
        vibrato_depth = np.clip((t_note - 0.2) / 0.5, 0.0, 1.0) * 0.02
        vibrato = 1.0 + vibrato_depth * np.sin(2.0 * np.pi * 5.5 * t_note)
        f0 = f0_base * vibrato
        env = np.sin(np.pi * np.clip(t_note / (n_len / sample_rate), 0.0, 1.0)) ** 0.4

        phase = 2.0 * np.pi * np.cumsum(f0 / sample_rate)
        harmonic_sig = np.zeros(n_len)
        for h in range(1, 24):
            if h * f0_base < sample_rate / 2:
                bridge_resonance = np.exp(-((h * f0_base - 2500.0) / 800.0) ** 2) * 1.5 + 1.0
                amp = (1.0 / (h ** 0.95)) * bridge_resonance
                harmonic_sig += amp * np.sin(h * phase)

        bow_noise = (np.random.randn(n_len) * 0.06) * env
        note_audio = (harmonic_sig * env * 0.7) + bow_noise
        audio[start_idx:end_idx] += note_audio[: end_idx - start_idx]

    audio = audio / (np.max(np.abs(audio)) + 1e-7) * 0.9
    return audio


def generate_flute_dataset(duration_sec: float = 40.0, sample_rate: int = 16000) -> np.ndarray:
    """Generates solo woodwind flute audio with continuous breath modulation."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = np.zeros_like(t)
    melody = [349.23, 392.00, 440.00, 523.25, 587.33, 659.25, 587.33, 523.25]
    note_dur = 1.5
    total_notes = int(duration_sec / note_dur)

    for i in range(total_notes):
        f0_base = melody[i % len(melody)]
        start_idx = int(i * note_dur * sample_rate)
        end_idx = min(int((i + 1) * note_dur * sample_rate), len(audio))
        n_len = end_idx - start_idx

        t_note = np.linspace(0, n_len / sample_rate, n_len, endpoint=False)
        vibrato = 1.0 + 0.015 * np.sin(2.0 * np.pi * 5.0 * t_note)
        f0 = f0_base * vibrato

        env = np.sin(np.pi * np.clip(t_note / (n_len / sample_rate), 0.0, 1.0)) ** 0.5
        phase = 2.0 * np.pi * np.cumsum(f0 / sample_rate)
        harmonics = (
            1.0 * np.sin(phase)
            + 0.6 * np.sin(2 * phase)
            + 0.25 * np.sin(3 * phase)
            + 0.1 * np.sin(4 * phase)
        )
        breath = (np.random.randn(n_len) * 0.09) * env
        audio[start_idx:end_idx] = (harmonics * env) + breath

    return audio / (np.max(np.abs(audio)) + 1e-7) * 0.9


def generate_kalimba_dataset(duration_sec: float = 40.0, sample_rate: int = 16000) -> np.ndarray:
    """Generates traditional African Kalimba / Mbira plucked acoustic recordings."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = np.zeros_like(t)
    scale = [220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33]
    np.random.seed(42)
    note_dur = 0.6
    total_notes = int(duration_sec / note_dur)

    for i in range(total_notes):
        f0 = float(np.random.choice(scale))
        start_idx = int(i * note_dur * sample_rate)
        end_idx = min(int((i + 1.4) * note_dur * sample_rate), len(audio))
        n_len = end_idx - start_idx

        t_note = np.linspace(0, n_len / sample_rate, n_len, endpoint=False)
        harm_sig = np.zeros(n_len)
        for h in range(1, 14):
            decay = np.exp(-t_note * (2.8 + 1.1 * h))
            amp = (1.0 / (h ** 1.35)) * decay
            harm_sig += amp * np.sin(2.0 * np.pi * (h * f0) * t_note)

        pluck = (np.random.rand(n_len) * 2.0 - 1.0) * np.exp(-t_note * 50.0) * 0.45
        audio[start_idx:end_idx] += (harm_sig + pluck)[: end_idx - start_idx]

    return audio / (np.max(np.abs(audio)) + 1e-7) * 0.9


def generate_vocal_inputs(sample_rate: int = 16000) -> dict:
    """Generates realistic singing voice inputs for Timbre Transfer and benchmarks."""
    duration_sec = 8.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    f0_curve = 220.0 + 90.0 * np.sin(2.0 * np.pi * 0.22 * t) + 15.0 * np.sin(2.0 * np.pi * 5.0 * t)
    phase = 2.0 * np.pi * np.cumsum(f0_curve / sample_rate)

    vocal_harm = (
        0.9 * np.sin(phase)
        + 0.7 * np.sin(2 * phase)
        + 0.45 * np.sin(3 * phase)
        + 0.3 * np.sin(4 * phase)
        + 0.2 * np.sin(5 * phase)
    )
    dynamics = 0.4 + 0.6 * (np.sin(2.0 * np.pi * 0.3 * t) ** 2)
    vocal_melody = (vocal_harm * dynamics)
    vocal_melody = vocal_melody / (np.max(np.abs(vocal_melody)) + 1e-7) * 0.9

    return {"hummed_voice": vocal_melody}


def setup_all_datasets(data_dir: str = "data"):
    """Populates data/ with full multi-instrument datasets."""
    train_dir = os.path.join(data_dir, "instrument_train")
    test_dir = os.path.join(data_dir, "test_inputs")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    violin = generate_violin_dataset(duration_sec=48.0)
    flute = generate_flute_dataset(duration_sec=48.0)
    kalimba = generate_kalimba_dataset(duration_sec=48.0)
    vocals = generate_vocal_inputs()

    sf.write(os.path.join(train_dir, "violin_solo.wav"), violin, 16000)
    sf.write(os.path.join(train_dir, "flute_solo.wav"), flute, 16000)
    sf.write(os.path.join(train_dir, "kalimba_solo.wav"), kalimba, 16000)
    sf.write(os.path.join(test_dir, "hummed_voice_input.wav"), vocals["hummed_voice"], 16000)

