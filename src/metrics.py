"""
Evaluation Metrics for DDSP based on Engel et al. (ICLR 2020).
Includes:
- Multi-Scale Spectral Loss (MSS) across 6 resolutions
- Pitch Tracking RMSE (in Cents)
- Loudness Reconstruction Error (L1 dB)
- Harmonic-to-Noise Ratio (HNR) Consistency
"""

import math
import numpy as np
import torch
import torch.nn as nn
import librosa

from .loss import MultiScaleSpectralLoss
from .data import AudioFeatureExtractor


def compute_f0_cents_error(
    f0_target_hz: np.ndarray,
    f0_pred_hz: np.ndarray,
    f_ref: float = 10.0,
    voiced_threshold: float = 10.0,
) -> dict:
    """
    Computes pitch error in Cents on voiced frames:
        cents = 1200 * log2(f0 / f_ref)
    """
    f0_tgt = np.squeeze(f0_target_hz)
    f0_prd = np.squeeze(f0_pred_hz)

    # Consider only voiced frames in both
    voiced_mask = (f0_tgt > voiced_threshold) & (f0_prd > voiced_threshold)
    if not np.any(voiced_mask):
        return {"cents_rmse": 0.0, "cents_mae": 0.0, "voiced_frame_ratio": 0.0}

    cents_tgt = 1200.0 * np.log2(f0_tgt[voiced_mask] / f_ref)
    cents_prd = 1200.0 * np.log2(f0_prd[voiced_mask] / f_ref)

    diff = np.abs(cents_tgt - cents_prd)
    cents_mae = float(np.mean(diff))
    cents_rmse = float(np.sqrt(np.mean(diff ** 2)))
    voiced_ratio = float(np.mean(voiced_mask))

    return {
        "cents_rmse": cents_rmse,
        "cents_mae": cents_mae,
        "voiced_frame_ratio": voiced_ratio,
    }


def compute_loudness_error_db(
    loudness_target_db: np.ndarray, loudness_pred_db: np.ndarray
) -> dict:
    """Computes L1 and RMSE error between target and synthesized loudness contours in dB."""
    ld_tgt = np.squeeze(loudness_target_db)
    ld_prd = np.squeeze(loudness_pred_db)

    diff = np.abs(ld_tgt - ld_prd)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return {"loudness_mae_db": mae, "loudness_rmse_db": rmse}


def compute_harmonic_noise_ratio(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """Estimates the Harmonic-to-Noise Ratio (HNR in dB) using autocorrelation."""
    autocorr = librosa.autocorrelate(audio)
    if len(autocorr) < 2 or autocorr[0] <= 1e-7:
        return 0.0
    r0 = autocorr[0]
    # Peak beyond lag 16 (corresponding to max 1000 Hz)
    r_peak = np.max(autocorr[16:]) if len(autocorr) > 16 else 1e-7
    r_peak = max(r_peak, 1e-7)
    hnr = 10.0 * np.log10(r_peak / max(1e-7, (r0 - r_peak)))
    return float(np.clip(hnr, -50.0, 50.0))


class DDPSEvaluator:
    """Full paper benchmark evaluation suite."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.extractor = AudioFeatureExtractor(sample_rate=sample_rate, frame_rate=250)
        self.spectral_loss_fn = MultiScaleSpectralLoss(
            fft_sizes=(2048, 1024, 512, 256, 128, 64), alpha_log=1.0
        )

    def evaluate_pair(
        self, target_audio: np.ndarray, synth_audio: np.ndarray
    ) -> dict:
        """Evaluates a single pair of ground truth vs synthesized audio."""
        min_len = min(len(target_audio), len(synth_audio))
        target_audio = target_audio[:min_len].astype(np.float32)
        synth_audio = synth_audio[:min_len].astype(np.float32)

        # 1. Multi-Scale Spectral Loss (MSS)
        t_tgt = torch.from_numpy(target_audio).unsqueeze(0)
        t_syn = torch.from_numpy(synth_audio).unsqueeze(0)
        with torch.no_grad():
            mss_loss = float(self.spectral_loss_fn(t_tgt, t_syn).item())

        # 2. Extract features from both
        _, f0_tgt, ld_tgt = self.extractor.process_audio(target_audio)
        _, f0_syn, ld_syn = self.extractor.process_audio(synth_audio)

        pitch_metrics = compute_f0_cents_error(f0_tgt, f0_syn)
        loudness_metrics = compute_loudness_error_db(ld_tgt, ld_syn)

        # 3. HNR Consistency
        hnr_tgt = compute_harmonic_noise_ratio(target_audio, self.sample_rate)
        hnr_syn = compute_harmonic_noise_ratio(synth_audio, self.sample_rate)
        hnr_diff = abs(hnr_tgt - hnr_syn)

        return {
            "multi_scale_spectral_loss": mss_loss,
            "f0_cents_rmse": pitch_metrics["cents_rmse"],
            "f0_cents_mae": pitch_metrics["cents_mae"],
            "loudness_mae_db": loudness_metrics["loudness_mae_db"],
            "loudness_rmse_db": loudness_metrics["loudness_rmse_db"],
            "target_hnr_db": hnr_tgt,
            "synth_hnr_db": hnr_syn,
            "hnr_abs_diff_db": hnr_diff,
        }
