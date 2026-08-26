"""
DDSP Core Package - Differentiable Digital Signal Processing in PyTorch
Based on Engel et al., ICLR 2020 (https://github.com/magenta/ddsp)
"""

from .dsp import HarmonicOscillator, FilteredNoiseSynthesizer, DifferentiableReverb
from .model import DDSPAutoencoder
from .loss import MultiScaleSpectralLoss
from .data import AudioFeatureExtractor, MonophonicAudioDataset, create_dataloader
from .metrics import DDPSEvaluator
from .baselines import ClassicalSourceFilterVocoder, HarmonicOnlyDDSP, NoiseOnlyDDSP
from .experiments import (
    run_modular_decomposition,
    run_pitch_extrapolation,
    run_dereverberation_and_acoustic_transfer,
    run_independent_control_interpolation,
    demonstrate_phase_invariance,
)
from .utils import (
    save_audio,
    compute_linear_log_mag_spectrogram,
    plot_spectrogram_comparison,
    plot_modular_decomposition_grid,
)

__all__ = [
    "HarmonicOscillator",
    "FilteredNoiseSynthesizer",
    "DifferentiableReverb",
    "DDSPAutoencoder",
    "MultiScaleSpectralLoss",
    "AudioFeatureExtractor",
    "MonophonicAudioDataset",
    "create_dataloader",
    "DDPSEvaluator",
    "ClassicalSourceFilterVocoder",
    "HarmonicOnlyDDSP",
    "NoiseOnlyDDSP",
    "run_modular_decomposition",
    "run_pitch_extrapolation",
    "run_dereverberation_and_acoustic_transfer",
    "run_independent_control_interpolation",
    "demonstrate_phase_invariance",
    "save_audio",
    "compute_linear_log_mag_spectrogram",
    "plot_spectrogram_comparison",
    "plot_modular_decomposition_grid",
]
