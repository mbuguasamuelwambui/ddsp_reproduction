"""
DDSP Timbre Transfer Script
Transforms any input audio (e.g., singing voice, whistling, or speech) 
into the target instrument's acoustic timbre using a trained DDSP model.
"""

import os
import sys
import argparse
import torch
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.model import DDSPAutoencoder
    from src.data import AudioFeatureExtractor
    from src.utils import save_audio, plot_spectrogram_comparison
except ImportError:
    from model import DDSPAutoencoder
    from data import AudioFeatureExtractor
    from utils import save_audio, plot_spectrogram_comparison


def pitch_shift_f0(f0_hz: np.ndarray, semitones: float = 0.0, octave_shift: int = 0) -> np.ndarray:
    """
    Shifts the F0 frequency contour in pitch space without changing temporal timing.
    """
    total_semitones = semitones + (octave_shift * 12.0)
    factor = 2.0 ** (total_semitones / 12.0)
    voiced_mask = (f0_hz > 10.0)
    f0_shifted = f0_hz * factor * voiced_mask
    return f0_shifted


def perform_timbre_transfer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Timbre Transfer] Using device: {device}")

    # 1. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}. Please train the model first!")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    model = DDSPAutoencoder(
        sample_rate=config.get("sample_rate", 16000),
        n_harmonics=config.get("n_harmonics", 64),
        n_filter_banks=config.get("n_filter_banks", 65),
        hidden_dim=config.get("hidden_dim", 256),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model successfully loaded!")

    # 2. Load Input Audio
    if not os.path.exists(args.input_audio):
        raise FileNotFoundError(f"Input audio not found: {args.input_audio}")

    print(f"Processing input audio: {args.input_audio}")
    raw_audio, sr = librosa.load(args.input_audio, sr=16000, mono=True)

    # 3. Extract Features
    extractor = AudioFeatureExtractor(sample_rate=16000, frame_rate=250)
    audio_cropped, f0, loudness = extractor.process_audio(raw_audio)

    # Apply Pitch Shift if specified
    if args.pitch_shift != 0.0 or args.octave_shift != 0:
        print(f"Applying pitch shift: {args.pitch_shift} semitones, {args.octave_shift} octaves")
        f0 = pitch_shift_f0(f0, semitones=args.pitch_shift, octave_shift=args.octave_shift)

    # Apply Loudness scaling if specified
    if args.loudness_scale != 1.0:
        loudness = loudness * args.loudness_scale

    # 4. Prepare Tensors
    f0_tensor = torch.from_numpy(f0).unsqueeze(0).to(device)  # [1, T_frames, 1]
    loudness_tensor = torch.from_numpy(loudness).unsqueeze(0).to(device)  # [1, T_frames, 1]
    target_samples = len(audio_cropped)

    # 5. Synthesize in Target Instrument Timbre
    print("Synthesizing audio through DDSP differentiable synthesizer...")
    with torch.no_grad():
        outputs = model(f0_tensor, loudness_tensor, target_samples)

        transferred_audio = outputs["audio"]
        harmonic_only = outputs["harmonic_audio"]
        noise_only = outputs["noise_audio"]

    # 6. Save Outputs
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_audio))[0]

    out_full = os.path.join(args.output_dir, f"{base_name}_timbre_transferred.wav")
    out_harm = os.path.join(args.output_dir, f"{base_name}_harmonic_component.wav")
    out_noise = os.path.join(args.output_dir, f"{base_name}_noise_component.wav")
    out_plot = os.path.join(args.output_dir, f"{base_name}_timbre_transfer_spectrogram.png")

    save_audio(out_full, transferred_audio[0], 16000)
    save_audio(out_harm, harmonic_only[0], 16000)
    save_audio(out_noise, noise_only[0], 16000)

    plot_spectrogram_comparison(
        torch.from_numpy(audio_cropped),
        transferred_audio[0],
        sample_rate=16000,
        title="Input vs Timbre Transferred Output",
        save_path=out_plot,
    )

    print(f"\n[Success] Generated files:")
    print(f"  - Transferred Audio: {out_full}")
    print(f"  - Harmonic Component: {out_harm}")
    print(f"  - Noise Component:    {out_noise}")
    print(f"  - Spectrogram Plot:   {out_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform DDSP Timbre Transfer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pt checkpoint")
    parser.add_argument("--input_audio", type=str, required=True, help="Input .wav to transform")
    parser.add_argument("--output_dir", type=str, default="outputs/timbre_transfer", help="Output directory")
    parser.add_argument("--pitch_shift", type=float, default=0.0, help="Pitch shift in semitones (e.g. +7 or -5)")
    parser.add_argument("--octave_shift", type=int, default=0, help="Octave shift (e.g. +1 or -1)")
    parser.add_argument("--loudness_scale", type=float, default=1.0, help="Loudness multiplier")

    args = parser.parse_args()
    perform_timbre_transfer(args)
