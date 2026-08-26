"""
DDSP Training Script
Trains the Differentiable Digital Signal Processing autoencoder on monophonic audio clips.
"""

import os
import glob
import argparse
import time
import torch
import torch.optim as optim
from tqdm import tqdm

from src.model import DDSPAutoencoder
from src.loss import MultiScaleSpectralLoss
from src.data import MonophonicAudioDataset, create_dataloader
from src.utils import save_audio, plot_spectrogram_comparison


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DDSP Training] Using device: {device}")

    # 1. Discover audio files
    audio_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    if not audio_files:
        raise FileNotFoundError(
            f"No .wav files found in {args.data_dir}. Run download_sample_data.py first!"
        )
    print(f"Found {len(audio_files)} audio file(s) for training.")

    # 2. Build Dataset & DataLoader
    print("Extracting features (F0 and A-weighted Loudness)...")
    dataset = MonophonicAudioDataset(
        audio_files=audio_files,
        sample_rate=args.sample_rate,
        chunk_duration_sec=args.chunk_sec,
    )
    print(f"Total {args.chunk_sec}s training chunks: {len(dataset)}")

    dataloader = create_dataloader(
        audio_files=audio_files,
        batch_size=args.batch_size,
        shuffle=True,
        sample_rate=args.sample_rate,
        chunk_duration_sec=args.chunk_sec,
    )

    # 3. Model, Loss, Optimizer
    model = DDSPAutoencoder(
        sample_rate=args.sample_rate,
        n_harmonics=args.n_harmonics,
        n_filter_banks=args.n_filter_banks,
        hidden_dim=args.hidden_dim,
    ).to(device)

    criterion = MultiScaleSpectralLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nStarting Training...")
    step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            audio = batch["audio"].to(device)  # [B, T_samples]
            f0 = batch["f0"].to(device)        # [B, T_frames, 1]
            loudness = batch["loudness"].to(device)  # [B, T_frames, 1]

            target_samples = audio.shape[-1]

            optimizer.zero_grad()
            outputs = model(f0, loudness, target_samples)
            pred_audio = outputs["audio"]

            loss = criterion(audio, pred_audio)
            loss.backward()

            # Gradient clipping for numerical stability in phase accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} Completed | Average Multi-Scale Loss: {avg_loss:.4f}")

        # Save checkpoint periodically
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ddsp_model_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

            # Generate sample resynthesis demonstration
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(dataloader))
                t_audio = test_batch["audio"].to(device)
                t_f0 = test_batch["f0"].to(device)
                t_ld = test_batch["loudness"].to(device)
                t_out = model(t_f0, t_ld, t_audio.shape[-1])

                save_audio(
                    os.path.join(args.output_dir, f"resynth_target_ep{epoch}.wav"),
                    t_audio[0],
                    sample_rate=args.sample_rate,
                )
                save_audio(
                    os.path.join(args.output_dir, f"resynth_ddsp_ep{epoch}.wav"),
                    t_out["audio"][0],
                    sample_rate=args.sample_rate,
                )
                plot_spectrogram_comparison(
                    t_audio[0],
                    t_out["audio"][0],
                    sample_rate=args.sample_rate,
                    title=f"Epoch {epoch}",
                    save_path=os.path.join(args.output_dir, f"spectrogram_ep{epoch}.png"),
                )

    print(f"\nTraining Complete in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDSP PyTorch Model")
    parser.add_argument("--data_dir", type=str, default="data/instrument_train", help="Path to training .wav files")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for audio/plot outputs")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate (Hz)")
    parser.add_argument("--chunk_sec", type=float, default=4.0, help="Training chunk duration in seconds")
    parser.add_argument("--n_harmonics", type=int, default=64, help="Number of additive sinusoidal harmonics")
    parser.add_argument("--n_filter_banks", type=int, default=65, help="Number of noise filter frequency bands")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden state dimension of GRU")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=10, help="Checkpoint save frequency")

    args = parser.parse_args()
    train(args)
